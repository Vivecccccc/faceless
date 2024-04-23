import math
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from typing import List, Optional, Tuple
from torch.autograd import Variable

from get_nets import PNet, RNet, ONet
from first_stage import run_first_stage
from box_utils import correct_bboxes, nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess

from ...utils.datasets import FramesH5Dataset

# initialize the models
pnet = PNet()
rnet = RNet()
onet = ONet()

def _get_scales(frames: torch.Tensor,
                min_face_size=20.0):
    _, width, height, _ = frames.size()
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707

    scales = []

    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    return scales

def _resize_images(frames: torch.Tensor, scale) -> np.ndarray:
    _, width, height, _ = frames.size()
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    resized_frames = zoom(frames.numpy(), 
                          (1, sw / width, sh / height, 1),
                          order=1)
    resized_frames_list = []
    for frame in resized_frames:
        resized_frames_list.append(_preprocess(frame))
    return np.vstack(resized_frames_list)

def _first_stage_batch(frames: torch.Tensor,
                       scale: float,
                       threshold: float):
    """
    Applies the first stage of face detection to a batch of frames.

    Args:
        frames (torch.Tensor): The input frames as a tensor.
        scale (float): The scale factor for resizing the frames.
        threshold (float): The threshold for face detection.

    Returns:
        list: A list of bounding boxes for each frame. Each element in the list is either an ndarray of shape (num_boxes, 9) or None.
    """
    bounding_boxes = []
    resized_frames = _resize_images(frames, scale)
    outputs = pnet(resized_frames)
    for b, a in outputs:
        probs = a.data.numpy()[1, :, :] # shape: [n, m]
        offsets = b.data.numpy() # shape: [4, n, m]
        boxes = __generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            bounding_boxes.append(None)
            continue
        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        bounding_boxes.append(boxes[keep])
    return bounding_boxes # (num_frames, ndarray(num_boxes, 9) | None)

def __generate_bboxes(probs, offsets, scale, threshold):
    """
    Generate bounding boxes based on probabilities, offsets, scale, and threshold.

    Args:
        probs (numpy.ndarray): Array of probabilities.
        offsets (numpy.ndarray): Array of offsets.
        scale (float): Scale factor.
        threshold (float): Threshold value.

    Returns:
        numpy.ndarray: Array of bounding boxes.
    """
    stride = 2
    cell_size = 12

    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])
    
    tx1, ty1, tx2, ty2 = [offsets[i, inds[0], inds[1]] for i in range(4)]
    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    bounding_boxes = np.vstack([
        np.round((stride * inds[1] + 1.0) / scale),
        np.round((stride * inds[0] + 1.0) / scale),
        np.round((stride * inds[1] + 1.0 + cell_size) / scale),
        np.round((stride * inds[0] + 1.0 + cell_size) / scale),
        score,
        offsets
    ]).T
    return bounding_boxes

def _get_image_boxes(boxes: np.ndarray, frame: torch.Tensor, size=24):
    """
    Extracts image boxes from the given frame based on the provided bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes.
        frame (torch.Tensor): Input frame.
        size (int, optional): Size of the image boxes. Defaults to 24.

    Returns:
        np.ndarray: Array of image boxes.
    """
    num_boxes = boxes.shape[0]
    width, height = frame.size(0), frame.size(1)

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')
        img_array = frame.numpy()
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
        
        img_box = zoom(img_box, (size / h[i], size / w[i], 1), order=1)
        img_boxes[i, :, :, :] = _preprocess(img_box)
    return img_boxes # (num_boxes, 3, size, size)

def _multiscale_fusion(boxes: List[List[Optional[np.ndarray]]]) -> List[Optional[np.ndarray]]:
    """
    Fuse bounding boxes from multiple scales.

    Args:
        boxes (List[List[Optional[np.ndarray]]]): A list of lists containing bounding boxes for each scale.
            Each inner list represents the bounding boxes for a particular scale.
            Each element in the inner list is a numpy array representing the bounding boxes for a frame at that scale.
            If a frame does not have any bounding boxes at a particular scale, the corresponding element is None.

    Returns:
        List[Optional[np.ndarray]]: A list of fused bounding boxes for each frame.
            Each element in the list is a numpy array representing the fused bounding boxes for a frame.
            If a frame does not have any bounding boxes across all scales, the corresponding element is None.
    """
    if not all([len(box) == len(boxes[0]) for box in boxes]):
        raise ValueError('All scales must have the same number of frames')
    num_frames = len(boxes[0])
    fused_boxes = []
    for i in range(num_frames):
        frame_boxes_with_different_scales = [box[i] for box in boxes if box[i] is not None]
        if len(frame_boxes_with_different_scales) == 0:
            fused_boxes.append(None)
            continue
        fused_boxes.append(np.vstack(frame_boxes_with_different_scales))
    return fused_boxes

def _second_stage_batch(frames: List[np.ndarray],
                        boxes: List[np.ndarray],
                        threshold: float,
                        nms_threshold: float):
    """
    Applies the second stage of face detection to a batch of frames.

    Args:
        frames (List[np.ndarray]): The input frames.
        boxes (List[np.ndarray]): The bounding boxes for each frame.
        threshold (float): The threshold for face detection.
        nms_threshold (float): The NMS threshold.

    Returns:
        List[np.ndarray]: A list of bounding boxes for each frame.
    """
    # merge frames into a single tensor, record the original frame indices
    frame_indices = []
    for i, frame in enumerate(frames):
        frame_indices.extend([i] * frame.shape[0])
    frames = np.vstack(frames) # (batch_size * num_boxes, 3, 24, 24)
    frames = Variable(torch.FloatTensor(frames), volatile=True)
    outputs = rnet(frames)

    # extract probabilities and offsets
    offsets = outputs[0].data.numpy() # (batch_size * num_boxes, 4)
    probs = outputs[1].data.numpy() # (batch_size * num_boxes, 2)

    # filter boxes based on threshold
    keep = np.where(probs[:, 1] > threshold)[0]
    boxes = np.vstack(boxes)
    boxes = boxes[keep]
    boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    # apply NMS
    keep = nms(boxes, nms_threshold)
    boxes = boxes[keep]
    boxes = calibrate_box(boxes, offsets[keep])
    boxes = convert_to_square(boxes)
    boxes[:, 0:4] = np.round(boxes[:, 0:4])

    # split boxes back into individual frames
    frame_boxes = []
    for i in range(len(frames)):
        frame_boxes.append(boxes[frame_indices == i])
    return frame_boxes

def detect_faces_batch(batch: Tuple[torch.Tensor, torch.Tensor],
                       min_face_size=20.0,
                       thresholds=[0.6, 0.7, 0.8],
                       nms_thresholds=[0.7, 0.7, 0.7]):
    frames, masks = batch
    batch_size = frames.size(0)

    scales = _get_scales(frames, min_face_size)
    
    bounding_boxes = []

    for s in scales:
        bounding_boxes.append(_first_stage_batch(frames, s, thresholds[0])) # (scales, batch_size, (ndarray(num_boxes, 9) | None))
    
    fused_boxes = _multiscale_fusion(bounding_boxes) # (batch_size, (ndarray(num_boxes, 9) | None))
    boxed_frames = []
    for i, boxes in enumerate(fused_boxes):
        if boxes is None or masks[i] == 1:
            masks[i] = 1
            continue
        keep = nms(boxes[:, 0:5], nms_thresholds[0])
        boxes = boxes[keep]
        boxes = calibrate_box(boxes[:, 0:5], boxes[:, 5:])
        boxes = convert_to_square(boxes)
        boxes[:, 0:4] = np.round(boxes[:, 0:4])
        fused_boxes[i] = boxes # (batch_size, (ndarray(num_boxes, 9)))

        boxed_frame = _get_image_boxes(boxes, frames[i], size=24)
        boxed_frames.append(boxed_frame) # (batch_size, ndarray(num_boxes, 3, 24, 24))