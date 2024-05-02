import os
import cv2
import math
import logging
import numpy as np
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from apps.detector.align_trans import get_reference_facial_points, warp_and_crop_face
from apps.detector.detector_batch import detect_faces_batch
from utils.constants import META_CONSTANTS, DETECTOR_CONSTANTS
from utils.dataclasses import StatusEnum, Video
from utils.datasets import FramesH5DataLoader, FramesH5Dataset, serialize_faces, serialize_frames, remove_serialized_frames

import warnings
warnings.filterwarnings("ignore")

TEMP_VIDEO_STORAGE = META_CONSTANTS['TEMP_VIDEO_STORAGE']
VIDEO_EXTENSION = META_CONSTANTS['VIDEO_EXTENSION']

CROP_SIZE = DETECTOR_CONSTANTS['CROP_SIZE']
SCALE = DETECTOR_CONSTANTS['SCALE']
REF_POINTS = get_reference_facial_points(default_square=True) * SCALE

def _capture(v: Video, num_frames: int) -> List[Image.Image]:
    frames = []
    v_local_path = os.path.join(TEMP_VIDEO_STORAGE, f'{v.id}{VIDEO_EXTENSION}')
    convert_to_pil = lambda frame: Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if os.path.exists(v_local_path):
        cap = cv2.VideoCapture(v_local_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_frames = min(num_frames, total_frames)
            if num_frames == 0:
                raise ValueError(f'Video file {v_local_path} may be corrupted')
            sampling_rate = total_frames // num_frames
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if i % sampling_rate == 0:
                    frames.append(convert_to_pil(frame))
                if len(frames) >= num_frames:
                    break
        except Exception as e:
            raise e
        finally:
            cap.release()
    else:
        raise FileNotFoundError(f'Video file {v_local_path} does not exist')
    return frames

def run_batch_capture(v: Video, num_frames: int) -> Optional[Dict[int, List[Tuple[Image.Image, np.ndarray]]]]:
    valid_frames = defaultdict(list)
    try:
        frames = _capture(v, num_frames)
        frames_h5_path = serialize_frames([x for x in enumerate(frames)], num_frames, v)
        del frames
    except Exception as e:
        logging.error(f'Error while capturing frames for video {v.id}: {e}')
        v.status.captured = StatusEnum.FAILURE
        return None
    
    fr_ds = FramesH5Dataset(frames_h5_path)
    dataloader = FramesH5DataLoader(fr_ds)
    batch = next(iter(dataloader)) # batch size equals to len(dataset) equalled to num_frames
    try:
        boxes, landmarks, indices = detect_faces_batch(batch)

        if landmarks.size == 0:
            raise Exception(f'failed detecting any valid faces in video')
        
        for idx, landmark, box in zip(indices, landmarks, boxes):
            frame_arr = batch[idx].numpy()
            facial_keypoints = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face = Image.fromarray(warp_and_crop_face(frame_arr, facial_keypoints, REF_POINTS, (CROP_SIZE, CROP_SIZE)))
            valid_frames[idx].append((warped_face, box))
    except Exception as e:
        logging.error(f'Error during detection phase for video {v.id}: {e}')
        v.status.captured = StatusEnum.FAILURE
        return None

    return valid_frames

def postprocessing(valid_frames: Dict[int, List[Tuple[Image.Image, np.ndarray]]], 
                   v: Video,
                   num_frames: int) -> Optional[str]:
    frame_items = list(valid_frames.items())
    frame_items = sorted(frame_items, key=lambda x: x[0])

    get_face_center = lambda box: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    get_face_size = lambda box: (box[2] - box[0]) * (box[3] - box[1])

    get_shifting = lambda coord1, coord2: np.linalg.norm([coord1[0] - coord2[0], coord1[1] - coord2[1]])
    
    # for each pair of consecutive frames,
    # calculate the shifts and changes of size of the face center between the largest face in frame 1 and the faces in frame 2
    # the face in frame 2 that has the least change should be the largest face in frame 2
    # use itertools to iterate over pairs of consecutive frames
    from itertools import pairwise
    trajectory = []
    for (idx1, candidates1), (idx2, candidates2) in pairwise(frame_items):
        candidate_faces1, candidate_boxes1 = zip(*candidates1)
        candidate_faces2, candidate_boxes2 = zip(*candidates2)

        largest_box1_i = np.argmax([get_face_size(box) for box in candidate_boxes1])
        largest_box1 = candidate_boxes1[largest_box1_i]
        largest_box2_i = np.argmax([get_face_size(box) for box in candidate_boxes2])
        largest_face2 = candidate_faces2[largest_box2_i]

        if not trajectory:
            trajectory.append(((idx1, candidate_faces1[largest_box1_i]), True))

        mini_traj = []
        for i2, box2 in enumerate(candidate_boxes2):
            shift = get_shifting(get_face_center(largest_box1), get_face_center(box2))
            size_change = math.sqrt(abs(get_face_size(largest_box1) - get_face_size(box2)))
            change_score = shift + size_change
            mini_traj.append((i2, change_score))
        mini_traj = sorted(mini_traj, key=lambda x: x[1])
        integrity = mini_traj[0][0] == largest_box2_i # the largest face in frame 2 should have the least change compared to that in frame 1
        trajectory.append(((idx2, largest_face2), integrity))
    # get the longest sub trajectory with integrity equals to True
    longest_sub_traj = []
    current_sub_traj = []
    for item in trajectory:
        if item[1]:
            current_sub_traj.append(item[0])
        else:
            if len(current_sub_traj) > len(longest_sub_traj):
                longest_sub_traj = current_sub_traj
            current_sub_traj = []
    if len(current_sub_traj) > len(longest_sub_traj):
        longest_sub_traj = current_sub_traj
    
    if len(longest_sub_traj) < round(num_frames / 3):
        v.status.captured = StatusEnum.FAILURE
        logging.error(f'Failed to capture enough valid frames for video {v.id}')
        return None
    
    try:
        stored_path = serialize_faces(longest_sub_traj, num_frames, v)
        v.status.captured = StatusEnum.SUCCESS
    except Exception as e:
        logging.error(f'Error while serializing faces for video {v.id}: {e}')
        v.status.captured = StatusEnum.FAILURE
        return None
    finally:
        remove_serialized_frames(v)
    return stored_path