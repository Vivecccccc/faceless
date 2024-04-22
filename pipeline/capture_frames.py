import os
import cv2
import shutil
import logging
import h5py
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

from apps.detector.align_trans import get_reference_facial_points, warp_and_crop_face
from apps.detector.detector import detect_faces
from ..utils.constants import META_CONSTANTS, DETECTOR_CONSTANTS
from ..utils.dataclasses import StatusEnum, Video
from ..utils.datasets import FramesH5DataLoader, FramesH5Dataset, serialize_frames

import warnings
warnings.filterwarnings("ignore")

TEMP_VIDEO_STORAGE = META_CONSTANTS['TEMP_VIDEO_STORAGE']
VIDEO_EXTENSION = META_CONSTANTS['VIDEO_EXTENSION']

CROP_SIZE = DETECTOR_CONSTANTS['CROP_SIZE']
SCALE = DETECTOR_CONSTANTS['SCALE']
REF_POINTS = get_reference_facial_points(default_square=True) * SCALE

def _capture(v: Video, num_frames: int) -> List[Image.Image]:
    frames = []
    v_local_path = os.path.join(TEMP_VIDEO_STORAGE, v.id, VIDEO_EXTENSION)
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

def _capture_frames(v: Video, num_frames: int) -> Optional[List[Tuple[int, Image.Image]]]:
    valid_frames = []

    try:
        frames = _capture(v, num_frames)
    except Exception as e:
        logging.error(f'Error while capturing frames for video {v.id}: {e}')
        v.status.captured = StatusEnum.FAILURE
        return None
    
    for i, frame in enumerate(frames):
        try:
            _, landmarks = detect_faces(frame)
            if len(landmarks) == 0:
                continue
            facial_keypoints = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = Image.fromarray(warp_and_crop_face(np.array(frame), facial_keypoints, REF_POINTS, (CROP_SIZE, CROP_SIZE)))
            valid_frames.append((i, warped_face))
        except:
            continue

    if not valid_frames:
        logging.error(f'failed detecting any valid faces in video {v.id}')
        v.status.captured = StatusEnum.FAILURE
    else:
        v.status.captured = StatusEnum.SUCCESS
    return valid_frames if v.status.captured else None

def run_batch_capture(v: Video, num_frames: int):
    valid_frames = []
    try:
        frames = _capture(v, num_frames)
        frames_h5_path = serialize_frames(frames, v, is_raw=True)
        del frames
    except Exception as e:
        logging.error(f'Error while capturing frames for video {v.id}: {e}')
        v.status.captured = StatusEnum.FAILURE
        return None
    fr_ds = FramesH5Dataset(frames_h5_path, is_raw=True)
    dataloader = FramesH5DataLoader(fr_ds)
    