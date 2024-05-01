from typing import List
from collections import defaultdict

from utils.constants import *
from utils.dataclasses import Video, StatusEnum

from pipeline.fetch_videos import fetch_videos, fetch_prev_failure
from pipeline.capture_frames import run_batch_capture, postprocessing
from pipeline.extract_features import extract_features
from pipeline.index_es import index_videos

def run_pipelines():
    latest_videos: List[Video] = fetch_videos(since=None)
    prev_failures: List[Video] = fetch_prev_failure()

    valid_videos = latest_videos + prev_failures
    valid_videos = list(filter(lambda x: x.status.fetched == StatusEnum.SUCCESS, valid_videos))
    
    num_frames = DETECTOR_CONSTANTS['NUM_FRAMES']
    h5_paths = defaultdict(list)
    for video in valid_videos:
        frames = run_batch_capture(video, num_frames=num_frames)
        if frames is not None:
            face_h5_path = postprocessing(frames, video, num_frames)
            if face_h5_path is not None and video.status.captured == StatusEnum.SUCCESS:
                h5_paths[face_h5_path].append(video)
    
    batch_size = RECOGNIZER_CONSTANTS['BATCH_SIZE']
    backbone_ckpt_path = RECOGNIZER_CONSTANTS['BACKBONE_CKPT_PATH']
    for h5_path, videos in h5_paths.items():
        extract_features(videos, h5_path, num_frames, batch_size, backbone_ckpt_path=backbone_ckpt_path)
    valid_videos = list(filter(lambda x: x.status.peated == StatusEnum.SUCCESS \
                               and x.embedding is not None, valid_videos))
    
    latest_indexed_at = latest_videos[0].indexed_at if latest_videos else None
    is_valid_mapping, maybe_retry = index_videos(valid_videos)
    n = 0
    while is_valid_mapping and maybe_retry and n < 3:
        is_valid_mapping, maybe_retry = index_videos(maybe_retry, current_indexed_at=latest_indexed_at)
        n += 1

if __name__ == '__main__':
    run_pipelines()