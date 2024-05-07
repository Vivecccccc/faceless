import json
import logging
from typing import List
from collections import defaultdict

from utils.constants import *
from utils.dataclasses import Video, StatusEnum
from utils.datasets import remove_serialized_frames

from pipeline.fetch_videos import fetch_videos, fetch_prev_failure
from pipeline.capture_frames import run_batch_capture, postprocessing
from pipeline.extract_features import extract_features
from pipeline.index_es import index_videos, get_top_k

from utils.exceptions import StageFetchException, StageCaptureException, StagePeatException, StageIndexException

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def run_pipelines():
    try:
        latest_videos: List[Video] = fetch_videos(since=None)
        prev_failures: List[Video] = fetch_prev_failure()
    except Exception as e:
        logging.error(StageFetchException(f'Error while fetching videos: {e}'))
        return

    videos = latest_videos + prev_failures
    valid_videos = list(filter(lambda x: x.status.fetched == StatusEnum.SUCCESS, videos))
    
    num_frames = DETECTOR_CONSTANTS['NUM_FRAMES']
    h5_paths = defaultdict(list)

    try:
        for video in valid_videos:
            frames = run_batch_capture(video, num_frames=num_frames)
            if frames is not None:
                face_h5_path = postprocessing(frames, video, num_frames)
                if face_h5_path is not None and video.status.captured == StatusEnum.SUCCESS:
                    h5_paths[face_h5_path].append(video)
            else:
                remove_serialized_frames(video)
    except Exception as e:
        logging.error(StageCaptureException(f'Error while capturing frames: {e}'))
        return
    
    batch_size = RECOGNIZER_CONSTANTS['BATCH_SIZE']
    backbone_ckpt_path = RECOGNIZER_CONSTANTS['BACKBONE_CKPT_PATH']

    try:
        for h5_path, part_videos in h5_paths.items():
            extract_features(part_videos, h5_path, num_frames, batch_size, backbone_ckpt_path=backbone_ckpt_path)
    except Exception as e:
        logging.error(StagePeatException(f'Error while extracting features: {e}'))
        return
    
    try:
        latest_indexed_at = latest_videos[0].indexed_at if latest_videos else None
        is_valid_mapping, maybe_retry = index_videos(videos)
        n = 0
        while is_valid_mapping and maybe_retry and n < 3:
            is_valid_mapping, maybe_retry = index_videos(maybe_retry, current_indexed_at=latest_indexed_at)
            n += 1
        similars = get_top_k(k=5, num_candidates=100, threshold=0.7, videos=videos)
        # serialize similars to JSON
        with open('similars.json', 'w') as f:
            json.dump([x.model_dump() for x in similars], f, indent=4)
    except Exception as e:
        logging.error(StageIndexException(f'Error while indexing videos: {e}'))
        return

if __name__ == '__main__':
    run_pipelines()