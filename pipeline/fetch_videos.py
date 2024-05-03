import os
import uuid
import logging
import cv2
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from boto3.session import Session

from apps.indexer import es_client, INDEX_NAME
from apps.indexer.index_helpers import get_last_index_time, create_or_check_index

from utils.constants import S3_CONSTANTS, META_CONSTANTS, PORTAL_CONSTANTS
from utils.dataclasses import StatusEnum, Video, VideoMetadata, VideoStatus

import warnings
warnings.filterwarnings("ignore")

AWS_ACCESS_KEY_ID = S3_CONSTANTS['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = S3_CONSTANTS['AWS_SECRET_ACCESS_KEY']
S3_ENDPOINT_URL = S3_CONSTANTS['ENDPOINT_URL']
S3_BUCKET_NAME = S3_CONSTANTS['BUCKET_NAME']
S3_REGION_NAME = S3_CONSTANTS['REGION_NAME']

session = Session()
s3_client = session.client(service_name='s3',
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                           endpoint_url=S3_ENDPOINT_URL,
                           region_name=S3_REGION_NAME)

def _get_video(s3_file_key: str, id: str) -> Tuple[bool, datetime]:
    proposed_path = os.path.join(META_CONSTANTS['TEMP_VIDEO_STORAGE'], f'{id}{META_CONSTANTS["VIDEO_EXTENSION"]}')
    created_at = None
    try:
        created_at = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_file_key)['LastModified']
        s3_client.download_file(Bucket=S3_BUCKET_NAME, Key=s3_file_key, 
                                Filename=proposed_path)
        integrity = _check_video_integrity(proposed_path)
        return integrity, created_at
    except Exception:
        return False, created_at
    
def _check_video_integrity(file_path: str) -> bool:
    cap = None
    flag = False
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames != 0:
            flag = True
    except Exception:
        logging.error(f'Video file {file_path} may be corrupted')
    finally:
        if cap:
            cap.release()
    return flag
    
def fetch_prev_failure() -> List[Video]:
    videos: List[Video] = []
    try:
        index_name = create_or_check_index(exists_ok=True)
        if index_name is None:
            raise Exception(f'Index with name {INDEX_NAME} has a different mapping')
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"bool": {"must_not": {"term": {"status.fetched": 1}}}},
                        {"bool": {"must_not": {"term": {"status.captured": 1}}}},
                        {"bool": {"must_not": {"term": {"status.peated": 1}}}}
                    ],
                    "must_not": [
                        {"range": {"attempt_times": {"gt": 3}}}
                    ]
                }
            }
        }
        response = es_client.search(index=index_name, body=query)
    except Exception as e:
        logging.error(f'Error fetching previous failures from {INDEX_NAME}: {e}')
        return videos
    # perform query to fetch videos with any failure status 
    # i.e., any of fetched / captured / peated not equal to 1
    hits = response['hits']['hits']
    for hit in hits:
        video_raw = hit['_source']
        flag = False
        try:
            video = Video(id=hit['_id'],
                          indexed_at=video_raw['indexed_at'],
                          attempt_times=video_raw['attempt_times'],
                          metadata=VideoMetadata(application_id=video_raw['metadata']['application_id'],
                                                 s3_file_key=video_raw['metadata']['s3_file_key'],
                                                 created_at=video_raw['metadata']['created_at']),
                          status=VideoStatus(fetched=StatusEnum(video_raw['status']['fetched']),
                                             captured=StatusEnum(video_raw['status']['captured']),
                                             peated=StatusEnum(video_raw['status']['peated'])),
                          embedding=video_raw['embedding'])
        except Exception as e:
            logging.error(f'Error while fetching info of previous failure {hit["_id"]}: {e}')
            continue

        flag, created_at = _get_video(video.metadata.s3_file_key, video.id)
        if flag and created_at is not None:
            video.status.fetched = StatusEnum.SUCCESS
        else:
            logging.error(f'Error while fetching video of previous failure {hit["_id"]}')
            video.status.fetched = StatusEnum.FAILURE
        video.attempt_times += 1
        video.metadata = VideoMetadata(application_id=video.metadata.application_id,
                                       s3_file_key=video.metadata.s3_file_key,
                                       created_at=created_at)
        video.status = VideoStatus(fetched=video.status.fetched,
                                   captured=StatusEnum.NEVER,
                                   peated=StatusEnum.NEVER)
        videos.append(video)
    return videos

def fetch_videos(since: Optional[datetime]) -> List[Video]:
    videos: List[Video] = []
    try:
        if since is None:
            since: datetime = get_last_index_time()
    except Exception as e:
        logging.warning('Failed to retrieve last indexed time, defaulting to one day ago')
        since: datetime = datetime.now() - timedelta(days=1)
    now: datetime = datetime.now()
    try:
        response_get = requests.get(url=PORTAL_CONSTANTS['ENDPOINT_URL'], params={'since': since, 'until': datetime.now()})
        response_get.raise_for_status()
    except Exception as e:
        logging.error(f'Failed to retrieve video ids due to {e}')
        return videos
    
    data = response_get.json()

    for item in data:
        flag = False
        id: str = uuid.uuid4().hex
        flag, created_at = _get_video(item['file_id'], id)
        if not flag or created_at is None:
            logging.error(f'Error while fetching video {item["file_id"]}')

        metadata: VideoMetadata = VideoMetadata(application_id=item['app_id'],
                                                s3_file_key=item['file_id'],
                                                created_at=created_at)
        status = VideoStatus(fetched=StatusEnum.SUCCESS if flag else StatusEnum.FAILURE)
        videos.append(Video(id=id, indexed_at=now, attempt_times=1, metadata=metadata, status=status))

        item['status_id'] = status.fetched.value
        item['timestamp'] = created_at

    # try:
    #     response_post = requests.post(url=PORTAL_CONSTANTS['ENDPOINT_URL'], json=data)
    #     response_post.raise_for_status()
    # except Exception as e:
    #     logging.error(f'Failed to update video status due to {e}')
    
    return videos