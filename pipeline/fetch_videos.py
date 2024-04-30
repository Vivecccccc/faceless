import os
import uuid
import logging
import cv2
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from boto3.session import Session

from ..apps.indexer import es_client
from ..apps.indexer.index_helpers import get_last_index_time, create_or_check_index

from ..utils.constants import S3_CONSTANTS, META_CONSTANTS, PORTAL_CONSTANTS
from ..utils.dataclasses import StatusEnum, Video, VideoMetadata, VideoStatus

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
    try:
        created_at = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=s3_file_key)['LastModified']
        if not isinstance(created_at, datetime):
            raise Exception('Failure retrieving video creation time')
        s3_client.download_file(Bucket=S3_BUCKET_NAME, Key=s3_file_key, 
                                Filename=proposed_path)
        integrity = _check_video_integrity(proposed_path)
        return integrity, created_at # returning value is always [true, datetime]
    except Exception as e:
        raise Exception(f'Error while fetching video {s3_file_key}: {e}')
    
def _check_video_integrity(file_path: str) -> bool:
    cap = None
    try:
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError('Video file has no frames')
        return True
    except Exception as e:
        raise Exception(f'Video file may be corrupted') from e
    finally:
        if cap:
            cap.release()
    
def fetch_prev_failure() -> List[Video]:
    videos: List[Video] = []
    try:
        index_name = create_or_check_index(exists_ok=True)
        if index_name is None:
            raise Exception(f'Index with name {index_name} has a different mapping')
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"bool": {"must_not": {"term": {"status.fetched": 1}}}},
                        {"bool": {"must_not": {"term": {"status.captured": 1}}}},
                        {"bool": {"must_not": {"term": {"status.peated": 1}}}}
                    ]
                }
            }
        }
        response = es_client.search(index=index_name, body=query)
    except Exception as e:
        logging.error(f'Error fetching previous failures from {index_name}: {e}')
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
                          metadata=VideoMetadata(application_id=video_raw['metadata']['application_id'],
                                                 s3_file_key=video_raw['metadata']['s3_file_key'],
                                                 created_at=video_raw['metadata']['created_at']),
                          status=VideoStatus(fetched=StatusEnum(video_raw['status']['fetched']),
                                             captured=StatusEnum(video_raw['status']['captured']),
                                             peated=StatusEnum(video_raw['status']['peated'])),
                          embedding=video_raw['embedding'])
            flag, created_at = _get_video(video_raw['metadata']['s3_file_key'], hit['_id'])
            video.status.fetched = StatusEnum.SUCCESS
        except Exception as e:
            logging.error(f'Error while fetching previous failure {hit["_id"]}: {e}')
            video.status.fetched = StatusEnum.FAILURE
        finally:
            if flag and created_at != video_raw['metadata']['created_at']:
                video.metadata.created_at = created_at
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
        try:
            flag, created_at = _get_video(item['file_id'], id)
        except Exception as e:
            logging.error(f'Error while fetching video {item["file_id"]}: {e}')

        metadata: VideoMetadata = VideoMetadata(application_id=item['app_id'],
                                                s3_file_key=item['file_id'],
                                                created_at=created_at if flag else None)
        status = VideoStatus(fetched=StatusEnum.SUCCESS if flag else StatusEnum.FAILURE)
        videos.append(Video(id=id, indexed_at=now, metadata=metadata, status=status))

        item['status_id'] = status.fetched.value
        item['timestamp'] = created_at

    # try:
    #     response_post = requests.post(url=PORTAL_CONSTANTS['ENDPOINT_URL'], json=data)
    #     response_post.raise_for_status()
    # except Exception as e:
    #     logging.error(f'Failed to update video status due to {e}')
    
    return videos