import os
import uuid
import logging
import requests
from datetime import datetime
from typing import List, Optional, Tuple
from boto3.session import Session

from ..apps.indexer.index_helpers import get_last_index_time

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

def _get_video(s3_file_key: str, id: str) -> Tuple[bool, Optional[datetime]]:
    try:
        created_at = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_file_key)['LastModified']
        s3_client.download_file(Bucket=S3_BUCKET_NAME, Key=s3_file_key, Filename=os.path.join(META_CONSTANTS['TEMP_VIDEO_STORAGE'], id, META_CONSTANTS['VIDEO_EXTENSION']))
        return True, created_at if isinstance(created_at, datetime) else None
    except Exception as e:
        logging.error(f'Error while fetching video {s3_file_key}: {e}')
        return False, None
    
def fetch_videos(since: Optional[datetime]):
    videos: List[Video] = []
    if since is None:
        since: datetime = get_last_index_time()
    now: datetime = datetime.now()
    try:
        response_get = requests.get(url=PORTAL_CONSTANTS['ENDPOINT_URL'], params={'since': since, 'until': datetime.now()})
        response_get.raise_for_status()
    except Exception as e:
        logging.error(f'Failed to retrieve video ids due to {e}')
        return videos
    
    data = response_get.json()

    for item in data:
        id: str = uuid.uuid4().hex
        flag, created_at = _get_video(item['file_id'], id)

        metadata: VideoMetadata = VideoMetadata(application_id=item['app_id'],
                                                s3_file_key=item['file_id'],
                                                created_at=created_at)
        status = VideoStatus(fetched=StatusEnum.SUCCESS if flag else StatusEnum.FAILURE)
        videos.append(Video(id=id, indexed_at=now, metadata=metadata, status=status))

        item['status_id'] = status.fetched
        item['timestamp'] = created_at

    try:
        response_post = requests.post(url=PORTAL_CONSTANTS['ENDPOINT_URL'], json=data)
        response_post.raise_for_status()
    except Exception as e:
        logging.error(f'Failed to update video status due to {e}')
    
    return videos