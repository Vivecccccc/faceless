import os
import uuid
import logging
import requests
from datetime import datetime
from typing import List, Optional
from boto3.session import Session

from ..utils.constants import S3_CONSTANTS
from ..utils.dataclasses import Video, VideoMetadata, VideoStatus

import warnings

warnings.filterwarnings("ignore")

AWS_ACCESS_KEY_ID = S3_CONSTANTS['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY = S3_CONSTANTS['AWS_SECRET_ACCESS_KEY']
ENDPOINT_URL = S3_CONSTANTS['ENDPOINT_URL']
BUCKET_NAME = S3_CONSTANTS['BUCKET_NAME']
REGION_NAME = S3_CONSTANTS['REGION_NAME']

session = Session()
s3_resource = session.resource(service_name='s3',
                               aws_access_key_id=AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                               endpoint_url=ENDPOINT_URL,
                               region_name=REGION_NAME)
resource_bucket = s3_resource.Bucket(BUCKET_NAME)

# assuming that we get the application ID and corresponding video file key from the API with `GET` request

