from enum import Enum
from hashlib import md5
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

def hash_anystring(s: str):
    return md5(s.encode()).hexdigest()

class StatusEnum(Enum):
    NEVER = 0
    SUCCESS = 1
    FAILURE = -1

class VideoMetadata(BaseModel):
    application_id: str
    s3_file_key: str
    created_at: Optional[datetime]

class VideoStatus(BaseModel):
    fetched: StatusEnum = StatusEnum.NEVER
    captured: StatusEnum = StatusEnum.NEVER
    peated: StatusEnum = StatusEnum.NEVER # this indicates the status of video being feature extracted

    def is_all_green(self):
        return self.fetched.value + self.captured.value + self.peated.value == 3

class Video(BaseModel):
    id: str
    indexed_at: datetime
    metadata: VideoMetadata
    status: VideoStatus
    embedding: Optional[List[float]] = None

    def get_job_id(self):
        indexed_at_str = self.indexed_at.strftime('%Y%m%d%H%M%S')
        return hash_anystring(indexed_at_str)
    
    def to_es_obj(self):
        return {
            'indexed_at': self.indexed_at,
            'metadata': {
                'application_id': self.metadata.application_id,
                's3_file_key': self.metadata.s3_file_key,
                'created_at': self.metadata.created_at
            },
            'status': {
                'fetched': self.status.fetched.value,
                'captured': self.status.captured.value,
                'peated': self.status.peated.value
            },
            'embedding': self.embedding
        }