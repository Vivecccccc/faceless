from enum import Enum
from hashlib import md5
from typing import Optional, List
from pydantic import BaseModel, validator
from datetime import datetime
from dateutil import tz

def hash_anystring(s: str):
    return md5(s.encode()).hexdigest()

class StatusEnum(Enum):
    NEVER = 0
    SUCCESS = 1
    FAILURE = -1

class VideoMetadata(BaseModel):
    application_id: str
    s3_file_key: str
    created_at: Optional[str]

    @validator('created_at', pre=True)
    def localize_created_at(cls, v):
        if v is None:
            return v
        bj_tz = tz.gettz('Asia/Shanghai')
        if isinstance(v, datetime):
            v = v.replace(tzinfo=bj_tz).isoformat()
        elif isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v)
                v = v_dt.replace(tzinfo=bj_tz).isoformat()
            except Exception as e:
                raise ValueError(f'Error parsing created_at: {e}')
        else:
            raise ValueError('created_at should be either datetime or isoformat string')
        return v

class VideoStatus(BaseModel):
    fetched: StatusEnum = StatusEnum.NEVER
    captured: StatusEnum = StatusEnum.NEVER
    peated: StatusEnum = StatusEnum.NEVER # this indicates the status of video being feature extracted

    def is_all_green(self):
        return self.fetched.value + self.captured.value + self.peated.value == 3

class Video(BaseModel):
    id: str
    indexed_at: str
    attempt_times: int
    metadata: VideoMetadata
    status: VideoStatus
    embedding: Optional[List[float]] = None

    def get_job_id(self):
        return hash_anystring(self.indexed_at)
    
    def to_es_obj(self):
        return {
            'indexed_at': self.indexed_at,
            'attempt_times': self.attempt_times,
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
    
    # validator to change indexed_at from datetime to timezone-awared isoformat string
    @validator('indexed_at', pre=True)
    def localize_indexed_at(cls, v):
        bj_tz = tz.gettz('Asia/Shanghai')
        if isinstance(v, datetime):
            v = v.replace(tzinfo=bj_tz).isoformat()
        elif isinstance(v, str):
            try:
                v_dt = datetime.fromisoformat(v)
                v = v_dt.replace(tzinfo=bj_tz).isoformat()
            except Exception as e:
                raise ValueError(f'Error parsing indexed_at: {e}')
        else:
            raise ValueError('indexed_at should be either datetime or isoformat string')
        return v