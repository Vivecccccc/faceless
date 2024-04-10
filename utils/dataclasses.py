from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

class VideoMetadata(BaseModel):
    application_id: str
    s3_file_key: str
    create_at: datetime

class VideoStatus(BaseModel):
    fetched: bool = False
    captured: bool = False
    peated: bool = False # this indicates the stastus of video being feature extracted

class Video(BaseModel):
    id: str
    metadata: VideoMetadata
    status: VideoStatus
    embedding: Optional[List[float]] = None