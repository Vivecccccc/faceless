import logging
from pydantic import BaseModel
from typing import Optional, List, Tuple
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


from ..utils.dataclasses import Video
from ..utils.datasets import FaceH5Dataset
from ..apps.recognizer.aggregator import Aggregator

import warnings
warnings.filterwarnings("ignore")

def extract(model: Aggregator, dataloader: DataLoader) -> List[Tuple[str, torch.Tensor]]:
    model.eval()
    features: List[Tuple[str, torch.Tensor]] = []
    with torch.no_grad():
        for batch in dataloader:
            images, masks, vid_ids = batch
            real_batch_size = images.size(0)
            outputs = model(images, mask=masks)
            features.extend([(vid_ids[j], outputs[j]) for j in range(real_batch_size)])
    return features

def init_model(num_frames: int, 
               input_size: Tuple[int, int] = (112, 112),
               ckpt_path: Optional[str] = None, 
               backbone_ckpt_path: Optional[str] = None) -> Aggregator:
    if not ckpt_path and not backbone_ckpt_path:
        raise ValueError('At least one of the model checkpoint paths must be provided')
    model = Aggregator(num_frames, input_size=input_size)
    try:
        if backbone_ckpt_path:
            model.backbone.load_state_dict(torch.load(backbone_ckpt_path))
        if ckpt_path:
            model.load_state_dict(torch.load(ckpt_path))
    except Exception as e:
        raise e
    return model

def init_data(path: str, num_frames: int, batch_size: int) -> DataLoader:
    face_ds = FaceH5Dataset(path, num_frames)
    dataloader = DataLoader(face_ds, batch_size=batch_size, shuffle=False, collate_fn=face_ds.collate_fn)
    return dataloader

def extract_features(vs: List[Video],
                     h5_path: str,
                     num_frames: int,
                     batch_size: int,
                     ckpt_path: Optional[str] = None,
                     backbone_ckpt_path: Optional[str] = None):
    pass