import logging
from pydantic import BaseModel
from typing import Optional, List, Tuple
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from ..apps.recognizer.aggregator import Aggregator

import warnings
warnings.filterwarnings("ignore")

def _extract(model: Aggregator, dataloader: DataLoader):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            images, masks, vid_ids = batch
            real_batch_size = images.size(0)
            outputs = model(images, mask=masks)
            features.extend([(vid_ids[j], outputs[j]) for j in range(real_batch_size)])
    return features

def _init_model(num_frames: int, 
                input_size: Tuple[int, int] = (112, 112),
                ckpt_path: Optional[str] = None, 
                backbone_ckpt_path: Optional[str] = None):
    if not ckpt_path and not backbone_ckpt_path:
        raise ValueError('At least one of the model checkpoint paths must be provided')
    model = Aggregator(num_frames, input_size=input_size)
    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path))
    if backbone_ckpt_path:
        model.backbone.load_state_dict(torch.load(backbone_ckpt_path))
    return model