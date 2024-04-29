import logging
from typing import Optional, List, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from ..utils.dataclasses import StatusEnum, Video
from ..utils.datasets import FaceH5Dataset
from ..apps.recognizer.aggregator import Aggregator
from ..apps.recognizer.utils import l2_norm

import warnings
warnings.filterwarnings("ignore")

def _extract(model: Aggregator, dataloader: DataLoader) -> List[Tuple[str, torch.Tensor]]:
    model.eval()
    features: List[Tuple[str, torch.Tensor]] = []
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

def _init_data(path: str, 
               num_frames: int, 
               batch_size: int,
               input_size: Tuple[int, int] = (112, 112)) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize([int(128 * input_size[0] / input_size[1]), 
                           int(128 * input_size[0] / input_size[1])]),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_ds = FaceH5Dataset(path, num_frames, transform=transform)
    dataloader = DataLoader(face_ds, batch_size=batch_size, shuffle=False, collate_fn=face_ds.collate_fn)
    return dataloader

def extract_features(vs: List[Video],
                     h5_path: str,
                     num_frames: int,
                     batch_size: int,
                     input_size: Tuple[int, int] = (112, 112),
                     ckpt_path: Optional[str] = None,
                     backbone_ckpt_path: Optional[str] = None):
    vid_dict = {v.id: v for v in vs}
    try:
        dataloader = _init_data(h5_path, num_frames, batch_size)
        model = _init_model(num_frames, ckpt_path=ckpt_path, backbone_ckpt_path=backbone_ckpt_path)
    except Exception as e:
        logging.error(f'Error while initializing model and data: {e}')
        return vs
    features = _extract(model, dataloader)
    peated_vids = set([])
    for vid_id, embedding in features:
        vid_dict[vid_id].embedding = l2_norm(embedding).tolist()
        vid_dict[vid_id].status.peated = StatusEnum.SUCCESS
        peated_vids.add(vid_id)
    for vid in vs:
        if vid.id not in peated_vids:
            vid.status.peated = StatusEnum.FAILURE
    return vs