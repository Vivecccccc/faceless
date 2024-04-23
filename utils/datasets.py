import h5py
import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from .dataclasses import Video
from .constants import *

def _infer_shape(frames: List[Tuple[int, Image.Image]]):
    if all([frame[1].size == frames[0][1].size for frame in frames]):
        return frames[0][1].size
    else:
        raise ValueError('All frames must have the same shape')

def serialize_frames(frames: List[Tuple[int, Image.Image]], 
                     num_frames: int,
                     v: Video, 
                     is_raw: bool) -> Optional[str]:
    stored_path = os.path.join(META_CONSTANTS['TEMP_IMAGE_STORAGE'], f'{v.id}.h5')
    os.makedirs(os.path.dirname(stored_path), exist_ok=True)
    try:
        unified_shape = _infer_shape(frames)
        with h5py.File(stored_path, 'w') as file:
            if is_raw:
                if len(frames) != num_frames:
                    raise ValueError('Frames captured mismatch with the specified number of frames')
                file.create_dataset('raw', shape=(num_frames, *unified_shape, 3), dtype='uint8')
            else:
                file.create_dataset('face', shape=(num_frames, *unified_shape, 3), dtype='uint8')
                file.create_dataset('mask', shape=(num_frames,), dtype='bool')  # Change dtype to 'bool'
            dataset = file['raw' if is_raw else 'face']
            mask_dataset = file['mask'] if not is_raw else None  # Create a reference to the 'mask' dataset
            for i, frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                frame_array = np.array(frame, dtype='uint8')
                dataset[i] = frame_array
                if not is_raw:
                    mask_dataset[i] = True  # Set the mask value to True for the current frame
    except Exception as e:
        raise e
    return stored_path

class FramesH5Dataset(Dataset):
    def __init__(self, path: str, is_raw: bool):
        self.path = path
        self.mask = None
        with h5py.File(path, 'r') as file:
            self.data = file['raw' if is_raw else 'face']
            self.mask = file['mask'] if not is_raw else None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        flag = False
        if self.mask is not None:
            flag = self.mask[idx]
        return self.data[idx], flag

def collate_frames(batch):
    frames = [item[0] for item in batch]
    flags = [item[1] for item in batch]
    return torch.stack(frames), torch.tensor(flags)

class FramesH5DataLoader(DataLoader):
    def __init__(self, dataset: Dataset):
        batch_size = len(dataset)
        super().__init__(dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         collate_fn=collate_frames)