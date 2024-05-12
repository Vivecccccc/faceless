import h5py
import logging
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader

from .dataclasses import Video
from .constants import *
from .exceptions import DataMismatchException, H5FileInvalidException

def _infer_shape(frames: List[Tuple[int, Image.Image]]):
    if all([frame[1].size == frames[0][1].size for frame in frames]):
        return (frames[0][1].size[1], frames[0][1].size[0])
    else:
        raise DataMismatchException('All frames must have the same shape')

def serialize_frames(frames: List[Tuple[int, Image.Image]], 
                     num_frames: int,
                     v: Video) -> str:
    stored_path = os.path.join(META_CONSTANTS['TEMP_IMAGE_STORAGE'], f'{v.id}.h5')
    os.makedirs(os.path.dirname(stored_path), exist_ok=True)
    try:
        unified_shape = _infer_shape(frames)
        with h5py.File(stored_path, 'a') as file:
            if len(frames) != num_frames:
                raise DataMismatchException('Frames captured mismatch with the specified number of frames')
            file.create_dataset('raw', shape=(num_frames, *unified_shape, 3), dtype='uint8')
            dataset = file['raw']
            for i, frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                frame_array = np.array(frame, dtype='uint8')
                dataset[i] = frame_array
    except Exception as e:
        raise H5FileInvalidException(e)
    return stored_path

def serialize_faces(frames: List[Tuple[int, Image.Image]],
                    num_frames: int,
                    v: Video) -> str:
    stored_path = os.path.join(META_CONSTANTS['TEMP_IMAGE_STORAGE'], f'{v.get_job_id()}.h5')
    os.makedirs(os.path.dirname(stored_path), exist_ok=True)
    try:
        unified_shape = _infer_shape(frames)
        with h5py.File(stored_path, 'a') as file:
            file.create_dataset(f'{v.id}-face', shape=(num_frames, 3, *unified_shape), dtype='uint8')
            file.create_dataset(f'{v.id}-mask', shape=(num_frames,), dtype='bool')
            dataset = file[f'{v.id}-face']
            mask_dataset = file[f'{v.id}-mask']
            for i, frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                frame_array = np.array(frame, dtype='uint8')
                dataset[i] = frame_array.transpose((2, 0, 1))
                mask_dataset[i] = True
    except Exception as e:
        raise H5FileInvalidException(e)
    return stored_path

def remove_serialized_frames(v: Video):
    stored_path = os.path.join(META_CONSTANTS['TEMP_IMAGE_STORAGE'], f'{v.id}.h5')
    try:
        os.remove(stored_path)
    except Exception as e:
        logging.warning(f'Error while removing serialized frames for video {v.id}: {e}')

class FramesH5Dataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        with h5py.File(path, 'r') as file:
            self.data = np.array(file['raw'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])

    def collate_fn(self, batch):
        return torch.stack(batch)

class FramesH5DataLoader(DataLoader):
    def __init__(self, dataset: FramesH5Dataset):
        batch_size = len(dataset)
        super().__init__(dataset, 
                         batch_size=batch_size, 
                         shuffle=False,
                         collate_fn=dataset.collate_fn,
                         pin_memory=True)
        
class FaceH5Dataset(Dataset):
    def __init__(self, path: str, num_frames: int, transform=None):
        self.path = path
        self.indices = []
        self.num_frames = num_frames
        self.transform = transform
        with h5py.File(self.path, 'r') as file:
            self.video_ids = [key.split('-face')[0] for key in file.keys() if key.endswith('-face')]
            for video_id in self.video_ids:
                key_face = f'{video_id}-face'
                if num_frames != file[key_face].shape[0]:
                    raise DataMismatchException(f'Number of frames in dataset of {video_id} mismatches with the specified number of frames')
                self.indices.append(video_id)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        video_id = self.indices[idx]
        with h5py.File(self.path, 'r') as file:
            faces_arr = np.array(file[f'{video_id}-face'])
            mask = np.array(file[f'{video_id}-mask'])
            D = {
                'mask': torch.from_numpy(mask),
                'video_id': video_id
            }
        if self.transform:
            faces = []
            for i in range(self.num_frames):
                face = Image.fromarray(faces_arr[i].transpose(1, 2, 0))
                faces.append(self.transform(face))
            D['img'] = torch.stack(faces)
        else:
            D['img'] = torch.from_numpy(faces_arr)
        return D
        
    def collate_fn(self, batch):
        images = torch.stack([item['img'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        video_ids = [item['video_id'] for item in batch]
        return images, masks, video_ids