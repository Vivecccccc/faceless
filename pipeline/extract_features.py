import time
import logging
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def l2_norm(input: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class FrameDataset(Dataset):
    pass
