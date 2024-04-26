import torch
import torch.nn as nn
from .model_irse import Backbone

import torch
import torch.nn as nn

class AttentionAggregation(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q0 = nn.Parameter(torch.zeros(d_model))
        self.transfer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
    def forward(self, x, mask=None):
        # x shape: (B, n, d)
        B, n, d = x.shape
        
        # the first attention block
        scores = torch.matmul(x, self.q0) / (self.d_model ** 0.5)  # (B, n)
        scores = scores + 1e-9
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, n)
        r0 = torch.matmul(weights.unsqueeze(-2), x).squeeze(-2)  # (B, d)
        return r0
        # TODO add 2nd attention block for training
        # transfer layer
        q1 = self.transfer(r0)  # (B, d)
        
        # the second attention block
        scores = torch.matmul(x, q1.unsqueeze(-1)).squeeze(-1) / (self.d_model ** 0.5)  # (B, n)    
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, n)
        r1 = torch.matmul(weights.unsqueeze(-2), x).squeeze(-2)  # (B, d)
        
        return r1
    
class Aggregator(nn.Module):
    def __init__(self, num_frames, d_model=512, ir_layers=50, input_size=(112, 112)):
        super().__init__()
        self.backbone = Backbone(input_size, ir_layers)
        self.aggregator = AttentionAggregation(d_model)
        self.num_frames = num_frames

    def forward(self, x: torch.Tensor, mask=None):
        # x shape: (B, n, 3, 112, 112)
        B, n, c, h, w = x.shape
        x = x.view(B * n, c, h, w)
        x = self.backbone(x)
        x = x.view(B, n, -1)
        x = self.aggregator(x, mask)
        return x
