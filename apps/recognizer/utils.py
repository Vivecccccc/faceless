import torch

def l2_norm(x: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(x, 2, axis, True)
    output = torch.div(x, norm)
    return output