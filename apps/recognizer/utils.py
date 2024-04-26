import torch

def l2_norm(input: torch.Tensor, axis: int = 1) -> torch.Tensor:
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output