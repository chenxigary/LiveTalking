import os

import torch

def initialize_device():
    override = os.getenv("V3_AVATAR_DEVICE", "auto").strip().lower()
    if override not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError("V3_AVATAR_DEVICE must be auto, cpu, mps, or cuda")
    if override == "cpu":
        return torch.device("cpu")
    if override == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("V3_AVATAR_DEVICE=mps requested but MPS is unavailable")
        return torch.device("mps")
    if override == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V3_AVATAR_DEVICE=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
