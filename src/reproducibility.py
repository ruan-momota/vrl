from __future__ import annotations

import os
import random


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch when those libraries are available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    return seed
