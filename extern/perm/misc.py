from typing import Dict, Union

import numpy as np
import torch

EPSILON = 1e-7


def copy2cpu(data: Union[torch.Tensor, Dict]) -> Union[np.ndarray, Dict]:
    if isinstance(data, dict):
        return {k: v.detach().cpu().numpy() for k, v in data.items()}
    return data.detach().cpu().numpy()
