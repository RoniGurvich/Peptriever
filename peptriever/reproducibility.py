import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)
