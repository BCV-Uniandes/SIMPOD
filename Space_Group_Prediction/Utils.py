import numpy as np
import os
import torch
import random


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def gray2rgb(x):
    x = x.repeat(3,1,1) if x.shape[0] == 1 else x
    return x