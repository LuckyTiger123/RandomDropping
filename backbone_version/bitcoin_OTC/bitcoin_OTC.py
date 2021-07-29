import os
import gc
import sys
import torch
import random
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import BitcoinOTC
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

cuda_device = 3

train_dataset = 'Cora'
drop_method = 'DropMessage'
drop_rate = 0.5
backbone = 'APPNP'
unbias = False

# random generate train, validate, test mask
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

dataset = BitcoinOTC(root='/home/luckytiger/TestDataset/BitcoinOTC', edge_window_size=10)
data = dataset[0].to(device=device)
print(data)
