import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_scatter import scatter
from typing import Optional


class MADGap_calculator(MessagePassing):
    def __init__(self):
        super(MADGap_calculator, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1)
        self.dist = None

    def forward(self, x: Tensor, neb: Adj, rmt: Adj):
        self.propagate(edge_index=neb, size=None, x=x)
        mad_neb = self.dist
        self.propagate(edge_index=rmt, size=None, x=x)
        mad_rmt = self.dist

        return mad_rmt - mad_neb

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, size_i: Optional[int]):
        cos_sim = self.cos(x_i, x_j)
        dist = 1 - cos_sim
        out_sum = scatter(dist, index, 0, dim_size=size_i, reduce='mean')
        self.dist = out_sum.sum() / out_sum.nonzero().size(0)
        # self.dist = out_sum.mean()
        return x_j

# device = torch.device('cuda:{}'.format(5) if torch.cuda.is_available() else 'cpu')
# x = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]).to(device)
# neb = Tensor([[0, 0, 1, 1, 2, 3], [1, 3, 0, 2, 1, 0]]).long().to(device)
# rmt = Tensor([[0, 1, 1, 2], [1, 0, 2, 1]]).long().to(device)
# cal = MADGap_calculator()
# MAD_gap = cal(x, neb, rmt)
# print(MAD_gap)
