import os
import sys
import torch
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.typing import Adj
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import backbone_version.backbone as Bb
import src.utils as utils


class DropBlock:
    def __init__(self, dropping_method: str):
        super(DropBlock, self).__init__()
        self.dropping_method = dropping_method

    def drop(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.dropping_method == 'DropNode':
            x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)
        elif self.dropping_method == 'DropEdge':
            edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
            perm = torch.randperm(edge_index.size(1))
            edge_index = edge_index.t()[perm][:edge_reserved_size].t()
        elif self.dropping_method == 'Dropout':
            x = F.dropout(x, drop_rate)

        return x, edge_index


class OurModelLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropping_method: str, backbone: str, heads: int = 1,
                 K: int = 1, alpha: float = 0, add_self_loops: bool = True, normalize: bool = True, bias: bool = True,
                 unbias: float = 0):
        super(OurModelLayer, self).__init__()
        self.dropping_method = dropping_method
        self.drop_block = DropBlock(dropping_method)

        if backbone == 'GCN':
            self.backbone = Bb.BbGCN(add_self_loops, normalize, unbias)
        elif backbone == 'GAT':
            self.backbone = Bb.BbGAT(in_channels, heads, add_self_loops, unbias)
        elif backbone == 'APPNP':
            self.backbone = Bb.BbAPPNP(K, alpha, add_self_loops, normalize, unbias)
        else:
            raise Exception('The backbone has not been realized')

        # parameters
        self.weight = Parameter(torch.Tensor(heads * in_channels, heads * out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.zeros(self.bias)
        self.backbone.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        message_drop = 0
        if self.dropping_method == 'DropMessage':
            message_drop = drop_rate
        if self.training:
            x, edge_index = self.drop_block.drop(x, edge_index, drop_rate)

        out = self.backbone(x, edge_index, message_drop)

        out = out.matmul(self.weight)
        if self.bias is not None:
            out += self.bias

        return out
