import os
import sys
import torch
import torch_geometric
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Parameter

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import src.utils as utils


class ModifiedGCN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, drop_method: float = 0, add_self_loops: bool = True,
                 normalize: bool = True, bias: bool = True):
        super(ModifiedGCN, self).__init__()
        self.drop_method = drop_method
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.deg = None

        # parameters
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj):

        # add self loops
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # normalize
        edge_weight = None
        if self.normalize:
            if isinstance(edge_index, Tensor):
                row, col = edge_index
            elif isinstance(edge_index, SparseTensor):
                row, col, _ = edge_index.coo()
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            self.deg = deg
            deg_inv_sqrt = deg.pow(-0.5)
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        if self.drop_method == -1:
            drop_rate_matrix = 1 / self.deg
        elif self.drop_method == 2:
            drop_rate_matrix = torch.rand(x.size(0)) * 0.3 + 0.1
        else:
            drop_rate_matrix = None
        if drop_rate_matrix is not None:
            drop_rate_matrix = drop_rate_matrix.view(-1, 1).to(x.device)

        out = self.propagate(edge_index=edge_index, size=None, x=x, edge_weight=edge_weight,
                             drop_rate_matrix=drop_rate_matrix)

        out = out.matmul(self.weight)
        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_index_j: Tensor, edge_weight: OptTensor, drop_rate_matrix: OptTensor):
        if self.training:
            if drop_rate_matrix is not None:
                drop_rate_matrix = drop_rate_matrix.index_select(0, edge_index_j)
                x_j = x_j * (1 / drop_rate_matrix)
                drop_rate_matrix = drop_rate_matrix.expand(-1, x_j.size(1))
                message_mask = torch.bernoulli(drop_rate_matrix).to(drop_rate_matrix.device)
                x_j = x_j.mul(message_mask)
            else:
                x_j = F.dropout(x_j, self.drop_method)

        if edge_weight is not None:
            x_j = edge_weight.view(-1, 1) * x_j

        return x_j
