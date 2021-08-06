import sys
import os
import torch_geometric
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import src.utils as utils


class ModifiedGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, drop_method: float = 0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ModifiedGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.drop_method = drop_method

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        utils.glorot(self.lin_l.weight)
        utils.glorot(self.lin_r.weight)
        utils.glorot(self.att_l)
        utils.glorot(self.att_r)
        utils.zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # cal nodes' degree
        if isinstance(edge_index, Tensor):
            row, col = edge_index
        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
        deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)

        if self.drop_method == -1:
            drop_rate_matrix = 1 / deg
        elif self.drop_method == 2:
            drop_rate_matrix = torch.rand(x.size(0)) * 0.3 + 0.1
        else:
            drop_rate_matrix = None
        if drop_rate_matrix is not None:
            drop_rate_matrix = drop_rate_matrix.view(-1, 1).to(x.device)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, drop_rate_matrix=drop_rate_matrix)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor,
                edge_index_j: Tensor, size_i: Optional[int], drop_rate_matrix: OptTensor) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        # alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j = x_j * alpha.unsqueeze(-1)
        if self.training:
            if drop_rate_matrix is not None:
                drop_rate_matrix = drop_rate_matrix.index_select(0, edge_index_j).unsqueeze(-1)
                x_j = x_j * (1 / drop_rate_matrix)
                drop_rate_matrix = drop_rate_matrix.expand(-1, x_j.size(1), x_j.size(-1))
                message_mask = torch.bernoulli(drop_rate_matrix).to(drop_rate_matrix.device)
                x_j = x_j.mul(message_mask)
            else:
                x_j = F.dropout(x_j, self.drop_method)
        return x_j

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
