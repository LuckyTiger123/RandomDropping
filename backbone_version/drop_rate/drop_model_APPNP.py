import os
import sys
import torch
import torch_geometric
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import src.utils as utils


class ModifiedAPPNPConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, K: int, alpha: float, drop_method: float = 0,
                 dropout: float = 0., cached: bool = False, add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ModifiedAPPNPConv, self).__init__(**kwargs)
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.drop_method = drop_method

        # parameters
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None
        utils.glorot(self.weight)
        utils.zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

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

        h = x
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, drop_rate_matrix=drop_rate_matrix)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        out = x.matmul(self.weight)
        if self.bias is not None:
            out += self.bias
        return out

    def message(self, x_j: Tensor, edge_index_j: Tensor, edge_weight: Tensor, drop_rate_matrix: OptTensor) -> Tensor:
        if self.training:
            if drop_rate_matrix is not None:
                drop_rate_matrix = drop_rate_matrix.index_select(0, edge_index_j)
                x_j = x_j * (1 / drop_rate_matrix)
                drop_rate_matrix = drop_rate_matrix.expand(-1, x_j.size(1))
                message_mask = torch.bernoulli(drop_rate_matrix).to(drop_rate_matrix.device)
                x_j = x_j.mul(message_mask)
            else:
                x_j = F.dropout(x_j, self.drop_method)
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)
