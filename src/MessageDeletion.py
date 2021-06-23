import torch
import torch_geometric
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from abc import ABCMeta, abstractmethod


# Abstract Class DeletionMethod
class DeletionMethod(metaclass=ABCMeta):
    @abstractmethod
    def drop(self, edge_index: Adj, x: Tensor, drop_rate: float, normalize: bool, unbias: bool, add_self_loop: bool):
        pass


# Message-Channel Random Dropping
class DropMessageChannel(MessagePassing, DeletionMethod):
    def __init__(self):
        super(DropMessageChannel, self).__init__()
        self.propagate_matrix = None
        self.edge_weight = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True,
             unbias: bool = True, add_self_loop: bool = True):
        # add self loop
        if add_self_loop:
            num_nodes = x.size(0)
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

        # normalize
        if normalize:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            self.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, unbias=unbias)
        return y

    def message(self, x_j: Tensor, drop_rate: float, unbias: bool):
        # drop message-channel
        self.propagate_matrix = x_j.mul(torch.bernoulli(torch.ones_like(x_j) - drop_rate).long().to(x_j.device))

        # adjust bias
        if unbias:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))

        # normalize
        if self.edge_weight is not None:
            self.propagate_matrix = self.propagate_matrix * self.edge_weight.view(-1, 1)

        return self.propagate_matrix


# Node Random Dropping
class DropNode(MessagePassing, DeletionMethod):
    def __init__(self):
        super(DropNode, self).__init__()
        self.propagate_matrix = None
        self.edge_weight = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True,
             unbias: bool = True, add_self_loop: bool = True):
        # drop node
        x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)

        # add self loop
        if add_self_loop:
            num_nodes = x.size(0)
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

        # normalize
        if normalize:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            self.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, unbias=unbias)
        return y

    def message(self, x_j: Tensor, drop_rate: float, unbias: bool):
        self.propagate_matrix = x_j

        # adjust bias
        if unbias:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))

        # normalize
        if self.edge_weight is not None:
            self.propagate_matrix = self.propagate_matrix * self.edge_weight.view(-1, 1)

        return self.propagate_matrix


# Edge Random Dropping
class DropEdge(MessagePassing, DeletionMethod):
    def __init__(self):
        super(DropEdge, self).__init__()
        self.propagate_matrix = None
        self.edge_weight = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True,
             unbias: bool = True, add_self_loop: bool = True):
        # drop edge
        edge_reserved_size = int(edge_index.size(1) * (1 - drop_rate))
        perm = torch.randperm(edge_index.size(1))
        edge_index = edge_index.t()[perm][:edge_reserved_size].t()

        # add self loop
        if add_self_loop:
            num_nodes = x.size(0)
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

        # normalize
        if normalize:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            self.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, unbias=unbias)
        return y

    def message(self, x_j: Tensor, drop_rate: float, unbias: bool):
        self.propagate_matrix = x_j

        # adjust bias
        if unbias:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))

        # normalize
        if self.edge_weight is not None:
            self.propagate_matrix = self.propagate_matrix * self.edge_weight.view(-1, 1)

        return self.propagate_matrix


# Random Dropping out
class Dropout(MessagePassing, DeletionMethod):
    def __init__(self):
        super(Dropout, self).__init__()
        self.propagate_matrix = None
        self.edge_weight = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True,
             unbias: bool = True, add_self_loop: bool = True):
        # dropout
        x = F.dropout(x, drop_rate)

        # add self loop
        if add_self_loop:
            num_nodes = x.size(0)
            edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
            edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

        # normalize
        if normalize:
            row, col = edge_index
            deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            self.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        y = self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, unbias=unbias)
        return y

    def message(self, x_j: Tensor, drop_rate: float, unbias: bool):
        self.propagate_matrix = x_j

        # adjust bias
        if unbias:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))

        # normalize
        if self.edge_weight is not None:
            self.propagate_matrix = self.propagate_matrix * self.edge_weight.view(-1, 1)

        return self.propagate_matrix

# device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
# drop_rate = 0.5
# feature_matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
# dense_matrix = torch.ones(3, 3)
# edge_index1 = torch_geometric.utils.dense_to_sparse(dense_matrix)[0].to(device)
# drop_out = DropMessageChannel()
# out = drop_out.drop(edge_index1, feature_matrix)
# print(out)
# x = feature_matrix * torch.bernoulli(torch.ones(feature_matrix.size(0), 1) - drop_rate).to(device)
# print(feature_matrix.device)
