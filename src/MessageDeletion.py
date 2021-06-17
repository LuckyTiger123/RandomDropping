import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from abc import ABCMeta, abstractmethod


# Abstract Class DeletionMethod
class DeletionMethod(metaclass=ABCMeta):
    @abstractmethod
    def drop(self, edge_index: Adj, x: Tensor, drop_rate: float, normalize: bool):
        pass


# Message-Channel Random Dropping
class DropMessageChannel(MessagePassing, DeletionMethod):
    def __init__(self):
        super(DropMessageChannel, self).__init__()
        self.propagate_matrix = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True):
        self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, normalize=normalize)
        return self.propagate_matrix

    def message(self, x_j: Tensor, drop_rate: float, normalize: bool):
        self.propagate_matrix = x_j.mul(torch.bernoulli(torch.ones_like(x_j) - drop_rate).long().to(x_j.device))
        if normalize:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))
        return x_j


# Node Random Dropping
class DropNode(MessagePassing, DeletionMethod):
    def __init__(self):
        super(DropNode, self).__init__()
        self.propagate_matrix = None

    def drop(self, edge_index: Adj, x: torch.Tensor, drop_rate: float = 0.5, normalize: bool = True):
        x = x * torch.bernoulli(torch.ones(x.size(0), 1) - drop_rate).to(x.device)
        self.propagate(edge_index=edge_index, size=None, x=x, drop_rate=drop_rate, normalize=normalize)
        return self.propagate_matrix

    def message(self, x_j: Tensor, drop_rate: float, normalize: bool):
        self.propagate_matrix = x_j
        if normalize:
            self.propagate_matrix = self.propagate_matrix * (1 / (1 - drop_rate))
        return x_j

# device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')
# drop_rate = 0.5
# feature_matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
# x = feature_matrix * torch.bernoulli(torch.ones(feature_matrix.size(0), 1) - drop_rate).to(device)
# print(feature_matrix.device)
