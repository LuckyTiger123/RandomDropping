import os
import sys
import torch
import torch_geometric
from torch.nn import Parameter
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import src.MessageDeletion as Md
import src.Augmentor as Aug
import src.utils as utils


class OurModelLayer(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, drop_method: str, sample_number: int,
                 add_self_loops: bool = True, normalize: bool = True, bias: bool = True, unbias: bool = True):
        super(OurModelLayer, self).__init__()
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.unbias = unbias
        self.edge_weight = None

        # parameters
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # drop method
        if drop_method == 'DropMessageChannel':
            self.drop_method = Md.DropMessageChannel()
        elif drop_method == 'DropNode':
            self.drop_method = Md.DropNode()
        elif drop_method == 'DropEdge':
            self.drop_method = Md.DropEdge()
        elif drop_method == 'Dropout':
            self.drop_method = Md.Dropout()
        else:
            self.drop_method = None

        # augmentor
        self.augmentor = Aug.Augmentor(sample_number, self.drop_method)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.5):

        if self.training and self.drop_method is not None:
            # message addition TODO

            # message deletion
            agg_result = self.augmentor.sample(edge_index, x, drop_rate, self.normalize, self.unbias,
                                               self.add_self_loops)

        else:
            # add self loop
            if self.add_self_loops:
                num_nodes = x.size(0)
                edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
                edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)

            # normalize
            if self.normalize:
                row, col = edge_index
                deg = torch_geometric.utils.degree(col, x.size(0), dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                self.edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            agg_result = self.propagate(edge_index, size=None, x=x)

        # transform
        out = agg_result.matmul(self.weight)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor):
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)
        return x_j

    def reset_parameters(self):
        utils.glorot(self.weight)
        utils.zeros(self.bias)

# device = torch.device('cuda:{}'.format(2) if torch.cuda.is_available() else 'cpu')
# drop_rate = 0.5
# feature_matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
# dense_matrix = torch.ones(3, 3)
# edge_index1 = torch_geometric.utils.dense_to_sparse(dense_matrix)[0].to(device)
# model = OurModelLayer(4, 2, 'DropMessageChannel', 2).to(device)
# out = model(feature_matrix, edge_index1)
# label = torch.tensor([1, 0, 1]).to(device)
# print(out)
# print(out.size())
# result = utils.average_agg(out)
# print(result)
# print(result.size())
