import sys
import os
import torch
from torch_geometric.typing import Adj

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import src.MessageDeletion as Md


class Augmentor:
    def __init__(self, sample_number: int, drop_method: Md.DeletionMethod):
        super(Augmentor, self).__init__()
        self.sample_number = sample_number
        self.drop_method = drop_method

    def sample(self, edge_index: Adj, x: torch.Tensor, drop_rate: float, normalize: bool = True, unbias: bool = True,
               add_self_loop: bool = True):
        result = list()
        for i in range(self.sample_number):
            result_feature = self.drop_method.drop(edge_index, x, drop_rate, normalize, unbias, add_self_loop)
            result.append(result_feature)

        return torch.stack(result)


# device = torch.device('cuda:{}'.format(2) if torch.cuda.is_available() else 'cpu')
# drop_rate = 0.5
# feature_matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
# dense_matrix = torch.ones(3, 3)
# edge_index1 = torch_geometric.utils.dense_to_sparse(dense_matrix)[0].to(device)
# drop_out = Md.DropMessageChannel()
# a = Augmentor(sample_number=2, drop_method=drop_out)
# result = a.sample(edge_index1, feature_matrix, drop_rate)
# print(result)
# params = torch.rand(4, 2).to(device)
# print(params)
# r = result.matmul(params)
# print(r.size())
