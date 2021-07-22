import random
import torch
import torch_geometric
from torch import Tensor


# add edge with parameter relates to existing edges
def add_related_edge(edge_index: Tensor, add_rate: float = 0.1) -> Tensor:
    edge_num = int(edge_index.size(1))
    add_num = int(edge_num * add_rate / 2)
    dense_matrix = torch.squeeze(torch_geometric.utils.to_dense_adj(edge_index)).to(device=edge_index.device)
    node_num = dense_matrix.size(0)
    for i in range(add_num):
        source_node = random.randint(0, node_num - 1)
        des_node = random.randint(0, node_num - 1)
        dense_matrix[source_node][des_node] = 1
        dense_matrix[des_node][source_node] = 1
    edge_final = torch_geometric.utils.dense_to_sparse(dense_matrix)[0]
    return edge_final
