import pickle
import torch
import torch_geometric.utils as utils
import networkx as nx
from torch import Tensor

x_path = '/data/luckytiger/chinatel/chinatel.x'
y_path = '/data/luckytiger/chinatel/chinatel.y'
graph_path = '/data/luckytiger/chinatel/chinatel.graph'
spilt_path = '/data/luckytiger/chinatel/chinatel0.5_0.2_0.3.split0'

f = open(y_path, 'rb')
y = pickle.load(f)
f.close()
f = open(x_path, 'rb')
x = pickle.load(f)
f.close()
f = open(graph_path, 'rb')
graph = utils.from_networkx(pickle.load(f))
f.close()
f = open(spilt_path, 'rb')
spilt = pickle.load(f)
f.close()
print(x.size())
print(y.size())
print(graph)
