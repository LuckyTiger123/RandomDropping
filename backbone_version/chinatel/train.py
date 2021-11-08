import os
import sys
import torch
import pickle
import torch_geometric.nn as nn
from torch import Tensor
from torch_geometric.typing import Adj
import torch.nn.functional as F
import torch_geometric.utils as utils

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

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

cuda_device = 9
train_dataset = 'chinatel'
drop_method = 'Dropout'
drop_rate = 0.1
backbone = 'GAT'
hidden_dimensions = 64
num_layers = 3
unbias = False

# random generate train, validate, test mask
random_seed = 1
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
x = x.to(device)
y = y.to(device)
edge_index = graph.edge_index.to(device)
train_mask = spilt['train'].to(device)
val_mask = spilt['val'].to(device)
test_mask = spilt['test'].to(device)


# Model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias))
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            self.batch_norms.append(nn.BatchNorm(hidden_channels))
        self.convs.append(Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, drop_rate)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


if unbias:
    unbias_rate = drop_rate
else:
    unbias_rate = 0

model = Model(261, hidden_dimensions, 2, num_layers, backbone, drop_method, unbias_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
epoch_num = 500
model.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index, drop_rate)
    loss = F.cross_entropy(out[train_mask], y[train_mask], weight=Tensor([1, 2]).to(device))
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(x, edge_index, drop_rate)
    _, pred = out.max(dim=1)
    # acc
    train_correct = int(pred[train_mask].eq(y[train_mask]).sum().item())
    train_acc = train_correct / len(train_mask)
    validate_correct = int(pred[val_mask].eq(y[val_mask]).sum().item())
    validate_acc = validate_correct / len(val_mask)
    test_correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
    test_acc = test_correct / len(test_mask)

    # f1
    train_1_num = y[train_mask][y[train_mask] == 1].size(0)
    train_pred_1_num = pred[train_mask][pred[train_mask] == 1].size(0)
    train_correct_1_num = pred[train_mask][y[train_mask] == 1].eq(1).sum().item()
    train_f1 = 2 * train_correct_1_num * train_correct_1_num / (0.0001 + (train_correct_1_num * train_1_num) + (
            train_correct_1_num * train_pred_1_num))
    val_1_num = y[val_mask][y[val_mask] == 1].size(0)
    val_pred_1_num = pred[val_mask][pred[val_mask] == 1].size(0)
    val_correct_1_num = pred[val_mask][y[val_mask] == 1].eq(1).sum().item()
    val_f1 = 2 * val_correct_1_num * val_correct_1_num / (0.0001 + (val_correct_1_num * val_1_num) + (
            val_correct_1_num * val_pred_1_num))
    test_1_num = y[test_mask][y[test_mask] == 1].size(0)
    test_pred_1_num = pred[test_mask][pred[test_mask] == 1].size(0)
    test_correct_1_num = pred[test_mask][y[test_mask] == 1].eq(1).sum().item()
    test_f1 = 2 * test_correct_1_num * test_correct_1_num / (0.0001 + (test_correct_1_num * test_1_num) + (
            test_correct_1_num * test_pred_1_num))

    return train_acc, validate_acc, test_acc, train_f1, val_f1, test_f1


best_val_acc = best_test_acc = 0
best_val_f1 = best_test_f1 = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, test_acc, train_f1, val_f1, test_f1 = test()
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 test_acc))
    print('The train f1 is {}, the val f1 is {}, the test f1 is {}.'.format(train_f1, val_f1, test_f1))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_test_f1 = test_f1

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_acc, best_test_acc))
print('The best val f1 is {}, and the test f1 is {}.'.format(best_val_f1, best_test_f1))
