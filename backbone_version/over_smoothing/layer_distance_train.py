import os
import sys
import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

cuda_device = 4
train_dataset = 'Cora'
drop_method = 'Dropout'
drop_rate = 0.5
number_layer = 8
hidden_dimension = 16
backbone = 'GCN'
unbias = False

# random generate train, validate, test mask
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# device selection
device = torch.device('cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')

# collect dataset
dataset = Planetoid(root="/home/luckytiger/TestDataset/{}".format(train_dataset), name=train_dataset,
                    transform=T.NormalizeFeatures())

data = dataset[0].to(device=device)


class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(Md.OurModelLayer(in_channels, out_channels, drop_method, backbone, unbias=unbias))
        else:
            self.convs.append(
                Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            for _ in range(num_layers - 2):
                self.convs.append(
                    Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias))
            self.convs.append(
                Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias))
        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        if self.num_layers == 1:
            return self.convs[0](x, edge_index, drop_rate)

        for _, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, drop_rate)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def cal_dist(self, x: Tensor, edge_index: Adj):
        dist_list = []

        if self.num_layers < 3:
            return dist_list

        previous_H = self.convs[0](x, edge_index)
        norm_previous_H = F.normalize(previous_H, p=1, dim=0)

        for i in range(self.num_layers - 2):
            item = F.relu(previous_H)
            current_H = self.convs[1 + i](item, edge_index)
            norm_current_H = F.normalize(current_H, p=1, dim=0)
            current_dist = torch.norm(norm_current_H - norm_previous_H, p=2)
            dist_list.append(float(current_dist))
            previous_H = current_H
            norm_previous_H = norm_current_H

        return dist_list

    def reset_parameters(self):
        for conv in self.convs:
            conv.special_reset_parameters()
        self.convs[-1].reset_parameters()
        self.convs[0].reset_parameters()


model = Model(dataset.num_features, hidden_dimension, dataset.num_classes, number_layer, backbone, drop_method,
              unbias).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.convs[-1].parameters(), lr=0.005, weight_decay=0.0005)
optimizer2 = torch.optim.Adam(model.convs[0].parameters(), lr=0.005, weight_decay=0.0005)
epoch_num = 200


def train():
    model.train()
    optimizer.zero_grad()
    optimizer2.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()
    optimizer2.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, drop_rate)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    return train_acc, validate_acc, test_acc


best_val_acc = test_acc = 0
for epoch in range(epoch_num):
    train()
    train_acc, val_acc, current_test_acc = test()
    print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch, train_acc,
                                                                                                 val_acc,
                                                                                                 current_test_acc))

    test_acc = current_test_acc

dist_list = model.cal_dist(data.x, data.edge_index)
print(dist_list)
