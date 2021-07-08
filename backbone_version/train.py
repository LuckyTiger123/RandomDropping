import os
import sys
import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import backbone_version.model as Md

cuda_device = 2

train_dataset = 'Cora'
drop_method = 'DropMessage'
drop_rate = 0
backbone = 'APPNP'
unbias = True

# random generate train, validate, test mask
random_seed = 1
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


# Model
class Model(torch.nn.Module):
    def __init__(self, feature_num, output_num, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.gnn1 = Md.OurModelLayer(feature_num, 64, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)
        self.gnn2 = Md.OurModelLayer(64, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


if unbias:
    unbias_rate = drop_rate
else:
    unbias_rate = 0

model = Model(dataset.num_features, dataset.num_classes, backbone, drop_method, unbias_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
epoch_num = 1000


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


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
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = current_test_acc

print('Mission completes.')
print('The best val acc is {}, and the test acc is {}.'.format(best_val_acc, test_acc))
