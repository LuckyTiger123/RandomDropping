import os
import sys
import torch
from torch import Tensor
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import backbone_version.model as Md

cuda_device = 1
train_dataset = 'ogbn-arxiv'
drop_method = 'Dropout'
drop_rate = 0
backbone = 'GCN'
hidden_dimensions = 256
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
dataset = PygNodePropPredDataset(name=train_dataset, root='/home/luckytiger/TestDataset', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()
data = data.to(device)


# Model
class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, backbone, drop_method, unbias):
        super(Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(Md.OurModelLayer(in_channels, hidden_channels, drop_method, backbone, unbias=unbias))
        for _ in range(num_layers - 2):
            self.convs.append(Md.OurModelLayer(hidden_channels, hidden_channels, drop_method, backbone, unbias=unbias))
        self.convs.append(Md.OurModelLayer(hidden_channels, out_channels, drop_method, backbone, unbias=unbias))

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, drop_rate)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


split_idx = dataset.get_idx_split()
train_idx = split_idx['train'].to(device)

if unbias:
    unbias_rate = drop_rate
else:
    unbias_rate = 0

model = Model(data.num_features, hidden_dimensions, dataset.num_classes, num_layers, backbone, drop_method,
              unbias_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
epoch_num = 1000

evaluator = Evaluator(name='ogbn-arxiv')
model.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t, drop_rate)
    loss = F.cross_entropy(out[train_idx], data.y.squeeze(1)[train_idx])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


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
