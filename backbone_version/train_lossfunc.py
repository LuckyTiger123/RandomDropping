import os
import sys
import torch
import pandas as pd
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import backbone_version.model as Md

cuda_device = 6

train_dataset = 'Cora'

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
        self.backbone = backbone
        self.gnn1 = Md.OurModelLayer(feature_num, 64, drop_method, backbone, unbias=unbias, alpha=0.1, K=10,
                                     normalize=True)
        self.gnn2 = Md.OurModelLayer(64, output_num, drop_method, backbone, unbias=unbias, alpha=0.1, K=10,
                                     normalize=True)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0):
        x = self.gnn1(x, edge_index, drop_rate)
        if self.backbone == 'GAT':
            x = F.elu(x)
        else:
            x = F.relu(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    print('the train loss is {}'.format(float(loss)))
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index, drop_rate)
    _, pred = out.max(dim=1)
    train_correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    train_acc = train_correct / int(data.train_mask.sum())
    validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
    validate_acc = validate_correct / int(data.val_mask.sum())
    val_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    test_acc = test_correct / int(data.test_mask.sum())
    test_loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask])
    return train_acc, validate_acc, test_acc, float(val_loss), float(test_loss)


for drop_method in ['Clean', 'DropNode', 'DropEdge', 'Dropout', 'DropMessage']:
    for current_drop_rate in [0.4, 0.5]:
        for backbone in ['GCN']:
            unbias = False
            if drop_method in ['DropNode']:
                unbias = True

            loss_statistics = pd.DataFrame(
                columns=['epoch', 'loss', 'val', 'test', 'val_loss', 'test_loss']
            )

            if unbias:
                unbias_rate = current_drop_rate
            else:
                unbias_rate = 0

            drop_rate = current_drop_rate
            if drop_method == 'Clean':
                drop_rate = 0
            model = Model(dataset.num_features, dataset.num_classes, backbone, drop_method, unbias_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
            epoch_num = 500

            best_val_acc = test_acc = 0
            for epoch in range(epoch_num):
                loss = train()
                train_acc, val_acc, current_test_acc, val_loss, test_loss = test()
                loss_statistics.loc[loss_statistics.shape[0]] = {
                    'epoch': epoch,
                    'loss': round(loss, 4),
                    'val': round(val_acc, 4),
                    'test': round(current_test_acc, 4),
                    'val_loss': round(val_loss, 4),
                    'test_loss': round(test_loss, 4)
                }
                print('For the {} epoch, the train acc is {}, the val acc is {}, the test acc is {}.'.format(epoch,
                                                                                                             train_acc,
                                                                                                             val_acc,
                                                                                                             current_test_acc))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = current_test_acc

            save_path = os.path.join('..', 'result', 'statistic', train_dataset)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            loss_statistics.to_excel(
                os.path.join(save_path,
                             '{}_{}_{}_lossfunc_with_normalization.xlsx'.format(backbone, drop_method,
                                                                                current_drop_rate)))
            print("File saved!")

            print('Mission completes.')
            print('The best val acc is {}, and the test acc is {}.'.format(best_val_acc, test_acc))
