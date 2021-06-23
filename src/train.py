import os
import sys
import torch
from torch import Tensor
from torch_geometric.datasets import Planetoid
from torch_geometric.typing import Adj
import torch_geometric.transforms as T
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import src.model as ours
import src.loss as loss
import src.utils as utils

train_dataset = 'Cora'
cuda_device = 3
# drop_method = 'DropMessageChannel'
drop_method = 'DropMessageChannel'
sample_number = 1
drop_rate = 0.6
unbias = True
first_layer_output_num = 32

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
    def __init__(self, feature_num, output_num, drop_method, sample_number, unbias: bool = True,
                 first_layer_output_num: int = 8):
        super(Model, self).__init__()
        self.gnn1 = ours.OurModelLayer(feature_num, first_layer_output_num, drop_method, sample_number, unbias=unbias)
        self.gnn2 = ours.OurModelLayer(first_layer_output_num, output_num, drop_method, sample_number, unbias=unbias)

    def forward(self, x: Tensor, edge_index: Adj, drop_rate: float = 0.5):
        x = self.gnn1(x, edge_index, drop_rate)
        x = F.relu(x)
        if self.training and len(x.size()) == 3:
            x = utils.average_agg(x)
        x = self.gnn2(x, edge_index, drop_rate)
        return x

    def reset_parameters(self):
        self.gnn1.reset_parameters()
        self.gnn2.reset_parameters()


model = Model(dataset.num_features, dataset.num_classes, drop_method, sample_number, unbias, first_layer_output_num).to(
    device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
epoch_num = 1000
loss_func = loss.AugmentedCrossEntropy()

model.reset_parameters()
optimizer.zero_grad()
best_validate_rate = 0
test_rate_under_best_validate = 0

for epoch in range(epoch_num):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, drop_rate)
    final_loss = loss_func.calculate(out, data.y, data.train_mask)
    final_loss.backward()
    optimizer.step()

    # validate set
    model.eval()
    with torch.no_grad():
        out_v = model(data.x, data.edge_index)
        _, pred = out_v.max(dim=1)
        validate_correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())
        validate_acc = validate_correct / int(data.val_mask.sum())
        if validate_acc > best_validate_rate:
            best_validate_rate = validate_acc
            test_correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
            test_acc = test_correct / int(data.test_mask.sum())
            test_rate_under_best_validate = test_acc

    print('for the {} epoch, the loss is {}.'.format(epoch, float(final_loss)))

print('Final acc on validate set is {}, and on test set is {}.'.format(best_validate_rate,
                                                                       test_rate_under_best_validate))
