import os
import torch
import random
from torch_geometric.data import Data
import pandas as pd
import numpy as np

dataset_path = '/data/luckytiger/elliptic_bitcoin_dataset'

# random generate train, validate, test mask
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Data Pretreatment
df_features = pd.read_csv(dataset_path + '/elliptic_txs_features.csv', header=None)
df_edges = pd.read_csv(dataset_path + '/elliptic_txs_edgelist.csv')
df_classes = pd.read_csv(dataset_path + '/elliptic_txs_classes.csv')
df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})

# merging dataframes
df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)
classified = df_merge.loc[df_merge['class'].loc[df_merge['class'] != 2].index].drop('txId', axis=1)
unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class'] == 2].index].drop('txId', axis=1)

# storing classified unclassified nodes seperatly for training and testing purpose
classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]
del df_features, df_classes

# all nodes in data
nodes = df_merge[0].values
map_id = {j: i for i, j in enumerate(nodes)}  # mapping nodes to indexes

edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)
edges = edges.astype(int)

edge_index = np.array(edges.values).T
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
weights = torch.tensor([1] * edge_index.shape[1], dtype=torch.double)

# maping txIds to corresponding indexes, to pass node features to the model
node_features = df_merge.drop(['txId'], axis=1).copy()
node_features[0] = node_features[0].map(map_id)
classified_idx = node_features['class'].loc[node_features['class'] != 2].index
classified_0_idx = node_features['class'].loc[node_features['class'] == 0].index
classified_1_idx = node_features['class'].loc[node_features['class'] == 1].index
unclassified_idx = node_features['class'].loc[node_features['class'] == 2].index
# replace unkown class with 0, to avoid having 3 classes, this data/labels never used in training
node_features['class'] = node_features['class'].replace(2, 0)

labels = node_features['class'].values
node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double),
                             dtype=torch.double)

# converting data to PyGeometric graph data format
data_train = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                  y=torch.tensor(labels, dtype=torch.double))  # , adj= torch.from_numpy(np.array(adj))

classified_0_idx_list = classified_0_idx.to_list()
classified_1_idx_list = classified_1_idx.to_list()
class_0_num = len(classified_0_idx_list)
class_1_num = len(classified_1_idx_list)
np.random.shuffle(classified_0_idx_list)
np.random.shuffle(classified_1_idx_list)

# train:val:test = 6:2:2
label0_train_split = int(np.floor(class_0_num * 0.6))
label0_validate_split = int(label0_train_split + np.floor(class_0_num * 0.2))

label1_train_split = int(np.floor(class_1_num * 0.6))
label1_validate_split = int(label1_train_split + np.floor(class_1_num * 0.2))

label0_train = classified_0_idx_list[:label0_train_split]
label0_val = classified_0_idx_list[label0_train_split:label0_validate_split]
label0_test = classified_0_idx_list[label0_validate_split:]

label1_train = classified_1_idx_list[:label1_train_split]
label1_val = classified_1_idx_list[label1_train_split:label1_validate_split]
label1_test = classified_1_idx_list[label1_validate_split:]

# combine 2 label
train_mask = np.append(label0_train, label1_train)
val_mask = np.append(label0_val, label1_val)
test_mask = np.append(label0_test, label1_test)

# save numpy array
np.save(os.path.join(dataset_path, 'train_mask.npy'), train_mask)
np.save(os.path.join(dataset_path, 'val_mask.npy'), val_mask)
np.save(os.path.join(dataset_path, 'test_mask.npy'), test_mask)

print('Mission complete.')
