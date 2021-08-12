import os
import numpy as np
from operator import itemgetter

file_director = '/data/luckytiger/ppd_cuishou'
date_month = '05'
date_day = '12'
save_file = True

label_path = os.path.join(file_director, 'cuishou_sample.csv')
node_path = os.path.join(file_director, 'file_all', 'zd_node_feature_{}{}.csv'.format(date_month, date_day))
edge_path = os.path.join(file_director, 'edges', '2020-{}-{}.csv'.format(date_month, date_day))

# fix random seed
random_seed = 0
np.random.seed(random_seed)

node_map = {}
node_list = []
node_num = 0
f_node = open(node_path)
skip = f_node.readline()
line = f_node.readline()
while line:
    item = line.split(',')
    if item[0] == '2020-{}-{}'.format(date_month, date_day):
        node_map[item[1]] = node_num
        node_list.append([int(i) for i in item[2:]])
        node_num += 1
    line = f_node.readline()
f_node.close()

label_list = []
label_mask = []
label_num = 0
label_map = {}
label_0_0_list = []
label_0_1_list = []
label_1_0_list = []
label_1_1_list = []

f_label = open(label_path)
skip = f_label.readline()
line = f_label.readline()
while line:
    item = line.split(',')
    if item[1] == '2020-{}-{}'.format(date_month, date_day):
        if node_map.get(item[0]) is None:
            line = f_label.readline()
            continue
        index = node_map[item[0]]
        label_mask.append(index)
        if int(item[2]) == 1 and int(item[3]) == 1:
            label_1_1_list.append(index)
        elif int(item[2]) == 1 and int(item[3]) == 0:
            label_1_0_list.append(index)
        elif int(item[2]) == 0 and int(item[3]) == 1:
            label_0_1_list.append(index)
        else:
            label_0_0_list.append(index)
        label_map[str(index)] = label_num
        label_list.append([int(item[2]), int(item[3])])
        label_num += 1
    line = f_label.readline()
f_label.close()

edge_set = set()
edge_index = []
edge_num = 0
f_edge = open(edge_path)
skip = f_edge.readline()
line = f_edge.readline()
while line:
    item = line.split(',')
    if node_map.get(item[0]) is not None and node_map.get(item[1]) is not None:
        source_index = node_map[item[0]]
        target_index = node_map[item[1]]
        edge_flag = str(source_index) + '-' + str(target_index)
        if edge_flag not in edge_set:
            edge_index.append([source_index, target_index])
            edge_num += 1
    line = f_edge.readline()
f_edge.close()
edge_index.sort(key=itemgetter(0, 1))

np_x = np.array(node_list)
np_y = np.array(label_list)
np_edge_index = np.array(edge_index).T
np_label_0_0_index = np.array(label_0_0_list, dtype=np.int64)
np_label_0_1_index = np.array(label_0_1_list, dtype=np.int64)
np_label_1_0_index = np.array(label_1_0_list, dtype=np.int64)
np_label_1_1_index = np.array(label_1_1_list, dtype=np.int64)

# spilt 5:3:2
label_0_0_num = np_label_0_0_index.shape[0]
label_0_1_num = np_label_0_1_index.shape[0]
label_1_0_num = np_label_1_0_index.shape[0]
label_1_1_num = np_label_1_1_index.shape[0]

label_0_0_train_split = int(np.floor(label_0_0_num * 0.5))
label_0_0_validate_split = int(label_0_0_train_split + np.floor(label_0_0_num * 0.3))

label_0_1_train_split = int(np.floor(label_0_1_num * 0.5))
label_0_1_validate_split = int(label_0_1_train_split + np.floor(label_0_1_num * 0.3))

label_1_0_train_split = int(np.floor(label_1_0_num * 0.5))
label_1_0_validate_split = int(label_1_0_train_split + np.floor(label_1_0_num * 0.3))

label_1_1_train_split = int(np.floor(label_1_1_num * 0.5))
label_1_1_validate_split = int(label_1_1_train_split + np.floor(label_1_1_num * 0.3))

label_0_0_train = np_label_0_0_index[:label_0_0_train_split]
label_0_0_val = np_label_0_0_index[label_0_0_train_split:label_0_0_validate_split]
label_0_0_test = np_label_0_0_index[label_0_0_validate_split:]

label_0_1_train = np_label_0_1_index[:label_0_1_train_split]
label_0_1_val = np_label_0_1_index[label_0_1_train_split:label_0_1_validate_split]
label_0_1_test = np_label_0_1_index[label_0_1_validate_split:]

label_1_0_train = np_label_1_0_index[:label_1_0_train_split]
label_1_0_val = np_label_1_0_index[label_1_0_train_split:label_1_0_validate_split]
label_1_0_test = np_label_1_0_index[label_1_0_validate_split:]

label_1_1_train = np_label_1_1_index[:label_1_1_train_split]
label_1_1_val = np_label_1_1_index[label_1_1_train_split:label_1_1_validate_split]
label_1_1_test = np_label_1_1_index[label_1_1_validate_split:]

# combine labels
train_mask = np.append(label_0_0_train, label_0_1_train)
train_mask = np.append(train_mask, label_1_0_train)
train_mask = np.append(train_mask, label_1_1_train)

val_mask = np.append(label_0_0_val, label_0_1_val)
val_mask = np.append(val_mask, label_1_0_val)
val_mask = np.append(val_mask, label_1_1_val)

test_mask = np.append(label_0_0_test, label_0_1_test)
test_mask = np.append(test_mask, label_1_0_test)
test_mask = np.append(test_mask, label_1_1_test)

label_train_mask = []
for item in train_mask:
    label_train_mask.append(label_map[str(item)])
label_val_mask = []
for item in val_mask:
    label_val_mask.append(label_map[str(item)])
label_test_mask = []
for item in test_mask:
    label_test_mask.append(label_map[str(item)])

label_train_mask = np.array(label_train_mask, dtype=np.int64)
label_val_mask = np.array(label_val_mask, dtype=np.int64)
label_test_mask = np.array(label_test_mask, dtype=np.int64)

if save_file:
    save_path = os.path.join(file_director, 'graph', '{}_{}'.format(date_month, date_day))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'node_x.npy'), np_x)
    np.save(os.path.join(save_path, 'node_y.npy'), np_y)
    np.save(os.path.join(save_path, 'edge_index.npy'), np_edge_index)
    np.save(os.path.join(save_path, 'train_mask.npy'), train_mask)
    np.save(os.path.join(save_path, 'val_mask.npy'), val_mask)
    np.save(os.path.join(save_path, 'test_mask.npy'), test_mask)
    np.save(os.path.join(save_path, 'label_train_mask.npy'), label_train_mask)
    np.save(os.path.join(save_path, 'label_val_mask.npy'), label_val_mask)
    np.save(os.path.join(save_path, 'label_test_mask.npy'), label_test_mask)
    f_statistic = open(os.path.join(save_path, 'statistic'), 'w')
    f_statistic.write('Date: {}-{}\n'.format(date_month, date_day))
    f_statistic.write('Node Num: {}\n'.format(node_num))
    f_statistic.write('Label num: {}\n'.format(label_num))
    f_statistic.write('Edge num: {}\n'.format(edge_num))
    f_statistic.write('Label 0 0 num: {}\n'.format(label_0_0_num))
    f_statistic.write('Label 0 1 num: {}\n'.format(label_0_1_num))
    f_statistic.write('Label 1 0 num: {}\n'.format(label_1_0_num))
    f_statistic.write('Label 1 1 num: {}\n'.format(label_1_1_num))
    f_statistic.close()

print('Mission completes.')
