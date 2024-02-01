import os
import pickle
import random

import mmcv
import networkx
import numpy

dir = 'IMDB-MULTI'
root_to_save = r'/remote-home/zhanglu/weakly_molecular/data/'
DIR_SAVE = os.path.join(root_to_save, dir)
if (os.path.exists(DIR_SAVE) == False):
    os.makedirs(DIR_SAVE)

path_A = os.path.join(dir, '{}_A.txt'.format(dir))
path_indicator = os.path.join(dir, '{}_graph_indicator.txt'.format(dir))
path_labels = os.path.join(dir, '{}_graph_labels.txt'.format(dir))

with open(path_A, 'r') as f:
    lines = f.read().strip().split('\n')

with open(path_indicator, 'r') as f:
    indicator = f.read().strip().split('\n')

with open(path_labels, 'r') as f:
    labels = f.read().strip().split('\n')

labels = [int(item) for item in labels]


def verify(labels):
    set_labels = list(set(labels))
    set_labels = sorted(set_labels)
    print('set labels:{}'.format(set_labels))
    for index, item_label in enumerate(set_labels):
        assert item_label == index


new_labels = []
map_labels = {1: 0, 2: 1, 3: 2}
for item_label in labels:
    new_l = map_labels[item_label]
    new_labels.append(new_l)
verify(new_labels)
labels = new_labels


indicator = [int(item) for item in indicator]
number_of_graphs = numpy.max(indicator)
print('number_of_graphs:{}'.format(number_of_graphs))

print('len lines:{}'.format(len(lines)))
print('len indicator:{}'.format(len(indicator)))
print('len graphs:{}'.format(len(labels)))
assert len(labels) == number_of_graphs

node_number_each_graph = numpy.zeros(number_of_graphs + 1)
for graph_id in indicator:
    node_number_each_graph[graph_id] += 1
print('node_number_each_graph:{}'.format(node_number_each_graph))
cum_sum_each_graph = numpy.cumsum(node_number_each_graph)
print('cum_sum_each_graph:{}'.format(cum_sum_each_graph))

node_id_to_graph_id = {}
for node_id, graph_id in enumerate(indicator, start = 1):
    node_id_to_graph_id[node_id] = graph_id

bar = mmcv.ProgressBar(task_num = len(lines))
node_ids_each_graph = [[] for _ in range(number_of_graphs + 1)]
for i, item_adj in enumerate(lines):
    a, b = item_adj.split(',')
    a = int(a)
    b = int(b)
    graph_id_a = node_id_to_graph_id[a]
    graph_id_b = node_id_to_graph_id[b]
    assert graph_id_a == graph_id_b
    node_id_reduction = cum_sum_each_graph[graph_id_a - 1]
    a = int(a - node_id_reduction)
    b = int(b - node_id_reduction)
    node_ids_each_graph[graph_id_a].append((a, b))
    if (i % 10000 == 0):
        bar.update(num_tasks = 10000)
print()

all_graphs = []

for index, nodes_cur_graph in enumerate(node_ids_each_graph):
    if (index == 0):
        continue
    # print(nodes_cur_graph)
    item_graph = networkx.Graph()
    item_graph.add_edges_from(nodes_cur_graph)
    all_graphs.append(item_graph)

label_items = set(labels)
labels = numpy.array(labels)
for item_label_value in label_items:
    count_cur = numpy.sum(labels == item_label_value)
    print('label value:{} count:{}'.format(item_label_value, count_cur))

assert len(all_graphs) == len(labels)
pairs = []
for item_graph, item_label in zip(all_graphs, labels):
    pairs.append((item_graph, item_label))
random.shuffle(pairs)

train_number = int(number_of_graphs * 0.6)
val_number = int(number_of_graphs * 0.2)

train = pairs[:train_number]
val = pairs[train_number:train_number + val_number]
test = pairs[train_number + val_number:]

train_only_graphs = [item[0] for item in train]

with open(os.path.join(DIR_SAVE, 'train.pkl'), 'wb') as f:
    pickle.dump(train, f)

with open(os.path.join(DIR_SAVE, 'val.pkl'), 'wb') as f:
    pickle.dump(val, f)

with open(os.path.join(DIR_SAVE, 'test.pkl'), 'wb') as f:
    pickle.dump(test, f)

with open(os.path.join(DIR_SAVE, 'train_unlabel.pkl'), 'wb') as f:
    pickle.dump(train_only_graphs, f)
