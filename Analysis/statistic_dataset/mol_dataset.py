import os

import mmcv
import numpy

from compent.molecular_ops import get_mol

# from compent.networkx_ops import judge_contain_a_subgraph

root_dataset = r'/remote-home/zhanglu/weakly_molecular/data/sider_Vascular disorders/'

splits = ['train', 'val', 'test']
all_graphs_items = []
train_len = 0
for item_split in splits:
    filename = '{}.json'.format(item_split)
    path_file = os.path.join(root_dataset, filename)
    top = mmcv.load(path_file)
    if (item_split == 'train'):
        train_len = len(top)
    all_graphs_items += top

number_of_graphs = len(all_graphs_items)
print('number_of_graphs:{}'.format(number_of_graphs))
all_labels = [item[1] for item in all_graphs_items]
all_labels = set(all_labels)
print('number of classes:{}'.format(len(all_labels)))

all_nodes_number = []
for item in all_graphs_items:
    graph = item[0]
    mol = get_mol(graph)
    number_node = mol.GetNumAtoms()
    all_nodes_number.append(number_node)
avg_number_nodes = numpy.mean(all_nodes_number)
print('avg_number_nodes:{}'.format(avg_number_nodes))

filename_key_subgraph = 'key_subgraph_each_class.json'
path_key_subgraph = os.path.join(root_dataset, filename_key_subgraph)
all_key_subgraphs = mmcv.load(path_key_subgraph)
number_key_subgraph_each_class = len(all_key_subgraphs[0])
print('number_key_subgraph_each_class:{}'.format(number_key_subgraph_each_class))

all_keys_list = []
for item_class in all_key_subgraphs:
    all_keys_list += item_class

path_smile_to_subgraph = os.path.join(root_dataset, 'subgraphs.json')
smile_to_subgraph = mmcv.load(path_smile_to_subgraph)
hit_number = 0
for item_smile in smile_to_subgraph:
    subgraphs = smile_to_subgraph[item_smile]
    for item_subgraph in all_keys_list:
        if (item_subgraph in subgraphs):
            hit_number += 1
            break
print('hit number:{}'.format(hit_number))
print('hit rate:{}'.format(hit_number / train_len))
