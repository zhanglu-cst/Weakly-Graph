import os

import mmcv
import numpy

# from compent.networkx_ops import judge_contain_a_subgraph

root_dataset = r'/remote-home/zhanglu/weakly_molecular/data/DBLP_v1/'
dir_label_martix = r'/remote-home/zhanglu/weakly_molecular/work_dir/DBLP-v1/baseline_ssl_KE/'
path_label_matrix = os.path.join(dir_label_martix, 'label_matrix_itr_0.pkl')

splits = ['train', 'val', 'test']
all_graphs_items = []
train_len = 0
for item_split in splits:
    filename = '{}.pkl'.format(item_split)
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
    number_node = graph.number_of_nodes()
    all_nodes_number.append(number_node)
avg_number_nodes = numpy.mean(all_nodes_number)
print('avg_number_nodes:{}'.format(avg_number_nodes))

filename_key_subgraph = 'key_subgraph_each_class.pkl'
path_key_subgraph = os.path.join(root_dataset, filename_key_subgraph)
all_key_subgraphs = mmcv.load(path_key_subgraph)
number_key_subgraph_each_class = len(all_key_subgraphs[0])
print('number_key_subgraph_each_class:{}'.format(number_key_subgraph_each_class))

all_cover_graphs, all_label_matrix = mmcv.load(path_label_matrix)
print('hit number:{}'.format(len(all_cover_graphs)))
hit_rate = len(all_cover_graphs) / train_len
print('hit rate:{}'.format(hit_rate))
