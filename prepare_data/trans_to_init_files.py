import os.path

import mmcv

from compent.cal_coverage import cal_coverage
from compent.networkx_ops import Graphs_Set, Graph_Dict

dir = r'/remote-home/zhanglu/weakly_molecular/data/DBLP_v1/'
filename_GT = r'key_subgraph_each_class_GTs.pkl'
filename_init = r'key_subgraph_each_class.pkl'
number_class = 2

keep_index_each_class = [[0], [0]]
path_GT = os.path.join(dir, filename_GT)
GTs_list_each_class = mmcv.load(path_GT)

subgraphs_repeat = Graphs_Set()
subgraphs_to_class = Graph_Dict()

for class_index, (item_keysubgraphs_class_list, keep_index_cur_class) in enumerate(
        zip(GTs_list_each_class, keep_index_each_class)):
    item_keysubgraphs_class_list = [item[0] for item in item_keysubgraphs_class_list]
    # item_keysubgraphs_class_list = item_keysubgraphs_class_list[:keep_number]
    ans_item_class_list = []
    for item_index in keep_index_cur_class:
        ans_item_class_list.append(item_keysubgraphs_class_list[item_index])
    for item_subgraph in ans_item_class_list:
        if (item_subgraph not in subgraphs_to_class):
            subgraphs_to_class[item_subgraph] = class_index
        else:
            origin_index = subgraphs_to_class[item_subgraph]
            if (origin_index != class_index):
                subgraphs_repeat.insert_set(item_subgraph)

subgraphs_each_class = [[] for _ in range(number_class)]
for item_subgraph, item_class in subgraphs_to_class:
    if (item_subgraph not in subgraphs_repeat):
        subgraphs_each_class[item_class].append(item_subgraph)

for item_class in subgraphs_each_class:
    print('number key subgraphs cur class:{}'.format(len(item_class)))
    for item in item_class:
        # show_a_graph(item)
        print('nodes:{} edges:{}'.format(item.number_of_nodes(), item.number_of_edges()))
path_target = os.path.join(dir, filename_init)
mmcv.dump(subgraphs_each_class, path_target)

all_subgraphs = []
for item_class in subgraphs_each_class:
    all_subgraphs += item_class

path_big_graphs = os.path.join(dir, 'train.pkl')
big_graphs_list = mmcv.load(path_big_graphs)
big_graphs_list = [item[0] for item in big_graphs_list]
rate = cal_coverage(all_subgraphs, all_bigraphs = big_graphs_list)
