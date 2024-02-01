import os.path

import mmcv

dir = r'/remote-home/zhanglu/weakly_molecular/data/sider_Vascular disorders/'
filename_GT = r'key_subgraph_each_class_GTs.json'
filename_init = r'key_subgraph_each_class.json'
number_class = 2

keep_number = 3
path_GT = os.path.join(dir, filename_GT)
GTs_list_each_class = mmcv.load(path_GT)

subgraphs_repeat = set()
subgraphs_to_class = {}

for class_index, item_class_list in enumerate(GTs_list_each_class):
    item_class_list = [item[0] for item in item_class_list]
    item_class_list = item_class_list[:keep_number]
    for item_subgraph in item_class_list:
        if (item_subgraph not in subgraphs_to_class):
            subgraphs_to_class[item_subgraph] = class_index
        else:
            origin_index = subgraphs_to_class[item_subgraph]
            if (origin_index != class_index):
                subgraphs_repeat.add(item_subgraph)

graphs_each_class = [[] for _ in range(number_class)]
for item_subgraph, item_class in subgraphs_to_class.items():
    if (item_subgraph not in subgraphs_repeat):
        graphs_each_class[item_class].append(item_subgraph)

for item_class in graphs_each_class:
    print(item_class)
path_target = os.path.join(dir, filename_init)
mmcv.dump(graphs_each_class, path_target)
