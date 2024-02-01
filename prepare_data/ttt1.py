import os
import time

import mmcv

from compent.networkx_ops import judge_contain_a_subgraph

dir = r'/remote-home/zhanglu/weakly_molecular/temp_dir/'
all_files = os.listdir(dir)
print(len(all_files))
all_res = []
for index, item_filename in enumerate(all_files):
    path = os.path.join(dir, item_filename)
    obj = mmcv.load(path)
    subgraph, biggraph = obj

    s_time = time.time()
    res = judge_contain_a_subgraph(big_graph = biggraph, subgraph = subgraph)
    e_time = time.time()
    total_time = e_time - s_time
    print('subgraph nodes:{}, big graph nodes:{}, time:{}, res:{}'.format(subgraph.number_of_nodes(),
                                                                          biggraph.number_of_nodes(),
                                                                          total_time, res))
    number_edges_big = biggraph.number_of_edges()
    number_edges_subgraph = subgraph.number_of_edges()
    item_res = (
    total_time, subgraph.number_of_nodes(), biggraph.number_of_nodes(), res, number_edges_big, number_edges_subgraph,
    subgraph,biggraph)
    all_res.append(item_res)
    if (index == 500):
        break

print('------------------')
all_res = sorted(all_res, key = lambda x: x[0], reverse = True)
for item in all_res:
    print(item)

item_sub = all_res[0][-2]
item_big = all_res[0][-1]
from compent.networkx_ops import show_a_graph
show_a_graph(item_sub)
show_a_graph(item_big)