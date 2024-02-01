import mmcv

from compent.networkx_ops import show_a_graph

path = r'/remote-home/zhanglu/weakly_molecular/work_dir/DBLP-v1/baseline_ssl_KE/key_subgraph_0.pkl'

keysubgraphs = mmcv.load(path)

for index, item in enumerate(keysubgraphs):
    print(item)
    class_index = keysubgraphs.subgraph_to_class[item]
    print(class_index)