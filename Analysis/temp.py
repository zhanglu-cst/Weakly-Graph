import mmcv

path = r'/remote-home/zhanglu/weakly_molecular/data/syn_version5/key_subgraph_each_class_GTs.pkl'
path2 = r'/remote-home/zhanglu/weakly_molecular/data/syn_version5/key_subgraph_each_class.pkl'

origin = mmcv.load(path)

new = []
for item_class in origin:
    keep = item_class[-1:]
    new.append(keep)

mmcv.dump(obj = new, file = path2)
