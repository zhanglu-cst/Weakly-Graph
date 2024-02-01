import os

import mmcv
import networkx
from matplotlib import pyplot as plt

dir = r'/remote-home/zhanglu/weakly_molecular/data/DBLP_v1/'

path_init_file = os.path.join(dir, 'key_subgraph_each_class.pkl')

key_subgraph_each_class = mmcv.load(path_init_file)
for class_index in range(0, 2):
    # class_index = 2

    color_maps = ['#0e87cc', '#017a79', '#6a79f7']
    cur_color = color_maps[class_index]
    print('cur color:{}'.format(cur_color))

    cur_subgraph = key_subgraph_each_class[class_index][0]

    # plt.figure(dpi = 1000, figsize = (8, 8))

    networkx.draw_networkx(cur_subgraph, with_labels = False,  # ax = ax,
                           node_color = cur_color)
    plt.savefig('init_{}.png'.format(class_index), dpi = 1000, format = 'png')
    plt.show()
