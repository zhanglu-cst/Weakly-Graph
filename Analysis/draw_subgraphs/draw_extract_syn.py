import os

# import cv2
import mmcv
import networkx
from matplotlib import pyplot as plt

dir = r'/remote-home/zhanglu/weakly_molecular/work_dir/DBLP-v1-NewK/DM_NoSSL/'
taskname = 'DBLP'
class_index = 1

path_init_file = os.path.join(dir, 'key_subgraph_1.pkl')

key_subgraph_each_class = mmcv.load(path_init_file)
key_subgraph_each_class = key_subgraph_each_class.subgraph_each_classes
print(key_subgraph_each_class)


color_maps = ['#0e87cc', '#017a79', '#6a79f7']
cur_color = color_maps[class_index]
print('cur color:{}'.format(cur_color))

for index in range(0, 5):
    cur_subgraph = key_subgraph_each_class[class_index][index]

    networkx.draw_networkx(cur_subgraph, with_labels = False,  # ax = ax,
                           node_color = cur_color)
    filename = 'extract_{}_{}_{}.png'.format(taskname, class_index, index)
    plt.savefig(filename, dpi = 1000, format = 'png')
    # img = cv2.imread(filename)
    # img = img[980:5600, 630:4175]
    # cv2.imwrite('{}.png'.format(index), img)
    # plt.show()
