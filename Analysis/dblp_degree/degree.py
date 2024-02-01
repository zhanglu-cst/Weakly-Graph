import os

import mmcv
import numpy

root_dataset = r'/remote-home/zhanglu/weakly_molecular/data/DBLP_v1/'

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

for cur_label in [0, 1]:

    all_max_degrees = []
    all_mean_degrees = []
    for item_label in all_graphs_items:
        item, label = item_label
        if (label == cur_label):
            degrees = list(dict(item.degree()).values())
            max_d = numpy.max(degrees)
            mean_d = numpy.mean(degrees)
            all_max_degrees.append(max_d)
            all_mean_degrees.append(mean_d)

    mean_max = numpy.mean(all_max_degrees)
    mean_mean = numpy.mean(all_mean_degrees)
    print(mean_max, mean_mean)
