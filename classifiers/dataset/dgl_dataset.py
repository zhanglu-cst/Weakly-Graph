import random

import torch
from torch.utils.data import Dataset

from compent import Global_Var
from ..build import TRAIN_EVAL_DATASET


@TRAIN_EVAL_DATASET.register_module()
class DGL_Dataset(Dataset):
    def __init__(self, graphs_dgl, labels, for_train = False, max_upsample = None):
        assert isinstance(graphs_dgl, list), type(graphs_dgl)
        assert isinstance(labels, list) or isinstance(labels, torch.Tensor)
        assert len(graphs_dgl) == len(labels)
        self.labels = torch.tensor(labels).long()
        self.all_graphs = graphs_dgl
        self.logger = Global_Var.logger()
        if for_train:
            self.upsample_for_imbalance(max_upsample = max_upsample)

    def upsample_for_imbalance(self, max_upsample):
        samples_each_class = []
        labels_list = self.labels.tolist()
        number_class = len(set(labels_list))
        for class_index in range(number_class):
            samples_each_class.append([])

        print('set labels:{}'.format(set(labels_list)))
        print('number of classes:{}'.format(number_class))
        for item_graph, item_label in zip(self.all_graphs, labels_list):
            item_label = int(item_label)
            samples_each_class[item_label].append([item_graph, item_label])

        self.logger.info('perform upsample for imbalance', key = 'state')
        max_data_point_number = 0
        count_samples_origin = []
        for class_index, data_cur_class in enumerate(samples_each_class):
            count_samples_origin.append(len(data_cur_class))
            max_data_point_number = max(max_data_point_number, len(data_cur_class))
        self.logger.info('origin count samples each class:{}'.format(count_samples_origin), key = 'state')
        self.logger.info('origin total count:{}'.format(sum(len(item) for item in samples_each_class)), key = 'state')

        count_samples_after_upsample = []
        for class_index, data_cur_class in enumerate(samples_each_class):
            unsample_times = max_data_point_number // len(data_cur_class)
            if (max_upsample is not None):
                unsample_times = min(max_upsample, unsample_times)
            unsample_times = max(1, unsample_times)
            # count_samples['train/upsample_time_class_{}'.format(class_index)] = unsample_times
            samples_each_class[class_index] = data_cur_class * unsample_times
            count_samples_after_upsample.append(len(samples_each_class[class_index]))
        self.logger.info(
                'after upsample, samples each class:{}'.format(count_samples_after_upsample, key = 'state'))
        all_samples = []
        for class_index, data_cur_class in enumerate(samples_each_class):
            all_samples += data_cur_class
        self.logger.info('after upsample, total samples:{}'.format(len(all_samples)), key = 'state')

        random.shuffle(all_samples)
        all_graphs = []
        all_labels = []
        for item in all_samples:
            all_graphs.append(item[0])
            all_labels.append(item[1])
        self.all_graphs = all_graphs
        self.labels = all_labels

    def __getitem__(self, index):
        return self.all_graphs[index], self.labels[index]

    def __len__(self):
        return len(self.all_graphs)
