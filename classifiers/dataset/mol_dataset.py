import random

import torch
from torch.utils.data import Dataset

from compent import Global_Var
from compent.molecular_ops import build_dgl_graph_from_smile
from ..build import TRAIN_EVAL_DATASET


@TRAIN_EVAL_DATASET.register_module()
class Mol_Dataset(Dataset):
    def __init__(self, smiles, labels, for_train = False, max_upsample = None):
        assert isinstance(smiles, list) and isinstance(labels, list)
        self.labels = torch.tensor(labels).long()
        self.smiles = smiles
        self.all_graphs = []
        self.logger = Global_Var.logger()
        for item_smile in smiles:
            graph = build_dgl_graph_from_smile(item_smile)
            self.all_graphs.append(graph)
        assert len(self.all_graphs) == len(self.labels)
        if for_train:
            self.upsample_for_imbalance(max_upsample = max_upsample)

    def upsample_for_imbalance(self, max_upsample):
        samples_each_class = []
        labels_list = self.labels.tolist()
        number_class = len(set(labels_list))
        for class_index in range(number_class):
            samples_each_class.append([])

        for item_graph, item_label, item_smile in zip(self.all_graphs, labels_list, self.smiles):
            item_label = int(item_label)
            samples_each_class[item_label].append([item_graph, item_label, item_smile])

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
        all_smiles = []
        for item in all_samples:
            all_graphs.append(item[0])
            all_labels.append(item[1])
            all_smiles.append(item[2])
        self.all_graphs = all_graphs
        self.labels = all_labels
        self.smiles = all_smiles

    def __getitem__(self, index):
        return self.all_graphs[index], self.labels[index], self.smiles[index]

    def __len__(self):
        return len(self.all_graphs)
