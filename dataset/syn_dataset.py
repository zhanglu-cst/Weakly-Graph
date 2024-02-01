import os

import mmcv

from compent.global_var import Global_Var
from compent.networkx_ops import networkx_to_DGL,Graph_Dict
from compent.utils import add_pre_key
from pseudo_label.metric import cal_metric_multi_class
from .builder import DATASET


@DATASET.register_module()
class SYN_Dataset():
    def __init__(self, root_dir, split):
        filename = '{}.pkl'.format(split)
        path_file = os.path.join(root_dir, filename)
        list_all = mmcv.load(file = path_file)
        self.graphs_networkx = []
        self.graphs_dgl = []
        self.labels = []
        self.logger = Global_Var.logger()
        for item in list_all:
            self.graphs_networkx.append(item[0])
            self.labels.append(item[1])
            self.graphs_dgl.append(networkx_to_DGL(item[0]))

        print('load graph len:{}'.format(len(self.graphs_dgl)))

        self.graphX_to_label = Graph_Dict()
        for item_graph, item_label in zip(self.graphs_networkx, self.labels):
            self.graphX_to_label[item_graph] = item_label

    def cal_pseudo_quality(self, pseudo_graphs, pseudo_labels, pre_key = 'pseudo_quality'):
        hit_rate = len(pseudo_graphs) / len(self.graphs_networkx)
        hit_number = len(pseudo_graphs)
        gt_labels = []
        for item_graph in pseudo_graphs:
            item_gt = self.graphX_to_label[item_graph]
            gt_labels.append(item_gt)
        metric = cal_metric_multi_class(all_pred_class = pseudo_labels, all_gt_labels = gt_labels)
        log_dict = {'hit_rate': hit_rate, 'hit_number': hit_number}
        log_dict.update(metric)
        log_dict = add_pre_key(log_dict, pre_key = pre_key)
        self.logger.dict(log_dict)
        return log_dict

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = SYN_Dataset(root_dir = './data/syn', split = 'train')
