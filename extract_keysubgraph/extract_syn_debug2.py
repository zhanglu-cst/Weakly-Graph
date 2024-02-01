import math
import os

import mmcv
from mmcv import Config

from compent import Global_Var, Logger_Wandb
from dataset import build_dataset, build_key_subgraph
from pseudo_label import build_pseudo_label_assigner

cfg = Config.fromfile('config/syn/cfg_syn.py')
cfg.logger.project = 'syn_small'
cfg.logger.name = 'debug_extractor'
cfg.work_dir = os.path.join('./work_dir', cfg.logger.project, cfg.logger.name)

logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)

syn_dataset = build_dataset(cfg.dataset.train)

list_subgraph_mi_each_class = mmcv.load('list_subgraph_mi_each_class.pkl')
list_subgraph_count_all_class = mmcv.load('list_subgraph_count_all_class.pkl')


def cal_TF(count_time, instance_number_cur_class):
    item_TF = count_time / instance_number_cur_class
    # item_TF = math.tanh(item_TF)
    return item_TF


def cal_IDF(class_index, cur_subgraph_count_all_class, intance_each_class):
    # sum_count = sum(len(item) for item in cur_subgraph_count_all_class)
    # return 1.0 / sum_count
    other_total_instance = 0
    subgraph_appear_time = 0
    for cur_class, (item_count_instance, item_count_subgraph) in enumerate(
            zip(intance_each_class, cur_subgraph_count_all_class)):
        if (cur_class == class_index):
            continue
        other_total_instance += len(item_count_instance)
        subgraph_appear_time += item_count_subgraph
    IDF = math.log(other_total_instance / (subgraph_appear_time + 1))
    return IDF


big_graph_each_class = [[] for _ in range(3)]
for item_big_graph, item_label in zip(syn_dataset.graphs_networkx, syn_dataset.labels):
    big_graph_each_class[item_label].append(item_big_graph)

list_subgraph_score_all_class = []
for class_index, (subgraph_mi_cur_class, subgraph_count_cur_class) in enumerate(
        zip(list_subgraph_mi_each_class, list_subgraph_count_all_class)):
    list_subgraph_score_cur_class = []
    for item_subgraph_mi, item_count_list in zip(subgraph_mi_cur_class, subgraph_count_cur_class):
        item_subgraph, item_avgmi = item_subgraph_mi
        item_TF = cal_TF(count_time = item_count_list[class_index],
                         instance_number_cur_class = len(big_graph_each_class[class_index]))
        item_IDF = cal_IDF(class_index, item_count_list, big_graph_each_class)
        score = item_avgmi * item_TF * item_IDF
        list_subgraph_score_cur_class.append([item_subgraph, score])
    list_subgraph_score_cur_class = sorted(list_subgraph_score_cur_class, key = lambda x: x[1], reverse = True)
    list_subgraph_score_cur_class = list_subgraph_score_cur_class[:50]
    # for item in list_subgraph_score_cur_class:
    #     show_a_graph(item[0])
    list_subgraph_score_all_class.append(list_subgraph_score_cur_class)
print('finish list_subgraph_score_all_class')

key_subgraph = build_key_subgraph(cfg.key_subgraph)
key_subgraph.update_key_subgraph(list_subgraph_score_all_class)
pseudo_label_assigner = build_pseudo_label_assigner(cfg.pseudo_label)
all_assign_graphs, all_assign_labels = pseudo_label_assigner(syn_dataset, key_subgraph)
syn_dataset.cal_pseudo_quality(pseudo_graphs = all_assign_graphs, pseudo_labels = all_assign_labels)
