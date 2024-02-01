import math
import multiprocessing

import dgl
import mmcv
import torch

from compent.global_var import Global_Var
from compent.mul_info import calc_MI
from compent.networkx_ops import networkx_to_DGL, batch_networkx_to_DGL, Graph_Dict, judge_contain_a_subgraph
from compent.utils import move_to_device
from extract_keysubgraph.enumerate_subgraphs_nx import Enumerate_Subgraphs_NetworkX
from .builder import UPDETAR


# sys.path.append('/remote-home/zhanglu/weakly_molecular')
def cal_occurrences_times(item_subgraph, big_graph_list):
    count = 0
    for index, item_big in enumerate(big_graph_list):
        if (judge_contain_a_subgraph(big_graph = item_big, subgraph = item_subgraph)):
            count += 1
    return count


def cal_occurrences_in_each_class(item_subgraph, big_graph_all_classes):
    count_list = []
    for big_graph_cur_class in big_graph_all_classes:
        count_cur_class = cal_occurrences_times(item_subgraph, big_graph_cur_class)
        count_list.append(count_cur_class)
    return count_list


def cal_TF(count_time, instance_number_cur_class):
    tf = count_time / instance_number_cur_class
    return tf


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


def one_process(rank, class_index, queue_input, global_result_list, big_graph_each_class):
    print('rank:{}, start'.format(rank))
    while not queue_input.empty():
        item = queue_input.get()
        item_subgraph, item_avgmi = item
        cur_subgraph_count_all_class = cal_occurrences_in_each_class(item_subgraph, big_graph_each_class)
        item_TF = cal_TF(count_time = cur_subgraph_count_all_class[class_index],
                         instance_number_cur_class = len(big_graph_each_class[class_index]))
        item_IDF = cal_IDF(class_index, cur_subgraph_count_all_class, big_graph_each_class)
        score = item_avgmi * item_TF * item_IDF
        global_result_list.append([item_subgraph, score])
        if (len(global_result_list) % 10 == 0):
            queue_size = queue_input.qsize()
            print('rank:{}, finish:{},queue_size:{}'.format(rank, len(global_result_list), queue_size))
    print('rank:{}, finish'.format(rank))


@UPDETAR.register_module()
class Extract_Syn_Multi_Process():
    def __init__(self, min_node_number = 4, max_node_number = 15, number_classes = 3, keep_subgraph_each_class = 10,
                 max_return_subgraph_number_each_graph = 1000, keep_subgraph_each_instance_topMI = 7,
                 multi_process_number = 30, top_K_mi_selected_for_TF = 300):
        self.enumrate_subgraph = Enumerate_Subgraphs_NetworkX(min_node_number = min_node_number,
                                                              max_node_number = max_node_number,
                                                              max_return_subgraph_number_each_graph = max_return_subgraph_number_each_graph)
        self.number_classes = number_classes
        self.keep_subgraph_each_class = keep_subgraph_each_class
        self.keep_subgraph_each_instance_topMI = keep_subgraph_each_instance_topMI
        self.top_K_mi_selected_for_TF = top_K_mi_selected_for_TF
        self.multi_process_number = multi_process_number
        self.logger = Global_Var.logger()

    def get_feature_of_graph(self, graph, model):
        if (isinstance(graph, list)):
            batch_size = 512
            graph = batch_networkx_to_DGL(graph)
            all_features = []
            for start_index in range(0, len(graph), batch_size):
                end_index = start_index + batch_size
                cur_batch_list = graph[start_index:end_index]
                cur_batch_list = dgl.batch(cur_batch_list)
                cur_batch_list = move_to_device(cur_batch_list)
                _, feature = model(cur_batch_list, return_last_feature = True)
                feature = feature.cpu()
                all_features.append(feature)
            # print('len all features:{} [0] shape:{}'.format(len(all_features), all_features[0].shape))
            all_features = torch.cat(all_features, dim = 0)
            return all_features
        else:
            graph = networkx_to_DGL(graph)
            graph = move_to_device(graph)
            _, feature = model(graph, return_last_feature = True)
            feature = feature.cpu()
            return feature

    def get_MI_of_subgraphs(self, big_graph_feature, all_subgraph_features):
        all_MI = []
        for item_sub_feature in all_subgraph_features:
            item_mi = calc_MI(X = big_graph_feature, Y = item_sub_feature)
            all_MI.append(item_mi)
        return all_MI

    def cal_MI_each_classes(self, graphs_networkx, labels, model):
        subgraphs_mi_each_classes = [[] for _ in range(self.number_classes)]
        print('start cal MI')
        bar = mmcv.ProgressBar(task_num = len(graphs_networkx))
        for item_big_graph, item_label in zip(graphs_networkx, labels):
            cur_big_graph_feature = self.get_feature_of_graph(item_big_graph, model)
            all_subgraphs_cur_big = self.enumrate_subgraph(item_big_graph)
            if (len(all_subgraphs_cur_big) == 0):
                continue
            all_subgraph_features = self.get_feature_of_graph(graph = all_subgraphs_cur_big, model = model)
            all_subgraphs_MI = self.get_MI_of_subgraphs(cur_big_graph_feature, all_subgraph_features)
            one_big_results = []
            for item_subgraph, item_mi in zip(all_subgraphs_cur_big, all_subgraphs_MI):
                one_big_results.append([item_subgraph, item_mi])
            subgraphs_mi_each_classes[item_label].append(one_big_results)
            bar.update()
        print()
        return subgraphs_mi_each_classes

    def __call__(self, graphs, labels, model):
        ITR = Global_Var.get('ITR')
        self.logger.info('start extract keyword, ITR:{}'.format(ITR), key = 'state')
        model.eval()
        big_graph_each_class = [[] for _ in range(self.number_classes)]
        for item_big_graph, item_label in zip(graphs, labels):
            big_graph_each_class[item_label].append(item_big_graph)
        # ---------

        subgraphs_mi_list_each_classes = self.cal_MI_each_classes(graphs, labels, model)
        subgraph_to_mi_each_class = [Graph_Dict() for _ in range(self.number_classes)]
        subgraph_to_count_each_class = [Graph_Dict() for _ in range(self.number_classes)]
        for class_index, subgraphs_each_big_cur_class in enumerate(subgraphs_mi_list_each_classes):
            print('start static class:{}'.format(class_index))
            cur_subgrpah_mi_dict = subgraph_to_mi_each_class[class_index]
            cur_subgraph_count_dict = subgraph_to_count_each_class[class_index]
            process_bar = mmcv.ProgressBar(task_num = len(subgraphs_each_big_cur_class))
            for subgraphs_in_one_big in subgraphs_each_big_cur_class:
                subgraphs_in_one_big = sorted(subgraphs_in_one_big, key = lambda x: x[1], reverse = True)
                subgraphs_in_one_big = subgraphs_in_one_big[:self.keep_subgraph_each_instance_topMI]
                for item_sub, item_MI in subgraphs_in_one_big:
                    if (item_sub in cur_subgrpah_mi_dict):
                        cur_subgrpah_mi_dict[item_sub] += item_MI
                        cur_subgraph_count_dict[item_sub] += 1
                    else:
                        cur_subgrpah_mi_dict[item_sub] = item_MI
                        cur_subgraph_count_dict[item_sub] = 1
                process_bar.update()
            print()
        # --------- avg mi -----------
        subgraph_to_avg_mi = [Graph_Dict() for _ in range(self.number_classes)]
        for subgraph_to_avg_mi_cur_class, subgraph_to_mi_cur_class, subgraph_to_count_cur_class in zip(
                subgraph_to_avg_mi, subgraph_to_mi_each_class, subgraph_to_count_each_class):
            for item_subgraph, item_MI in subgraph_to_mi_cur_class:
                item_count = subgraph_to_count_cur_class[item_subgraph]
                item_avg = item_MI / item_count
                subgraph_to_avg_mi_cur_class[item_subgraph] = item_avg

        # 每个类别取前K个，然后计算这K个的TF，IDF等指标
        list_subgraph_score_each_class = []
        print('start cal final score')
        for class_index, subgraph_to_avg_mi_cur_class in enumerate(subgraph_to_avg_mi):
            print('start class:{}'.format(class_index))
            list_subgraph_avgmi_cur_class = []

            for item in subgraph_to_avg_mi_cur_class:
                list_subgraph_avgmi_cur_class.append(item)
            list_subgraph_avgmi_cur_class = sorted(list_subgraph_avgmi_cur_class, key = lambda x: x[1], reverse = True)
            list_subgraph_avgmi_cur_class = list_subgraph_avgmi_cur_class[:self.top_K_mi_selected_for_TF]

            queue_input = multiprocessing.Queue()
            for item in list_subgraph_avgmi_cur_class:
                queue_input.put(item)
            all_process = []
            global_list = multiprocessing.Manager().list()
            for i in range(self.multi_process_number):
                p = multiprocessing.Process(target = one_process,
                                            args = (i, class_index, queue_input, global_list, big_graph_each_class))
                all_process.append(p)
            for item in all_process:
                item.start()
            for item in all_process:
                item.join()

            list_subgraph_score_cur_class = list(global_list)
            list_subgraph_score_cur_class = sorted(list_subgraph_score_cur_class, key = lambda x: x[1], reverse = True)
            list_subgraph_score_cur_class = list_subgraph_score_cur_class[:self.keep_subgraph_each_class]

            list_subgraph_score_each_class.append(list_subgraph_score_cur_class)

        self.logger.info('finish extract keyword, ITR:{}'.format(ITR), key = 'state')
        return list_subgraph_score_each_class

#
# @UPDETAR.register_module()
# class Extract_Social_Networks(Extract_Syn_Multi_Process):
#     def __init__(self, min_node_number = 4, max_node_number = 20, number_classes = 3,
#                  keep_subgraph_each_class = 10,
#                  max_return_subgraph_number_each_graph = 1000, keep_subgraph_each_instance_topMI = 7,
#                  multi_process_number = 30, top_K_mi_selected_for_TF = 1000):
#         super(Extract_Social_Networks, self).__init__(min_node_number, max_node_number, number_classes,
#                                                       keep_subgraph_each_class,
#                                                       max_return_subgraph_number_each_graph,
#                                                       keep_subgraph_each_instance_topMI,
#                                                       multi_process_number, top_K_mi_selected_for_TF)
#
#     def __call__(self, graphs, labels, model):
#         ITR = Global_Var.get('ITR')
#         self.logger.info('start extract keyword, ITR:{}'.format(ITR), key = 'state')
#         model.eval()
#         big_graph_each_class = [[] for _ in range(self.number_classes)]
#         for item_big_graph, item_label in zip(graphs, labels):
#             big_graph_each_class[item_label].append(item_big_graph)
#         # ---------
#
#         subgraphs_mi_list_each_classes = self.cal_MI_each_classes(graphs, labels, model)
#         subgraph_to_mi_each_class = [Graph_Dict() for _ in range(self.number_classes)]
#         for class_index, subgraphs_each_big_cur_class in enumerate(subgraphs_mi_list_each_classes):
#             print('start static class:{}'.format(class_index))
#             cur_subgrpah_mi_dict = subgraph_to_mi_each_class[class_index]
#             process_bar = mmcv.ProgressBar(task_num = len(subgraphs_each_big_cur_class))
#             for subgraphs_in_one_big in subgraphs_each_big_cur_class:
#                 subgraphs_in_one_big = sorted(subgraphs_in_one_big, key = lambda x: x[1], reverse = True)
#                 subgraphs_in_one_big = subgraphs_in_one_big[:self.keep_subgraph_each_instance_topMI]
#                 for item_sub, item_MI in subgraphs_in_one_big:
#                     if (item_sub in cur_subgrpah_mi_dict):
#                         cur_subgrpah_mi_dict[item_sub] += item_MI
#                     else:
#                         cur_subgrpah_mi_dict[item_sub] = item_MI
#                 process_bar.update()
#             print()
#
#         list_subgraph_score_each_class = []
#         print('start cal final score')
#         for class_index, subgraph_to_mi_cur_class in enumerate(subgraph_to_mi_each_class):
#             print('start class:{}'.format(class_index))
#             list_subgraph_avgmi_cur_class = []
#
#             for item in subgraph_to_mi_cur_class:
#                 list_subgraph_avgmi_cur_class.append(item)
#             list_subgraph_avgmi_cur_class = sorted(list_subgraph_avgmi_cur_class, key = lambda x: x[1], reverse = True)
#
#             list_subgraph_avgmi_cur_class = list_subgraph_avgmi_cur_class[:self.keep_subgraph_each_class]
#
#             list_subgraph_score_each_class.append(list_subgraph_avgmi_cur_class)
#
#         self.logger.info('finish extract keyword, ITR:{}'.format(ITR), key = 'state')
#         return list_subgraph_score_each_class
