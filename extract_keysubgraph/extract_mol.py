import dgl
import mmcv
import numpy
import torch

from compent.global_var import Global_Var
from compent.mul_info import calc_MI
from compent.utils import move_to_device
from .builder import UPDETAR


def cal_occurrences_in_each_class(item_subgraph, smiles_each_class, dataset, number_class):
    ans = numpy.zeros(number_class)
    for class_index, cur_class_smiles in enumerate(smiles_each_class):
        for item_smile in cur_class_smiles:
            if (item_smile not in dataset.smile_to_subgraphs):
                print('\n not exist:{}\n'.format(item_smile))
                continue
            cur_subgraphs = dataset.smile_to_subgraphs[item_smile]
            if (item_subgraph in cur_subgraphs):
                ans[class_index] += 1
    return ans


def cal_TF(occure_each_class, class_index, instance_number_each_class):
    occure_cur_class = occure_each_class[class_index]
    TF = occure_cur_class / instance_number_each_class[class_index]
    return TF


def cal_IDF(occure_each_class, class_index, instance_number_each_class):
    other_total_instance = 0
    subgraph_appear_time = 0
    for cur_class, (item_count_instance, item_count_subgraph) in enumerate(
            zip(instance_number_each_class, occure_each_class)):
        if (cur_class == class_index):
            continue
        other_total_instance += item_count_instance
        subgraph_appear_time += item_count_subgraph
    IDF = other_total_instance / (subgraph_appear_time + 1)
    return IDF


@UPDETAR.register_module()
class Extract_Mol():
    def __init__(self, number_classes = 3, keep_subgraph_each_class = 10,
                 max_return_subgraph_number_each_graph = 500, keep_subgraph_for_TFEX = 8000):
        self.number_classes = number_classes
        self.keep_subgraph_each_class = keep_subgraph_each_class
        self.max_return_subgraph_number_each_graph = max_return_subgraph_number_each_graph
        self.keep_subgraph_for_TFEX = keep_subgraph_for_TFEX
        self.logger = Global_Var.logger()
        print('Global_Var:{}'.format(Global_Var.GLOBAL_VARS_DICT.keys()))
        self.dataset = Global_Var.get('dataset')

    def get_MI_of_subgraphs(self, big_graph_feature, all_subgraph_features):
        all_MI = []
        for item_sub_feature in all_subgraph_features:
            item_mi = calc_MI(X = big_graph_feature, Y = item_sub_feature)
            all_MI.append(item_mi)
        return all_MI

    def get_feature_of_graph(self, graph, model):
        if (isinstance(graph, list) or isinstance(graph, set)):
            batch_size = 256
            ans = []
            for item_subgraph in graph:
                item_dgl = self.dataset.subgraph_smile_to_dgl_graphs[item_subgraph]
                if (item_dgl is not None):
                    ans.append(item_dgl)
            # print('ans len:{}'.format(len(ans)))
            # if (len(ans) > self.max_return_subgraph_number_each_graph):
            #     print('exceed')
            # graph = [self.dataset.subgraph_smile_to_dgl_graphs[item] for item in graph]
            graph = ans[:self.max_return_subgraph_number_each_graph]
            # print('len graph:{}'.format(len(graph)))
            all_features = []
            # print('len all subgraphs:{}'.format(len(graph)))
            for start_index in range(0, len(graph), batch_size):
                end_index = start_index + batch_size
                cur_batch_list = graph[start_index:end_index]
                # print('len cur_batch_list:{}'.format(len(cur_batch_list)))
                cur_batch_list = dgl.batch(cur_batch_list)
                cur_batch_list = move_to_device(cur_batch_list)
                _, feature = model(cur_batch_list, return_last_feature = True)
                feature = feature.cpu()
                all_features.append(feature)
            if (len(all_features) == 0):
                return None
            else:
                all_features = torch.cat(all_features, dim = 0)
                return all_features
        else:
            graph_dgl = self.dataset.smile_to_graph[graph]
            # print(graph)
            graph_dgl = move_to_device(graph_dgl)
            _, feature = model(graph_dgl, return_last_feature = True)
            feature = feature.cpu()

            return feature

    def cal_MI_each_classes(self, smile_to_subgraphs, graphs, labels, model):
        # dump_filename = 'temp_dump_mi.pkl'
        # if (os.path.exists(dump_filename)):
        #     print('load mi from file')
        #     subgraphs_mi_each_classes = mmcv.load(dump_filename)
        #     return subgraphs_mi_each_classes
        # else:
        subgraphs_mi_each_classes = [[] for _ in range(self.number_classes)]
        print('start cal MI')
        bar = mmcv.ProgressBar(task_num = len(graphs))
        for item_big_graph, item_label in zip(graphs, labels):
            try:
                cur_big_graph_feature = self.get_feature_of_graph(item_big_graph, model)
            except:
                continue
            if (item_big_graph not in smile_to_subgraphs):
                bar.update()
                print('\nnot exist:{}\n'.format(item_big_graph))
                continue
            all_subgraphs_cur_big = smile_to_subgraphs[item_big_graph]
            if (len(all_subgraphs_cur_big) == 0):
                bar.update()
                continue
            all_subgraph_features = self.get_feature_of_graph(graph = all_subgraphs_cur_big, model = model)
            if (all_subgraph_features is None):
                bar.update()
                continue
            all_subgraphs_MI = self.get_MI_of_subgraphs(cur_big_graph_feature, all_subgraph_features)
            # one_big_results = []
            # for item_subgraph, item_mi in zip(all_subgraphs_cur_big, all_subgraphs_MI):
            #     one_big_results.append([item_subgraph, item_mi])
            for item_subgraph, item_mi in zip(all_subgraphs_cur_big, all_subgraphs_MI):
                subgraphs_mi_each_classes[item_label].append([item_subgraph, item_mi])
            bar.update()
        print()
        # mmcv.dump(obj = subgraphs_mi_each_classes, file = dump_filename)
        return subgraphs_mi_each_classes

    def __call__(self, graphs, labels, model):
        ITR = Global_Var.get('ITR')
        self.logger.info('start extract key subgraphs, ITR:{}'.format(ITR), key = 'state')
        model.eval()
        smile_to_subgraphs = self.dataset.smile_to_subgraphs
        subgraphs_mi_each_classes = self.cal_MI_each_classes(smile_to_subgraphs, graphs, labels, model)
        avg_subgraph_mi_each_classes = []
        self.logger.info('start cal avg MI')
        for class_index, subgraph_mi_cur_class in enumerate(subgraphs_mi_each_classes):
            subgraph_to_mi_sum_cur_class = {}
            subgraph_to_count_cur_class = {}
            for item_subgraph, item_mi in subgraph_mi_cur_class:
                if (item_subgraph in subgraph_to_mi_sum_cur_class):
                    subgraph_to_mi_sum_cur_class[item_subgraph] += item_mi
                    subgraph_to_count_cur_class[item_subgraph] += 1
                else:
                    subgraph_to_mi_sum_cur_class[item_subgraph] = item_mi
                    subgraph_to_count_cur_class[item_subgraph] = 1
            subgraph_to_avg_mi_cur_class = {}
            for item_subgraph, mi_sum in subgraph_to_mi_sum_cur_class.items():
                count = subgraph_to_count_cur_class[item_subgraph]
                avg_mi = mi_sum / count
                subgraph_to_avg_mi_cur_class[item_subgraph] = avg_mi
            print('class:{}, subgraph_to_avg_mi_cur_class:{}'.format(class_index, len(subgraph_to_avg_mi_cur_class)))
            avg_subgraph_mi_each_classes.append(subgraph_to_avg_mi_cur_class)

        smiles_each_class = [[] for _ in range(self.number_classes)]
        for item_graph, item_label in zip(graphs, labels):
            smiles_each_class[item_label].append(item_graph)

        instance_number_each_class = [len(item) for item in smiles_each_class]
        self.logger.info('start cal final score')
        extract_res = []
        for class_index, (smiles_cur_class, subgraph_to_avg_mi_cur_class) in enumerate(
                zip(smiles_each_class, avg_subgraph_mi_each_classes)):
            self.logger.info('start class:{}'.format(class_index))
            subgraph_scores_cur_class = []
            # -------
            # all_subgraphs_voc = set()
            # for item_smile in smiles_cur_class:
            #     cur_subgraphs = self.dataset.smile_to_subgraphs[item_smile]
            #     all_subgraphs_voc.update(cur_subgraphs)
            # print('class_index:{},all_subgraphs_voc len:{}'.format(class_index, len(all_subgraphs_voc)))

            # -----
            all_subgraphs_mi_list = list(subgraph_to_avg_mi_cur_class.items())
            print('class_index:{},all_subgraphs_voc origin len:{}'.format(class_index, len(all_subgraphs_mi_list)))
            all_subgraphs_mi_list = sorted(all_subgraphs_mi_list, key = lambda x: x[1], reverse = True)
            all_subgraphs_mi_list = all_subgraphs_mi_list[:self.keep_subgraph_for_TFEX]
            all_subgraphs_voc = [item[0] for item in all_subgraphs_mi_list]
            print('class_index:{},all_subgraphs_voc filger len:{}'.format(class_index, len(all_subgraphs_mi_list)))

            bar = mmcv.ProgressBar(task_num = len(all_subgraphs_voc))
            # count_find = 0
            # count_miss = 0
            for index, item_subgraph in enumerate(all_subgraphs_voc):
                occure_each_class = cal_occurrences_in_each_class(item_subgraph, smiles_each_class, self.dataset,
                                                                  self.number_classes)
                item_TF = cal_TF(occure_each_class, class_index, instance_number_each_class)
                item_Ex = cal_IDF(occure_each_class, class_index, instance_number_each_class)
                item_mi = subgraph_to_avg_mi_cur_class[item_subgraph]
                # if (item_subgraph in subgraph_to_avg_mi_cur_class):
                #     item_mi = subgraph_to_avg_mi_cur_class[item_subgraph]
                #     count_find += 1
                # else:
                #     item_mi = 0
                # count_miss += 1
                # print('ignore, not exist in avg mi: {}'.format(item_subgraph))
                # item_mi = subgraph_to_avg_mi_cur_class.get(item_subgraph, 0)
                score = item_TF * item_Ex * item_mi
                subgraph_scores_cur_class.append([item_subgraph, score])
                if (index % 100 == 0):
                    bar.update(num_tasks = 100)
                    # print()
                    # print('find:{}, miss:{}'.format(count_find, count_miss))
            print()
            subgraph_scores_cur_class = sorted(subgraph_scores_cur_class, key = lambda x: x[1], reverse = True)
            subgraph_scores_cur_class = subgraph_scores_cur_class[:self.keep_subgraph_each_class]
            extract_res.append(subgraph_scores_cur_class)
        return extract_res
