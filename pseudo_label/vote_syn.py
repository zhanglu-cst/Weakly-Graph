import multiprocessing

import numpy
from mmcv import ProgressBar

from compent import Global_Var
from compent.networkx_ops import judge_contain_a_subgraph
from .build import PSEUDO_LABEL_ASSIGNER


@PSEUDO_LABEL_ASSIGNER.register_module()
class Vote_Syn():
    def __init__(self, number_classes = 3):
        super(Vote_Syn, self).__init__()
        self.logger = Global_Var.logger()
        self.number_classes = number_classes

    def __call__(self, syn_dataset, key_subgraph):
        all_assign_graphs = []
        all_assign_labels = []
        self.logger.info('assigning pseudo labels...')
        bar = ProgressBar(task_num = len(syn_dataset.graphs_networkx))
        for item_graph in syn_dataset.graphs_networkx:
            scores_each_class = numpy.zeros(self.number_classes)
            for item_key, item_class in key_subgraph.subgraph_to_class.items():
                if (judge_contain_a_subgraph(item_graph, item_key)):
                    score = key_subgraph.subgraph_to_score[item_key]
                    scores_each_class[item_class] += score
            if (numpy.sum(scores_each_class) > 0):
                class_index = numpy.argmax(scores_each_class)
                all_assign_graphs.append(item_graph)
                all_assign_labels.append(class_index)
            bar.update()
        print('finish assigning pseudo labels')
        return all_assign_graphs, all_assign_labels


def one_process(rank, queue_input, global_ans, key_subgraph, number_classes, with_weight):
    print('start rank:{}'.format(rank))
    while not queue_input.empty():
        item_graph, index = queue_input.get()
        scores_each_class = numpy.zeros(number_classes)
        for item_key, item_class in key_subgraph.subgraph_to_class.items():
            if (judge_contain_a_subgraph(item_graph, item_key)):
                if (with_weight):
                    score = key_subgraph.subgraph_to_score[item_key]
                else:
                    score = 1
                scores_each_class[item_class] += score
        if (numpy.sum(scores_each_class) > 0):
            class_index = numpy.argmax(scores_each_class)
            ans_item = (index, class_index)
            global_ans.append(ans_item)
            if (len(global_ans) % 100 == 0):
                print('rank:{}, global len:{}, left:{}'.format(rank, len(global_ans), queue_input.qsize()))
    print('rank:{}, finish'.format(rank))


@PSEUDO_LABEL_ASSIGNER.register_module()
class Vote_Syn_Multi_Process():
    def __init__(self, number_classes = 3, number_of_process = 30, with_weight = True):
        super(Vote_Syn_Multi_Process, self).__init__()
        self.logger = Global_Var.logger()
        self.number_classes = number_classes
        self.number_of_process = number_of_process
        self.with_weight = with_weight

    def __call__(self, syn_dataset, key_subgraph):
        self.logger.info('assigning pseudo labels...')
        queue_input = multiprocessing.Queue()
        for index, item_graph in enumerate(syn_dataset.graphs_networkx):
            queue_input.put((item_graph, index))
            # if (index == 35):
            #     break

        global_ans = multiprocessing.Manager().list()
        all_process = []

        for i in range(self.number_of_process):
            p = multiprocessing.Process(target = one_process,
                                        args = (i, queue_input, global_ans, key_subgraph, self.number_classes,
                                                self.with_weight))
            all_process.append(p)
        for item in all_process:
            item.start()
        for item in all_process:
            item.join()

        all_assign_graphs = []
        all_assign_labels = []
        for item in global_ans:
            item_graph = syn_dataset.graphs_networkx[item[0]]
            all_assign_graphs.append(item_graph)
            all_assign_labels.append(item[1])

        print('finish assigning pseudo labels')
        return all_assign_graphs, all_assign_labels
