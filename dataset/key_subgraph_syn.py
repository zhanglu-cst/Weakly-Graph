import os
import pickle

from compent import Global_Var
from compent.networkx_ops import Graphs_Set, Graph_Dict
from .builder import KEY_SUBGRAPH


@KEY_SUBGRAPH.register_module()
class Key_SubGraph_Syn():
    def __init__(self, init_subgraph_filename,
                 root_dir,
                 init_subgraph_score = 1,
                 incremental = True,
                 overwrite_confict = True,
                 max_capacity_each_class = 500,
                 number_classes = 3):
        super(Key_SubGraph_Syn, self).__init__()
        self.incremental = incremental
        self.overwrite_confict = overwrite_confict
        self.max_capacity_each_class = max_capacity_each_class
        self.number_classes = number_classes

        self.logger = Global_Var.logger()

        # ----------- load init key subgraph -----------
        path_key_subgraph = os.path.join(root_dir, init_subgraph_filename)
        with open(path_key_subgraph, 'rb') as f:
            subgraph_each_classes = pickle.load(f)
        assert isinstance(subgraph_each_classes, list) and len(subgraph_each_classes) == number_classes
        assert isinstance(subgraph_each_classes[0], list)

        self.subgraph_each_classes = subgraph_each_classes
        # self.subgraph_each_classes = self.process_multi_exist(subgraph_each_classes)

        # for subgraphs_one_class in subgraph_each_classes:
        #     self.subgraph_each_classes.append(subgraphs_one_class[-number_subgraph_each_class:])

        # ----------- build ---------
        self.build_subgraph_to_class_subgraphs_list()

        self.subgraph_to_score = Graph_Dict()
        for item_subgraph in self.subgraph_to_class:
            self.subgraph_to_score[item_subgraph] = init_subgraph_score

        self.record_subgraph_info()

        # -------------- 3个变量： 1、subgraph_each_classes (subgraph_to_class) 每个类别的subgraph，
        # -------------            2. subgraph_to_score  每个子图的score
        # -------------            3. subgraph_list

    def build_subgraph_to_class_subgraphs_list(self):
        self.subgraph_to_class = {}
        self.subgraph_list = []
        for class_index, all_subgraph_cur_class in enumerate(self.subgraph_each_classes):
            for item_subgraph in all_subgraph_cur_class:
                self.subgraph_to_class[item_subgraph] = class_index
                self.subgraph_list.append(item_subgraph)
        self.subgraph_to_index = {}
        for index, item_subgraph in enumerate(self.subgraph_list):
            self.subgraph_to_index[item_subgraph] = index
        self.index_to_class = []
        for item_subgraph in self.subgraph_list:
            class_index = self.subgraph_to_class[item_subgraph]
            self.index_to_class.append(class_index)

    def process_multi_exist(self, extract_key_subgraph):
        subgraph_to_class_index = Graph_Dict()
        for class_index, subgraphs_cur_class in enumerate(self.subgraph_each_classes):
            for item_subgraph in subgraphs_cur_class:
                subgraph_to_class_index[item_subgraph] = class_index
        subgraphs_repeat = Graphs_Set()
        for class_index, subgraphs_cur_class in enumerate(extract_key_subgraph):
            for item_subgraph, item_score in subgraphs_cur_class:
                if (item_subgraph in subgraph_to_class_index):
                    origin_class = subgraph_to_class_index[item_subgraph]
                    if (class_index != origin_class):
                        subgraphs_repeat.insert_set(item_subgraph)
                else:
                    subgraph_to_class_index[item_subgraph] = class_index
        ans = []
        for class_index, subgraphs_cur_class in enumerate(extract_key_subgraph):
            self.logger.info('before remove repeat, class:{}, number new key subgraphs:{}'.format(class_index,
                                                                                                  len(subgraphs_cur_class)))
            ans_cur_class = []
            for item_subgraph, item_score in subgraphs_cur_class:
                if (subgraphs_repeat.judge_exist(item_subgraph) == False):
                    ans_cur_class.append([item_subgraph, item_score])
            self.logger.info(
                    'after remove repeat, class:{}. number new key subgraphs:{}'.format(class_index,
                                                                                        len(ans_cur_class)))
            ans.append(ans_cur_class)
        return ans

    def update_key_subgraph(self, extract_key_subgraph):
        assert isinstance(extract_key_subgraph, list) and len(extract_key_subgraph) == self.number_classes
        # subgraph_each_classes
        extract_key_subgraph = self.process_multi_exist(extract_key_subgraph)

        # ---------- subgraph_each_classes ---------------
        set_subgraph_each_classes = [Graphs_Set() for _ in range(self.number_classes)]
        for graph_set_cur_class, exist_subgraphs_cur_class in zip(set_subgraph_each_classes,
                                                                  self.subgraph_each_classes):
            for item_graph in exist_subgraphs_cur_class:
                graph_set_cur_class.insert_set(item_graph)

        for class_index, (new_subgraph_mi_one_class, graph_set_cur_class) in enumerate(zip(extract_key_subgraph,
                                                                                           set_subgraph_each_classes)):
            for (item_subgraph, item_score) in new_subgraph_mi_one_class:
                graph_set_cur_class.insert_set(item_subgraph)
            self.subgraph_each_classes[class_index] = graph_set_cur_class.graphs

        # ------------  subgraph_to_class -----------
        self.build_subgraph_to_class_subgraphs_list()

        # ------------  subgraph_to_score -----------
        for new_subgraph_mi_one_class in extract_key_subgraph:
            for item_subgraph, item_score in new_subgraph_mi_one_class:
                self.subgraph_to_score[item_subgraph] = item_score

        self.record_subgraph_info()

    def record_subgraph_info(self):
        # self.logger.info('score_each_subgraph:{}'.format(self.score_each_subgraph))
        self.logger.info('number of subgraphs:{}'.format(self.__len__()), key = 'state')
        self.logger.dict({'pseudo_quality/number_subgraphs': self.__len__()})

    def __len__(self):
        return len(self.subgraph_to_score)

    def __getitem__(self, index):
        return self.subgraph_list[index]


if __name__ == '__main__':
    key_sub = Key_SubGraph_Syn(init_subgraph_filename = 'key_subgraph_each_class.pkl',
                               root_dir = './data/syn_class3_key3')
