import json
import os

from compent import Global_Var
from .builder import KEY_SUBGRAPH


@KEY_SUBGRAPH.register_module()
class Key_SubGraph_Mol():
    def __init__(self, init_subgraph_filename,
                 root_dir,
                 init_subgraph_score = 1,
                 incremental = True,
                 overwrite_confict = True,
                 max_capacity_each_class = 500,
                 number_classes = 2,
                 ):
        super(Key_SubGraph_Mol, self).__init__()

        self.incremental = incremental
        self.overwrite_confict = overwrite_confict
        self.max_capacity_each_class = max_capacity_each_class
        self.number_classes = number_classes

        self.logger = Global_Var.logger()

        # ----------- load init key subgraph -----------
        path_key_subgraph = os.path.join(root_dir, init_subgraph_filename)
        with open(path_key_subgraph, 'r') as f:
            self.subgraph_each_classes = json.load(f)
        assert isinstance(self.subgraph_each_classes, list) and len(self.subgraph_each_classes) == number_classes
        assert isinstance(self.subgraph_each_classes[0], list)

        # -------- init score ---------
        self.subgraph_to_score = {}
        for keys_cur_class in self.subgraph_each_classes:
            for item_key in keys_cur_class:
                self.subgraph_to_score[item_key] = init_subgraph_score

        self.build_subgraph_to_class_subgraphs_list()
        self.record_subgraph_info()

    def build_subgraph_to_class_subgraphs_list(self):
        self.subgraph_to_class = {}
        self.subgraph_list = []
        self.index_to_class = []
        for class_index, keys_cur_class in enumerate(self.subgraph_each_classes):
            for item_key in keys_cur_class:
                self.subgraph_to_class[item_key] = class_index
                self.subgraph_list.append(item_key)
                self.index_to_class.append(class_index)

        self.subgraph_to_index = {}
        for index, item_subgraph in enumerate(self.subgraph_list):
            self.subgraph_to_index[item_subgraph] = index

    def update_key_subgraph(self, extract_key_subgraph):
        for class_index, subgraphs_cur_class in enumerate(extract_key_subgraph):
            for item_subgraph, item_score in subgraphs_cur_class:
                if (item_subgraph not in self.subgraph_to_class):
                    self.subgraph_each_classes[class_index].append(item_subgraph)
                    self.subgraph_to_score[item_subgraph] = item_score
                else:
                    origin_score = self.subgraph_to_score[item_subgraph]
                    if (origin_score < item_score):
                        origin_class = self.subgraph_to_class[item_subgraph]
                        self.subgraph_each_classes[origin_class].remove(item_subgraph)
                        self.subgraph_each_classes[class_index].append(item_subgraph)
                        self.subgraph_to_score[item_subgraph] = item_score
        self.build_subgraph_to_class_subgraphs_list()
        self.record_subgraph_info()

    def record_subgraph_info(self):
        # self.logger.info('score_each_subgraph:{}'.format(self.score_each_subgraph))
        self.logger.info('number of subgraphs:{}'.format(self.__len__()), key = 'state')
        self.logger.dict({'pseudo_quality/number_subgraphs': self.__len__()})

    def __len__(self):
        return len(self.subgraph_list)

    def __getitem__(self, index):
        return self.subgraph_list[index]
