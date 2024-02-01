import copy
import json
import multiprocessing
import os

import mmcv
import numpy

from compent import Global_Var
from compent.molecular_ops import build_dgl_graph_from_smile
from compent.utils import add_pre_key
from extract_keysubgraph.enumrate_subgraphs_mol import Enumrate_Subgraphs_Mol
from pseudo_label.metric import cal_metric_multi_class
from .builder import DATASET


def process_enumrate(rank, queue_input, global_list, enumrate_max_frag_number):
    print('rank:{}, start'.format(rank))
    func_enumrate = Enumrate_Subgraphs_Mol(max_frag_number = enumrate_max_frag_number)
    while not queue_input.empty():
        item_smile = queue_input.get()
        subgraphs_cur_smiles, frags_cur_smiles, frags_cur_mols = func_enumrate(item_smile, return_fragment_list = True)
        item_res = (item_smile, subgraphs_cur_smiles, frags_cur_smiles, frags_cur_mols)
        global_list.append(item_res)
        if (len(global_list) % 100 == 0):
            queue_size = queue_input.qsize()
            print('rank:{}, finish:{},queue_size:{}'.format(rank, len(global_list), queue_size))
    print('rank:{} finish'.format(rank))


def generate_dump_files(smiles, multi_process_number, enumrate_max_frag_number):
    print('init len:{}'.format(len(smiles)))
    smile_to_subgraphs = {}
    smile_to_fragments = {}
    all_frags_smiles = []
    all_frags_mols = []
    queue_input = multiprocessing.Queue()
    for item_smile in smiles:
        queue_input.put(item_smile)
    print('start multi process enumrate...')
    all_process = []
    global_list = multiprocessing.Manager().list()
    for rank in range(multi_process_number):
        p = multiprocessing.Process(target = process_enumrate,
                                    args = (rank, queue_input, global_list, enumrate_max_frag_number))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()
    list_global_ans = list(global_list)
    print('after process, len:{}'.format(len(list_global_ans)))
    for item in list_global_ans:
        item_smile, subgraphs_cur_smiles, frags_cur_smiles, frags_cur_mols = item
        smile_to_subgraphs[item_smile] = set(subgraphs_cur_smiles)
        all_frags_smiles += frags_cur_smiles
        smile_to_fragments[item_smile] = frags_cur_smiles
        all_frags_mols += frags_cur_mols
    del all_process
    print()
    print('generate success')
    print('generate fragments number:{}'.format(len(all_frags_smiles)))

    return smile_to_subgraphs, all_frags_smiles, all_frags_mols, smile_to_fragments


def process_smile_to_dgl(rank, queue_input, global_list):
    print('rank:{}, start'.format(rank))
    while not queue_input.empty():
        item_smile = queue_input.get()
        dgl_subgraph = build_dgl_graph_from_smile(item_smile)
        if (dgl_subgraph is not None):
            num_edge_feature = dgl_subgraph.edata['x']
            if (len(num_edge_feature) == 0):
                dgl_subgraph = None
        global_list.append([item_smile, dgl_subgraph])

    print('rank:{} finish'.format(rank))


def generate_subgraph_smiles_to_dgl(all_subgraph_smiles, multi_process_number):
    # all_subgraph_smiles = set(all_subgraph_smiles)
    queue_input = multiprocessing.Queue()
    for item_smile in all_subgraph_smiles:
        queue_input.put(item_smile)
    print('generate_subgraph_smiles_to_dgl:')

    all_process = []
    global_list = multiprocessing.Manager().list()
    for rank in range(multi_process_number):
        p = multiprocessing.Process(target = process_smile_to_dgl,
                                    args = (rank, queue_input, global_list))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()
    list_global_ans = list(global_list)
    print('after process, len:{}'.format(len(list_global_ans)))
    ans = {}
    for item in list_global_ans:
        ans[item[0]] = item[1]
    return ans


@DATASET.register_module()
class Moleculars():
    def __init__(self, root_dir, split, multi_process_number = 30, enumrate_max_frag_number = 6):
        super(Moleculars, self).__init__()
        self.split = split
        filename = '{}.json'.format(split)
        path_dataset = os.path.join(root_dir, filename)
        self.logger = Global_Var.logger()
        self.logger.info('loading dataset from:{}'.format(path_dataset))
        with open(path_dataset, 'r') as f:
            cur_split_data = json.load(f)
        self.smiles = []
        self.labels = []
        self.graphs = []
        self.smile_to_graph = {}
        self.multi_process_number = multi_process_number
        # ---------- filter illegal smiles ------
        for item in cur_split_data:
            smile_item, label_item = item
            graph_item = build_dgl_graph_from_smile(smile_item)
            if (graph_item is not None):
                self.smiles.append(smile_item)
                self.labels.append(label_item)
                self.graphs.append(graph_item)
                self.smile_to_graph[smile_item] = graph_item

        filtered = len(cur_split_data) - len(self.smiles)
        print('split:{},left number of smiles:{}, filtered:{}'.format(self.split, len(self.smiles), filtered))

        if (split == 'train'):
            filename_subgraphs = 'subgraphs.json'
            filename_frags = 'all_frags_smiles.json'
            filename_smile_to_frags = 'smiles_to_frags.json'
            filename_subgraph_smile_to_dgl_graphs = 'subgraph_smile_to_dgl_graphs.pkl'
            filename_all_frags_mols = 'all_frags_mols.pkl'
            path_subgraphs = os.path.join(root_dir, filename_subgraphs)
            path_fragments = os.path.join(root_dir, filename_frags)
            path_smile_to_frags = os.path.join(root_dir, filename_smile_to_frags)
            # path_subgraph_smile_to_dgl_graphs = os.path.join(root_dir, filename_subgraph_smile_to_dgl_graphs)
            path_all_frags_mols = os.path.join(root_dir, filename_all_frags_mols)
            if (os.path.exists(path_subgraphs) and os.path.exists(path_smile_to_frags) and os.path.exists(
                    path_fragments)):
                self.logger.info('loading dump files...')
                smile_to_subgraphs = mmcv.load(path_subgraphs)
                smile_to_fragments = mmcv.load(path_smile_to_frags)
                all_frags_smiles = mmcv.load(path_fragments)
                all_frags_mols = mmcv.load(path_all_frags_mols)
                # subgraph_smile_to_dgl_graphs = mmcv.load(path_subgraph_smile_to_dgl_graphs)
                self.logger.info('load smile to subgraphs from file:{}'.format(path_subgraphs))
            else:
                self.logger.info('load failed, generate smile_to_subgraphs')
                smile_to_subgraphs, all_frags_smiles, all_frags_mols, smile_to_fragments = generate_dump_files(
                        smiles = copy.deepcopy(self.smiles),
                        multi_process_number = self.multi_process_number,
                        enumrate_max_frag_number = enumrate_max_frag_number)

                # mmcv.dump(obj = subgraph_smile_to_dgl_graphs, file = path_subgraph_smile_to_dgl_graphs)
                mmcv.dump(obj = smile_to_subgraphs, file = path_subgraphs)
                mmcv.dump(obj = all_frags_smiles, file = path_fragments)
                mmcv.dump(obj = smile_to_fragments, file = path_smile_to_frags)
                mmcv.dump(obj = all_frags_mols, file = path_all_frags_mols)
                self.logger.info('dump to file sucess')

            print('start building subgraph_smile_to_dgl_graphs')
            all_subgraph_smiles = set()
            for _, subgraph_smiles in smile_to_subgraphs.items():
                for item_subgraph_smile in subgraph_smiles:
                    all_subgraph_smiles.add(item_subgraph_smile)
            subgraph_smile_to_dgl_graphs = generate_subgraph_smiles_to_dgl(all_subgraph_smiles = all_subgraph_smiles,
                                                                           multi_process_number = self.multi_process_number)
            print()

            self.subgraph_smile_to_dgl_graphs = subgraph_smile_to_dgl_graphs
            self.smile_to_subgraphs = smile_to_subgraphs
            self.all_frags_smiles = all_frags_smiles
            self.smile_to_fragments = smile_to_fragments
            self.all_frags_mols = all_frags_mols

        self.smile_to_label = {}
        for item_smile, item_label in zip(self.smiles, self.labels):
            self.smile_to_label[item_smile] = item_label

        self.number_classes = numpy.max(self.labels) + 1
        self.label_count_each_class = self.count_label_infos()

    def __getitem__(self, item):
        return self.smiles[item]

    def __len__(self):
        return len(self.smiles)

    def count_label_infos(self):
        label_count = {}
        for item in self.labels:
            if (item not in label_count):
                label_count[item] = 0
            label_count[item] += 1
        self.logger.info('split:{},label count:{}'.format(self.split, label_count))
        return label_count

    def cal_pseudo_quality(self, pseudo_graphs, pseudo_labels, pre_key = 'pseudo_quality'):
        hit_rate = len(pseudo_graphs) / len(self.labels)
        hit_number = len(pseudo_graphs)
        gt_labels = []
        for item_graph in pseudo_graphs:
            item_gt = self.smile_to_label[item_graph]
            gt_labels.append(item_gt)
        metric = cal_metric_multi_class(all_pred_class = pseudo_labels, all_gt_labels = gt_labels)
        log_dict = {'hit_rate': hit_rate, 'hit_number': hit_number}
        log_dict.update(metric)
        log_dict = add_pre_key(log_dict, pre_key = pre_key)
        self.logger.dict(log_dict)
        return log_dict

# for item_smile in self.smiles:
#     subgraphs_cur_smiles, frags_cur_smiles = func_enumrate(item_smile, return_fragment_list = True)
#     smile_to_subgraphs[item_smile] = set(subgraphs_cur_smiles)
#     all_frags_smiles += frags_cur_smiles
#     smile_to_fragments[item_smile] = frags_cur_smiles
