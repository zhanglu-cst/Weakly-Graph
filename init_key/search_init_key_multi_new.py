import json
import os
import pickle

import numpy
from mmcv import Config, ProgressBar

from dataset.moleculars import Moleculars
from init_key.analyze_one_key import LF_Analyze_for_One

cfg = Config.fromfile('config/bbbp/bbbp_attentiveFP.py')
# print(cfg.pretty_text)
# filename = cfg.data_filename
filename_save_keyword = 'bbbp_500.json'
filename_of_dataset = 'bbbp.json'

fitler_times_thr = 4
load_from_file = True
precision_thr = cfg.init_key.precision_thr
print('precision_thr:{}'.format(precision_thr))
delta_number_smile = 0
top_N_scores_for_extract = 1000
max_keep_key_each_class = 500

dir_save_init_key = r'/remote-home/zhanglu/weakly_molecular/data/init_key'
mols_dataset = Moleculars(filename = filename_of_dataset, split = 'train', token_func = 'func_group')

token_to_times = {}
for item_smile_tokens in mols_dataset.smile_to_tokens.values():
    for item_token in item_smile_tokens:
        if (item_token not in token_to_times):
            token_to_times[item_token] = 0
        token_to_times[item_token] += 1
all_frequent_tokens = []
for item_token, item_time in token_to_times.items():
    if (item_time >= fitler_times_thr):
        all_frequent_tokens.append(item_token)
print('all tokens number:{}'.format(len(token_to_times)))
print('all common number:{}'.format(len(all_frequent_tokens)))

load_filename = 'temp_dump/key_infos_each_class_{}.pkl'.format(filename_of_dataset.split('.')[0])


def get_all_key_infos():
    key_infos_each_class = None
    if (load_from_file):
        try:
            with open(load_filename, 'rb') as f:
                key_infos_each_class = pickle.load(f)
            print('load success')
        except Exception as e:
            print(str(e))
            pass
    if (key_infos_each_class is None):
        key_infos_each_class = []
        for class_index in range(mols_dataset.number_classes):
            all_key_LF_infos_cur_class = []
            analyzer = LF_Analyze_for_One(class_index = class_index, mol_dataset = mols_dataset)
            bar = ProgressBar(task_num = len(all_frequent_tokens))
            for index, item_key_token in enumerate(all_frequent_tokens):
                result = analyzer(item_key_token)
                all_key_LF_infos_cur_class.append(result)
                if (index % 100 == 0):
                    bar.update(num_tasks = 100)
            key_infos_each_class.append(all_key_LF_infos_cur_class)
        with open(load_filename, 'wb') as f:
            pickle.dump(key_infos_each_class, f)
    return key_infos_each_class


key_infos_each_class = get_all_key_infos()

all_keep_tokens_each_class = []

for class_index in range(mols_dataset.number_classes):
    print('start class:{}'.format(class_index))
    all_key_infos_cur_class = key_infos_each_class[class_index]
    all_key_scores_cur_class = []
    for item_key_info in all_key_infos_cur_class:
        f1 = item_key_info['f1']
        precision = item_key_info['precision']
        precision_thr_cur = precision_thr[class_index]
        if (precision >= precision_thr_cur):
            indicator_cur = f1
            all_key_scores_cur_class.append((indicator_cur, item_key_info))

    print('sorting')
    all_key_scores_cur_class = sorted(all_key_scores_cur_class, key = lambda x: x[0], reverse = True)

    cur_labeled_GT_true_smiles = set()
    cur_labeled_all_samples = set()
    keep_key_token_cur_class = []

    for item_key_score in all_key_scores_cur_class[:top_N_scores_for_extract]:
        item_key_info = item_key_score[1]
        print('score:{}, f1:{}, recall:{}, precision:{}, key:{},'.format(item_key_score[0],
                                                                         item_key_info['f1'],
                                                                         item_key_info['recall'],
                                                                         item_key_info['precision'],
                                                                         item_key_info['key_token']))
        item_labeled_GT_true_smiles = item_key_info['find_GT_true_samples']
        item_labeled_GT_true_smiles = set(item_labeled_GT_true_smiles)
        inter_len = len(cur_labeled_GT_true_smiles & item_labeled_GT_true_smiles)
        delta_GT_true = len(item_labeled_GT_true_smiles) - inter_len
        if (delta_GT_true >= delta_number_smile):
            cur_labeled_GT_true_smiles.update(item_labeled_GT_true_smiles)
            keep_key_token_cur_class.append(item_key_info['key_token'])
            item_labeled_all_samples = item_key_info['find_all_samples']
            cur_labeled_all_samples.update(item_labeled_all_samples)

        if (len(keep_key_token_cur_class) >= max_keep_key_each_class):
            break

    print('keep_key_token_cur_class:{}'.format(keep_key_token_cur_class))
    print('keep_key_token len:{}'.format(len(keep_key_token_cur_class)))
    # print('cur_labeled_GT_true_smiles:{}'.format(cur_labeled_GT_true_smiles))
    print('len cur_labeled_GT_true_smiles:{}'.format(len(cur_labeled_GT_true_smiles)))
    print('len cur_labeled_all_samples:{}'.format(len(cur_labeled_all_samples)))

    all_GT_labels_for_all_key_tokens_cur_class = []
    for item_smile in cur_labeled_all_samples:
        item_label = mols_dataset.smile_to_label[item_smile]
        all_GT_labels_for_all_key_tokens_cur_class.append(item_label)
    all_GT_labels_for_all_key_tokens_cur_class = numpy.array(all_GT_labels_for_all_key_tokens_cur_class)
    positive = numpy.sum(all_GT_labels_for_all_key_tokens_cur_class == class_index)
    print('GT positive:{}, correct rate:{}'.format(positive, positive / len(cur_labeled_all_samples)))

    all_keep_tokens_each_class.append(keep_key_token_cur_class)

total_len = sum(len(item) for item in all_keep_tokens_each_class)
merge = set()
for item in all_keep_tokens_each_class:
    merge.update(item)
assert total_len == len(merge)

path_saving = os.path.join(dir_save_init_key, filename_save_keyword)
with open(path_saving, 'w') as f:
    json.dump(all_keep_tokens_each_class, f)
