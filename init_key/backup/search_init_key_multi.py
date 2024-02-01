import json
import os
import pickle

import numpy
from mmcv import Config

from dataset.moleculars import Moleculars
from init_key.analyze_one_key import ana_one_key_all_classes

cfg = Config.fromfile('config/default.py')
print(cfg.pretty_text)
filename = 'tox21_NR-ER.json'
dir_save_init_key = r'/remote-home/zhanglu/weakly_molecular/data/init_key'
mols_dataset = Moleculars(filename = filename, split = 'train', token_func = 'func_group_max4')

token_to_times = {}
for item_smile_tokens in mols_dataset.smile_to_tokens.values():
    for item_token in item_smile_tokens:
        if (item_token not in token_to_times):
            token_to_times[item_token] = 0
        token_to_times[item_token] += 1
common_tokens = []
for item_token, item_time in token_to_times.items():
    if (item_time > 2):
        common_tokens.append(item_token)
print('all tokens number:{}'.format(len(token_to_times)))
print('all common number:{}'.format(len(common_tokens)))

# all_aisigned_smiles = set()

load_from_file = False
load_filename = 'temp_dump/key_infos_each_class.pkl'


def generate_all_key_infos():
    key_infos_each_class = []
    for class_index in range(mols_dataset.number_classes):
        all_key_infos = {}
        for index, item_key in enumerate(common_tokens):
            # print('cur key:{}'.format(item_key))
            result = ana_one_key_all_classes(item_key, mol_dataset = mols_dataset, class_index = class_index)
            all_key_infos[item_key] = result
            if (index % 1000 == 0):
                print(index)
        key_infos_each_class.append(all_key_infos)
    with open(load_filename, 'wb') as f:
        pickle.dump(key_infos_each_class, f)
    return key_infos_each_class


if (load_from_file):
    try:
        with open(load_filename, 'rb') as f:
            key_infos_each_class = pickle.load(f)
        print('load success')
    except:
        key_infos_each_class = generate_all_key_infos()
else:
    key_infos_each_class = generate_all_key_infos()

all_keep_tokens_each_class = {}

for class_index in range(mols_dataset.number_classes):
    all_key_infos_cur_class = key_infos_each_class[class_index]
    all_key_scores_cur_class = []
    for item_key_info in all_key_infos_cur_class.values():
        item_key = item_key_info['key']
        positive_rate = item_key_info['positive_rate']
        if (positive_rate > 0.6):
            num_positive = item_key_info['num_positive']
            indicator_cur = positive_rate * positive_rate * num_positive
            all_key_scores_cur_class.append((indicator_cur, item_key, item_key_info))

    print('sorting')
    all_key_scores_cur_class = sorted(all_key_scores_cur_class, key = lambda x: x[0], reverse = True)

    cur_labeled_smiles = set()
    delta_number_smile = 5
    keep_key_token_cur_class = []

    for item_key_score in all_key_scores_cur_class[:50]:
        item_key_info = item_key_score[2]
        print('score:{}, positive number:{}, rate:{}, key:{},'.format(item_key_score[0],
                                                                      item_key_info['num_positive'],
                                                                      item_key_info['positive_rate'],
                                                                      item_key_info['key'], ))
        item_labeled_smiles = item_key_info['all_assigned_smile']
        item_labeled_smiles = set(item_labeled_smiles)
        inter_len = len(cur_labeled_smiles & item_labeled_smiles)
        delta = len(item_labeled_smiles) - inter_len
        if (delta >= delta_number_smile):
            cur_labeled_smiles.update(item_labeled_smiles)
            keep_key_token_cur_class.append(item_key_info['key'])

    print(keep_key_token_cur_class)
    print('keep_key_token len:{}'.format(len(keep_key_token_cur_class)))
    print(cur_labeled_smiles)
    print('len cur_labeled_smiles:{}'.format(len(cur_labeled_smiles)))

    all_labels = []
    for item_smile in cur_labeled_smiles:
        item_label = mols_dataset.smile_to_label[item_smile]
        all_labels.append(item_label)
    all_labels = numpy.array(all_labels)
    positive = numpy.sum(all_labels == class_index)
    print('GT positive:{} rate:{}'.format(positive, positive / len(cur_labeled_smiles)))

    all_keep_tokens_each_class[class_index] = keep_key_token_cur_class

path_saving = os.path.join(dir_save_init_key, filename)
with open(path_saving, 'w') as f:
    json.dump(all_keep_tokens_each_class, f)
