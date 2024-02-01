import json
import os
import pickle

import numpy
from mmcv import Config

from dataset.moleculars import Moleculars
from init_key.analyze_one_key import ana_one_key_posi

cfg = Config.fromfile('config/default.py')
print(cfg.pretty_text)
filename = 'tox21_NR-ER.json'
dir_save_init_key = r'/remote-home/zhanglu/weakly_molecular/data/init_key'
mols_dataset = Moleculars(filename = filename, split = 'train', token_func = 'func_group_max4')
# updater = build_updater(cfg.update_key_subgraph)
# key_tokens_each_class = updater.get_key_tokens(smiles = mols_dataset.smiles, labels = mols_dataset.labels,
#                                                smile_to_tokens = mols_dataset.smile_to_tokens)

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

load_from_file = True


def generate_all_key_infos():
    all_key_infos = {}
    for index, item_key in enumerate(common_tokens):
        # print('cur key:{}'.format(item_key))
        result = ana_one_key_posi(item_key, mol_dataset = mols_dataset)
        all_key_infos[item_key] = result
        if (index % 1000 == 0):
            print(index)
    with open('all_key_infos', 'wb') as f:
        pickle.dump(all_key_infos, f)
    return all_key_infos


if (load_from_file):
    try:
        with open('all_key_infos', 'rb') as f:
            all_key_infos = pickle.load(f)
        print('load success')
    except:
        all_key_infos = generate_all_key_infos()
else:
    all_key_infos = generate_all_key_infos()

all_key_scores = []
for item_key_info in all_key_infos.values():
    item_key = item_key_info['key']
    positive_rate = item_key_info['positive_rate']
    if (positive_rate > 0.6):
        num_positive = item_key_info['num_positive']
        indicator_cur = positive_rate * positive_rate * num_positive
        all_key_scores.append((indicator_cur, item_key, item_key_info))

print('sorting')
all_key_scores = sorted(all_key_scores, key = lambda x: x[0], reverse = True)

cur_labeled_smiles = set()
delta_number_smile = 5
keep_key_token = []

for item_key_score in all_key_scores[:50]:
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
        keep_key_token.append(item_key_info['key'])

print(keep_key_token)
print('keep_key_token len:{}'.format(len(keep_key_token)))
print(cur_labeled_smiles)
print('len(cur_labeled_smiles):{}'.format(len(cur_labeled_smiles)))

all_labels = []
for item_smile in cur_labeled_smiles:
    item_label = mols_dataset.smile_to_label[item_smile]
    all_labels.append(item_label)
all_labels = numpy.array(all_labels)
positive = numpy.sum(all_labels == 1)
print('GT positive:{} rate:{}'.format(positive, positive / len(cur_labeled_smiles)))

path_saving = os.path.join(dir_save_init_key, filename)
with open(path_saving, 'w') as f:
    json.dump(keep_key_token, f)
