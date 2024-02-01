import json
import os
import time

from mmcv import Config

from dataset.moleculars import Moleculars
from extract_keysubgraph import build_token_mol

cfg = Config.fromfile('config/default.py')
dataset_filename = 'clintox_CT_TOX.json'  # clintox_CT_TOX.json   tox21_NR-ER.json
save_dir = r'/apdcephfs/share_1364275/xluzhang/weakly_graph/data/token_results/func_group'

token_func = build_token_mol(cfg = cfg.token)

mols = Moleculars(filename = dataset_filename, split = 'train')

token_results_dict = {}
for item_smile in mols.smiles:
    start_time = time.time()
    token_res = token_func.do_token(item_smile)
    end_time = time.time()
    # print(token_res)
    token_results_dict[item_smile] = token_res
    print('time:{}'.format(end_time - start_time))

print(token_results_dict)

path_saving = os.path.join(save_dir, dataset_filename)
with open(path_saving, 'w') as f:
    json.dump(token_results_dict, f)
