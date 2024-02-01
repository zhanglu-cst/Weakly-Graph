import json

from mmcv import Config

from dataset.moleculars import Moleculars
from extract_keysubgraph.builder import build_extractor
from pseudo_label.pseudo_mol_baseline import Assign_Label_Contain_Number_Only_Positive

cfg = Config.fromfile('config/default.py')
print(cfg.pretty_text)

mols_dataset = Moleculars(filename = 'tox21_NR-ER.json', split = 'train', token_func = 'func_group_max4')
updater = build_extractor(cfg.update_key_subgraph)
key_tokens_each_class = updater.get_key_tokens(smiles = mols_dataset.smiles, labels = mols_dataset.labels,
                                               smile_to_tokens = mols_dataset.smile_to_tokens)

print(key_tokens_each_class)
assigner_label = Assign_Label_Contain_Number_Only_Positive()
assigner_label(key_tokens_each_class = key_tokens_each_class, mol_dataset = mols_dataset)
# print(key_tokens_each_class)
# with open('key_tokens.json', 'w') as f:
#     json.dump(key_tokens_each_class, f)


