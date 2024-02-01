import json

from dataset.moleculars import Moleculars
from extract_keysubgraph import Updater_TF_IDF_Mol
from pseudo_label.pseudo_mol_baseline import Assign_Label_Contain_Number_Only_Positive

choose_IDF = [3, 4, 5, 6]
choose_keep = [20]
choose_score_thr = [0]

mols_dataset = Moleculars(filename = 'tox21_NR-ER.json', split = 'train', token_func = 'func_group')

all_params_results = []
best_f1 = 0
best_param = None
for cur_IDF in choose_IDF:
    for cur_keep in choose_keep:
        for cur_score_thr in choose_score_thr:
            updater = Updater_TF_IDF_Mol(IDF_power = cur_IDF, number_classes = 2, keep_tokens = cur_keep)
            key_tokens_each_class = updater.get_key_tokens(smiles = mols_dataset.smiles, labels = mols_dataset.labels,
                                                           smile_to_tokens = mols_dataset.smile_to_tokens)
            assigner_label = Assign_Label_Contain_Number_Only_Positive()
            f1 = assigner_label(key_tokens_each_class = key_tokens_each_class, mol_dataset = mols_dataset)
            cur_param = {'IDF': cur_IDF, 'keep': cur_keep, 'score_thr': cur_score_thr}
            print('cur param:{}, f1:{}'.format(cur_param, f1))
            if (f1 > best_f1):
                best_f1 = f1
                best_param = cur_param
            print('best f1:{}, param:{}'.format(best_f1, best_param))
            all_params_results.append([cur_param, f1])

with open('seach_result.json', 'w') as f:
    json.dump(all_params_results, f)
# print(key_tokens_each_class)
# with open('key_tokens.json', 'w') as f:
#     json.dump(key_tokens_each_class, f)
