import mmcv
from mmcv import Config

from compent.saver import Saver
from dataset import build_dataset
from extract_keysubgraph import build_extractor

cfg = Config.fromfile('config/bbbp/bbbp_attentiveFP.py')
cfg.work_dir = '/remote-home/zhanglu/weakly_molecular/work_dir/bbbp_5_key_subgraph/run_with_5_keysubgrap_1'

saver = Saver(save_dir = cfg.work_dir, logger = None)

mol_dataset = build_dataset(cfg.dataset.train)

dict_smile_label = saver.load_from_file(filename = 'predicted_train_smiles_ITR_{}'.format(0))
smiles = dict_smile_label['smiles']
labels = dict_smile_label['labels']
extractor = build_extractor(cfg.extract_key_subgraph)
extract_key_subgraph = extractor(smiles, labels, smile_to_tokens = mol_dataset.smile_to_tokens)
print(extract_key_subgraph)
print(len(extract_key_subgraph))
print(len(extract_key_subgraph[0]))

path_GT_key_subgraph = '/remote-home/zhanglu/weakly_molecular/data/init_key/bbbp_500.json'

gt_key_subgraphs_each_class = mmcv.load(path_GT_key_subgraph)

for class_index, (predict_key_subgraps_cur_class, gt_key_subgraph_cur_class) in enumerate(
        zip(extract_key_subgraph, gt_key_subgraphs_each_class)):
    # print('gt_key_subgraph_cur_class:{}'.format(gt_key_subgraph_cur_class))
    # print('len gt:{}'.format(len(gt_key_subgraph_cur_class)))
    predict_key_subgraps = []
    for item in predict_key_subgraps_cur_class:
        predict_key_subgraps.append(item[0])
    hit_number = 0
    for item in predict_key_subgraps:
        if (item in gt_key_subgraph_cur_class):
            hit_number += 1
    # print('class index:{}'.format(class_index))
    print('hit number:{}'.format(hit_number))
    print('precision:{}'.format(hit_number / len(predict_key_subgraps_cur_class)))
    print('recall:{}'.format(hit_number / len(gt_key_subgraph_cur_class)))
    print()
