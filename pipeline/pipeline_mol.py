import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from mmcv import Config

from classifiers.trainer.trainer_st_mol import Trainer_ST_MOL
from compent import Global_Var, Logger_Wandb
from dataset import build_dataset, build_key_subgraph
from pseudo_label import build_pseudo_label_assigner
from extract_keysubgraph import build_extractor
from compent.saver import Saver

cfg = Config.fromfile('config/bbbp/bbbp_attentiveFP.py')
cfg.logger.project = 'bbbp_5_key_subgraph'
cfg.logger.name = 'run_with_5_keysubgrap_1'
cfg.work_dir = os.path.join('./work_dir', cfg.logger.project, cfg.logger.name)

logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)

saver = Saver(save_dir = cfg.work_dir, logger = logger)

mol_dataset = build_dataset(cfg.dataset.train)
key_subgraph = build_key_subgraph(cfg.key_subgraph)
pseudo_label_assigner = build_pseudo_label_assigner(cfg.pseudo_label)
extractor = build_extractor(cfg.extract_key_subgraph)

total_number_itr = cfg.total_number_itr

for ITR in range(total_number_itr):
    Global_Var.set('ITR', ITR)
    key_subgraph.show_key_subgraph_info(mol_dataset)
    pseudo_labels_dict = pseudo_label_assigner(key_subgraph, mol_dataset)
    trainer_classifier = Trainer_ST_MOL(cfg = cfg)
    smiles, labels = trainer_classifier.train_model(smiles = pseudo_labels_dict['all_asigned_smiles'],
                                                    labels = pseudo_labels_dict['all_asigned_labels'])
    saver.save_to_file(obj = {'smiles': smiles, 'labels': labels},
                       filename = 'predicted_train_smiles_ITR_{}'.format(ITR))
    extract_key_subgraph = extractor(smiles, labels, smile_to_tokens = mol_dataset.smile_to_tokens)
    saver.save_to_file(obj = extract_key_subgraph,
                       filename = 'extract_key_subgraph_ITR_{}'.format(ITR))
    key_subgraph.update_key_subgraph(extract_key_subgraph = extract_key_subgraph)
