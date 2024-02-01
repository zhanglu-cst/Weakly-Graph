import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from mmcv import Config

from compent import Global_Var, Logger_Wandb
from dataset import build_dataset, build_key_subgraph
from pseudo_label import build_pseudo_label_assigner
from extract_keysubgraph import build_extractor
from compent.saver import Saver

cfg = Config.fromfile('config/bbbp/bbbp_attentiveFP.py')
cfg.logger.enable_wandb = False
cfg.logger.project = 'bbbp_first_ITR'
cfg.logger.name = 'test_run_2'

logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)

saver = Saver(save_dir = './saved_data', logger = logger)

mol_dataset = build_dataset(cfg.dataset.train)
key_subgraph = build_key_subgraph(cfg.key_subgraph)
pseudo_label_assigner = build_pseudo_label_assigner(cfg.pseudo_label)
updater = build_extractor(cfg.extract_key_subgraph)

total_number_itr = cfg.total_number_itr

smile_label_dict = saver.load_from_file(filename = 'predicted_train_smiles_ITR_{}'.format(0))
labels = smile_label_dict['labels']

extract_key_subgraph = updater(smile_label_dict['smiles'], smile_label_dict['labels'],
                               smile_to_tokens = mol_dataset.smile_to_tokens)
saver.save_to_file(obj = extract_key_subgraph,
                   filename = 'extract_key_subgraph_ITR_{}'.format(0))
key_subgraph.update_key_subgraph(extract_key_subgraph = extract_key_subgraph)
key_subgraph.show_key_subgraph_info(mol_dataset = mol_dataset)
