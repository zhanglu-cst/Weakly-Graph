import os
import sys

import mmcv

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from mmcv import Config
from compent import Global_Var, Logger_Wandb

from dataset import build_dataset, build_key_subgraph
from pseudo_label import build_pseudo_label_assigner
from compent.saver import Saver
from extract_keysubgraph import build_extractor

cfg = Config.fromfile('config/syn/v4_16.py')
cfg.logger.project = 'syn_version5'
cfg.logger.name = 'extract_label_model_debug'
cfg.work_dir = os.path.join('./work_dir', cfg.logger.project, cfg.logger.name)

logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)

saver = Saver(save_dir = cfg.work_dir, logger = logger)
syn_dataset = build_dataset(cfg.dataset.train)
key_subgraph = build_key_subgraph(cfg.key_subgraph)
pseudo_label_assigner = build_pseudo_label_assigner(cfg.pseudo_label)
extractor = build_extractor(cfg.extract_key_subgraph)

Global_Var.set('ITR', 0)

ITR = 0

# dict_graphs_labels = mmcv.load(
#         '/remote-home/zhanglu/weakly_molecular/work_dir/syn_version5/count_baseline/pred_graphs_labels_ITR_{}.pkl'.format(
#                 ITR))
# graphs_all = dict_graphs_labels['graphs_all']
# pred_labels = dict_graphs_labels['pred_labels']
# model = build_classifier(cfg.classifier).cuda()
# checkpointer = CheckPointer(save_dir = '/remote-home/zhanglu/weakly_molecular/work_dir/syn_version5/count_baseline/',
#                             rank = 0)
# checkpointer.load_from_filename(model = model, filename = 'C_ITR_0')

# extracted_keysubgraph = extractor(graphs_networkx = graphs_all,
#                                   labels = pred_labels,
#                                   model = model)
# mmcv.dump(obj = extracted_keysubgraph, file = 'extracted_keysubgraph.pkl')
extracted_keysubgraph = mmcv.load('extracted_keysubgraph.pkl')
keep_len = 100
keeped_keysubgraph = []
for item_class in extracted_keysubgraph:
    keeped_keysubgraph.append(item_class[:keep_len])

key_subgraph.update_key_subgraph(keeped_keysubgraph)
all_assign_graphs, all_assign_labels = pseudo_label_assigner(syn_dataset, key_subgraph)
syn_dataset.cal_pseudo_quality(pseudo_graphs = all_assign_graphs, pseudo_labels = all_assign_labels)
