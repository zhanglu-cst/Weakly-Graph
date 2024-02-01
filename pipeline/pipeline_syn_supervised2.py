import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from mmcv import Config
from dataset import build_dataset
from trainer.trainer_st_syn import Trainer_ST_SYN
from compent import Global_Var, Logger_Wandb
from compent.saver import Saver

cfg = Config.fromfile('config/kki/supervised.py')
logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)
Global_Var.set('cfg', cfg)

saver = Saver(save_dir = cfg.work_dir, logger = logger)
syn_dataset = build_dataset(cfg.dataset.train)

Global_Var.set('ITR', 0)
trainer = Trainer_ST_SYN(cfg.trainer.cfg_trainer_finetune)
trainer.train_model(graphs = syn_dataset.graphs_dgl, labels = syn_dataset.labels)
