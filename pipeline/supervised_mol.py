import argparse
import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['https_proxy'] = 'http://10.162.159.248:7890'
os.environ['http_proxy'] = 'http://10.162.159.248:7890'
os.environ['all_proxy'] = 'socks5://10.162.159.248:7890'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from compent.utils import set_seed_all

from mmcv import Config
from dataset import build_dataset
from trainer.trainer_st_mol import Trainer_ST_MOL
from compent import Global_Var, Logger_Wandb
from compent.saver import Saver


def parser_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--seed')
    args = parser.parse_args()
    cfg = Config.fromfile(args.cfg)
    print('using cfg file:{}'.format(args.cfg))
    seed = args.seed
    if (seed):
        print('using seed:{}'.format(seed))
        set_seed_all(seed = seed)
    print('cfg:{}'.format(cfg.pretty_text))
    return cfg


# cfg = parser_cfg()
cfg = Config.fromfile('config/sider_Vascular/supervised.py')
logger = Logger_Wandb(cfg)
Global_Var.set_logger(logger)
Global_Var.set('cfg', cfg)

saver = Saver(save_dir = cfg.work_dir, logger = logger)
mol_dataset = build_dataset(cfg.dataset.train)

Global_Var.set('ITR', 0)
trainer = Trainer_ST_MOL(cfg.trainer.cfg_trainer_finetune)
trainer.train_model(smiles = mol_dataset.smiles, labels = mol_dataset.labels)
