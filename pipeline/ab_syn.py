import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse

from mmcv import Config
from compent import Global_Var, Logger_Wandb

from dataset import build_dataset, build_key_subgraph
from trainer import build_trainer
from compent.saver import Saver
from extract_keysubgraph import build_extractor


def parser_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    args = parser.parse_args()
    # cfg = Config.fromfile('config/syn/cfg_syn_baseline.py')
    cfg = Config.fromfile(args.cfg)
    print('using cfg file:{}'.format(args.cfg))
    print('cfg:{}'.format(cfg.pretty_text))
    return cfg


if __name__ == '__main__':
    cfg = parser_cfg()
    logger = Logger_Wandb(cfg)
    Global_Var.set_logger(logger)

    saver = Saver(save_dir = cfg.work_dir, logger = logger)
    Global_Var.set(key = 'saver', value = saver)

    syn_dataset = build_dataset(cfg.dataset.train)
    key_subgraph = build_key_subgraph(cfg.key_subgraph)
    trainer = build_trainer(cfg.trainer)


    Global_Var.set('ITR', 0)
    all_assign_graphs, all_assign_labels, classifier = trainer(syn_dataset, key_subgraph)
