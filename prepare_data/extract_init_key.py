import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from classifiers import build_classifier

import argparse
from mmcv import Config
import mmcv
from compent import Global_Var, Logger_Wandb

from dataset import build_dataset
from compent.saver import Saver
from extract_keysubgraph import build_extractor
from compent.utils import set_seed_all
from compent.checkpoint import CheckPointer


def parser_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--seed')
    args = parser.parse_args()
    # cfg = Config.fromfile('config/syn/cfg_syn_baseline.py')
    cfg = Config.fromfile(args.cfg)
    print('using cfg file:{}'.format(args.cfg))
    seed = args.seed
    if (seed):
        print('using seed:{}'.format(seed))
        set_seed_all(seed = seed)
    print('cfg:{}'.format(cfg.pretty_text))
    return cfg


if __name__ == '__main__':
    cfg = parser_cfg()
    logger = Logger_Wandb(cfg)
    Global_Var.set_logger(logger)
    Global_Var.set('cfg', cfg)

    saver = Saver(save_dir = cfg.work_dir, logger = logger)
    Global_Var.set(key = 'saver', value = saver)

    print('loading model')
    checkpoint = CheckPointer(save_dir = cfg.work_dir, rank = 0)
    classifier = build_classifier(cfg = cfg.classifier)
    checkpoint.load_from_filename(model = classifier, filename = 'C_ITR_0')
    classifier = classifier.cuda()

    print('building...')
    syn_dataset = build_dataset(cfg.dataset.train)
    extractor = build_extractor(cfg.extract_key_subgraph)

    Global_Var.set('ITR', 0)
    print('start extract init key')
    extracted_keysubgraph = extractor(graphs = syn_dataset.graphs_networkx,
                                      labels = syn_dataset.labels,
                                      model = classifier)

    print(extracted_keysubgraph)
    mmcv.dump(extracted_keysubgraph, file = os.path.join(cfg.root_dir, 'key_subgraph_each_class_GTs.pkl'))
