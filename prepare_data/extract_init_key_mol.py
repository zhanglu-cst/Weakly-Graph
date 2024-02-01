import os
import sys

sys.path.append('/remote-home/zhanglu/weakly_molecular')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['https_proxy'] = 'http://10.162.159.248:7890'
os.environ['http_proxy'] = 'http://10.162.159.248:7890'
os.environ['all_proxy'] = 'socks5://10.162.159.248:7890'


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
    cfg = Config.fromfile(args.cfg)
    print('using cfg file:{}'.format(args.cfg))
    seed = args.seed
    if (seed):
        print('using seed:{}'.format(seed))
        set_seed_all(seed = seed)
    print('cfg:{}'.format(cfg.pretty_text))
    return cfg


if __name__ == '__main__':
    # cfg = parser_cfg()
    cfg = Config.fromfile('config/sider_Vascular/supervised.py')
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
    mol_dataset = build_dataset(cfg.dataset.train)
    Global_Var.set(key = 'dataset', value = mol_dataset)
    # print('get dataset:')
    # print(Global_Var.get('dataset'))
    # print(Global_Var.GLOBAL_VARS_DICT)
    extractor = build_extractor(cfg.extract_key_subgraph)

    Global_Var.set('ITR', 0)
    print('start extract init key')
    extracted_keysubgraph = extractor(graphs = mol_dataset.smiles,
                                      labels = mol_dataset.labels,
                                      model = classifier)

    print(extracted_keysubgraph)
    mmcv.dump(extracted_keysubgraph, file = os.path.join(cfg.root_dir, 'key_subgraph_each_class_GTs.json'))
