import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import mmcv

sys.path.append('/remote-home/zhanglu/weakly_molecular')

os.environ['https_proxy'] = 'http://10.162.159.248:7890'
os.environ['http_proxy'] = 'http://10.162.159.248:7890'
os.environ['all_proxy'] = 'socks5://10.162.159.248:7890'

import argparse

from mmcv import Config
from compent import Global_Var, Logger_Wandb

from dataset import build_dataset, build_key_subgraph
from trainer import build_trainer
from compent.saver import Saver
from extract_keysubgraph import build_extractor
from compent.utils import set_seed_all


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


def get_start_itr(cfg):
    resume = cfg.resume
    if (resume == False):
        return None, 0
    else:
        work_dir = cfg.work_dir
        all_files = os.listdir(work_dir)
        all_key_subgraph_files = []
        for item_file in all_files:
            if (item_file.startswith('key_subgraph')):
                all_key_subgraph_files.append(item_file)
        if (len(all_key_subgraph_files) == 0):
            return None, 0
        all_pairs = []
        for item_key_file in all_key_subgraph_files:
            itr_num = item_key_file.split('.')[0].split('_')[-1]
            itr_num = int(itr_num)
            all_pairs.append((item_key_file, itr_num))
        all_pairs = sorted(all_pairs, key = lambda x: x[1], reverse = True)
        path = os.path.join(work_dir, all_pairs[0][0])
        return path, all_pairs[0][1]


if __name__ == '__main__':
    cfg = parser_cfg()
    logger = Logger_Wandb(cfg)
    Global_Var.set_logger(logger)
    Global_Var.set('cfg', cfg)

    saver = Saver(save_dir = cfg.work_dir, logger = logger)
    Global_Var.set(key = 'saver', value = saver)

    start_key_filename, start_itr = get_start_itr(cfg)
    if (start_key_filename is None):
        key_subgraph = build_key_subgraph(cfg.key_subgraph)
    else:
        key_subgraph = mmcv.load(start_key_filename)
        logger.info('resume from:{}'.format(start_key_filename), key = 'state')

    dataset = build_dataset(cfg.dataset.train)
    Global_Var.set('dataset', dataset)

    trainer = build_trainer(cfg.trainer)
    extractor = build_extractor(cfg.extract_key_subgraph)

    total_number_itr = cfg.total_number_itr

    for ITR in range(start_itr, total_number_itr):
        Global_Var.set('ITR', ITR)
        all_assign_graphs, all_assign_labels, classifier = trainer(dataset, key_subgraph)
        saver.save_to_file(obj = (all_assign_graphs, all_assign_labels, classifier),
                           filename = 'classification_res_{}.pkl'.format(ITR))

        extracted_keysubgraph = extractor(graphs = all_assign_graphs,
                                          labels = all_assign_labels,
                                          model = classifier)

        key_subgraph.update_key_subgraph(extracted_keysubgraph)
        saver.save_to_file(obj = key_subgraph, filename = 'key_subgraph_{}.pkl'.format(ITR + 1))
