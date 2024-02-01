# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import time

import wandb
from mmcv import get_logger

from compent.utils import make_dirs
from compent.visdom_show import My_Visdom


# from torch.utils.tensorboard import SummaryWriter


class Logger_Wandb():
    def __init__(self, cfg):
        entity = cfg.logger.entity
        project = cfg.logger.project
        name = cfg.logger.name
        self.enable_wandb = cfg.logger.enable_wandb
        if (self.enable_wandb):
            wandb.init(project = project, entity = entity, name = name, config = cfg)
            self.wandb_text = {}
        make_dirs(cfg.work_dir)
        log_path = os.path.join(cfg.work_dir, name + '.log')
        self.logger = get_logger(name = name, log_file = log_path, file_mode = 'w')

    def dict(self, key_value_pair):
        self.logger.info(str(key_value_pair))
        if (self.enable_wandb):
            wandb.log(key_value_pair)

    def info(self, text, key = None):
        text = str(text)
        if (key is None):
            self.logger.info(text)
        else:
            self.logger.info('key:{}, text:{}'.format(key, text))
        if (key is not None and self.enable_wandb):
            if (key not in self.wandb_text):
                self.wandb_text[key] = []
            self.wandb_text[key].append(text)
            str_text = ' <br/> '.join(self.wandb_text[key])
            str_text = wandb.Html(str_text)
            pair = {key: str_text}
            wandb.log(pair)


class Logger():
    def __init__(self, name, save_dir, distributed_rank, visdom_port = 8888, filename = None, only_main_rank = False):
        self.rank = distributed_rank
        self.only_main_rank = only_main_rank

        if (filename is None):
            filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(stream = sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            make_dirs(save_dir)
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        self.logger = logger
        if (self.rank == 0):
            self.visdom = My_Visdom(env_name = name, port = visdom_port)
            self.visdom.close_all_curves()
        self.inner_values = {}

    def set_value(self, key, value):
        self.inner_values[key] = value

    def get_value(self, key):
        return self.inner_values[key]

    def info(self, info, show_one = False):
        info = 'rank:{}, {}'.format(self.rank, info)
        if (self.only_main_rank and self.rank == 0):
            self.logger.info(info)
        elif (self.only_main_rank == False and show_one == False):
            self.logger.info(info)
        elif (self.only_main_rank == False and show_one == True and self.rank == 0):
            self.logger.info(info)

    def plot_record(self, value, win_name, X_value = None):
        if (self.rank == 0):
            self.visdom.plot_record(value, win_name, X_value)

    def visdom_text(self, text, win_name, append = True):
        text = str(text) + '\n\n'
        if (self.rank == 0):
            self.visdom.text(text, win_name, append = append)

    def visdom_table(self, table, win_name):
        if (self.rank == 0):
            self.visdom.table(table, win_name)

    def clear_record(self, win_name):
        if (self.rank == 0):
            self.visdom.clear_record(win_name)

    def close_all_curves(self):
        if (self.rank == 0):
            self.visdom.close_all_curves()


#
# class Logger_Board():
#     def __init__(self, name, save_dir, distributed_rank, filename = None, only_main_rank = False):
#         self.rank = distributed_rank
#         self.only_main_rank = only_main_rank
#
#         if (filename is None):
#             filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'
#
#         logger = logging.getLogger(name)
#         logger.setLevel(logging.DEBUG)
#
#         ch = logging.StreamHandler(stream = sys.stdout)
#         ch.setLevel(logging.DEBUG)
#         formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#         ch.setFormatter(formatter)
#         logger.addHandler(ch)
#
#         if save_dir:
#             make_dirs(save_dir)
#             fh = logging.FileHandler(os.path.join(save_dir, filename))
#             fh.setLevel(logging.DEBUG)
#             fh.setFormatter(formatter)
#             logger.addHandler(fh)
#
#         self.logger = logger
#         if (self.rank == 0):
#             log_dir = os.path.join(save_dir, name)
#             make_dirs(log_dir)
#             self.writer = SummaryWriter(log_dir = log_dir)
#             self.X_record = {}
#
#     def info(self, info, show_one = False):
#         info = 'rank:{}, {}'.format(self.rank, info)
#         if (self.only_main_rank and self.rank == 0):
#             self.logger.info(info)
#         elif (self.only_main_rank == False and show_one == False):
#             self.logger.info(info)
#         elif (self.only_main_rank == False and show_one == True and self.rank == 0):
#             self.logger.info(info)
#
#     def plot_record(self, value, win_name, X_value = None):
#         if (self.rank == 0):
#             if (X_value):
#                 self.writer.add_scalar(tag = win_name, scalar_value = value, global_step = X_value)
#             else:
#                 if (win_name not in self.X_record):
#                     self.X_record[win_name] = 0
#                     cur_X = 0
#                 else:
#                     cur_X = self.X_record[win_name]
#                     self.X_record[win_name] = cur_X + 1
#                 self.writer.add_scalar(tag = win_name, scalar_value = value, global_step = cur_X)
#
#     def visdom_text(self, text, win_name, append = True):
#         text = str(text) + '\n\n'
#         if (self.rank == 0):
#             self.writer.add_text(tag = win_name, text_string = text)
#
#     # def clear_record(self, win_name):
#     #     if (self.rank == 0):
#
#     # def close_all_curves(self):
#     #     if (self.visdom_only_main and self.rank == 0):
#     #         self.visdom.close_all_curves()


def setup_logger(name, save_dir, distributed_rank, filename = None, only_main_rank = True):
    if (filename is None):
        filename = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don'origin_str log results for the non-master process
    if only_main_rank and distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream = sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        make_dirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
