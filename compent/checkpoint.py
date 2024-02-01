# Copyright 2018 Dong-Hyun Lee, Kakao Brain.

""" Load a checkpoint file of pretrained transformer to a model in pytorch """

import os
import time

import torch

from compent import Global_Var
from compent.utils import make_dirs


class CheckPointer():
    def __init__(self, save_dir, rank):
        super(CheckPointer, self).__init__()
        self.logger = Global_Var.logger()
        self.rank = rank
        self.save_dir = save_dir
        make_dirs(self.save_dir)

    def load_from_filename(self, model, filename, strict = False):
        path = os.path.join(self.save_dir, filename)
        self.__load_from_file__(model, path, strict = strict)

    def __load_from_file__(self, model, path, strict):
        data = torch.load(path, map_location = 'cpu')
        model_time = data.pop('time')
        self.logger.info('Loading model from:{}, model save time:{}'.format(path, model_time))
        model_para = data.pop('model')
        keep_para = model_para

        if (hasattr(model, 'module')):
            model.module.load_state_dict(keep_para, strict = strict)
        else:
            model.load_state_dict(keep_para, strict = strict)
        self.logger.info('Load model success, other info:{}'.format(data))

    def __save_to_path__(self, model, path, other_info):
        if (self.rank != 0):
            return
        self.logger.info('save to file path:{}'.format(path))
        data = {}
        if (hasattr(model, 'module')):
            data['model'] = model.module.state_dict()
        else:
            data['model'] = model.state_dict()
        data['time'] = time.ctime()
        if (other_info):
            data.update(other_info)
        # dataset['optimizer'] = self.optimizer.state_dict()
        torch.save(data, path)

    # def save_to_best_model_file(self, model, other_info = None):
    #     self.__save_to_path__(model, self.best_model_path, other_info)
    # def load_from_best_model(self, model, strict = True):
    #     path = self.best_model_path
    #     self.__load_from_file__(model, path, strict = strict)

    def save_to_file_with_name(self, model, filename, other_info = None):
        path = os.path.join(self.save_dir, str(filename))
        self.__save_to_path__(model, path, other_info)
