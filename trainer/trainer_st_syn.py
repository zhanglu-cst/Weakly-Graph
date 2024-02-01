import time

import networkx
import torch
from torch import optim

from classifiers.build import build_dataloader, build_torch_dataset_for_DGL, build_classifier
from classifiers.evaler.eval_dgl import Evaler_DGL
from compent import Global_Var
from compent.checkpoint import CheckPointer
from compent.metric_logger import MetricLogger
from compent.networkx_ops import batch_networkx_to_DGL
from compent.utils import move_to_device


class Trainer_ST_SYN():
    def __init__(self, cfg):
        self.cfg = cfg

        self.logger = Global_Var.logger()
        self.evaler = Evaler_DGL(cfg)
        self.checkpoint = CheckPointer(cfg.work_dir, rank = 0)
        self.model_selection_by = 'acc'
        self.global_best = 0

    def build_optimizer(self, model):
        name = self.cfg.optimizer_classifier.name
        optimizer = getattr(optim, name)(model.parameters(), lr = self.cfg.optimizer_classifier.lr)
        return optimizer

    def build_loss_func(self):
        return torch.nn.CrossEntropyLoss()

    def get_itr_number(self):
        stop_list = self.cfg.stop_itr
        ITR = Global_Var.get('ITR')
        if (ITR < len(stop_list)):
            cur_stop_itr = stop_list[ITR]
        else:
            cur_stop_itr = stop_list[-1]
        return cur_stop_itr

    def build_train_loader(self, graphs, labels):
        if (isinstance(graphs[0], networkx.Graph)):
            graphs = batch_networkx_to_DGL(batch_graphs_networkX = graphs)
        training_dataset = build_torch_dataset_for_DGL(self.cfg, graphs_dgl = graphs, labels = labels,
                                                       for_train = True)
        train_dataloader = build_dataloader(self.cfg, dataset = training_dataset, for_train = True)
        return train_dataloader

    def log_training_info_to_wandb(self, info_dict):
        wandb_dict = {}
        pre_key = 'train'
        keep_key = ['epoch', 'itr', 'loss', 'eta_min']
        for item_key, item_value in info_dict.items():
            if (item_key in keep_key):
                new_key = pre_key + '/' + item_key
                wandb_dict[new_key] = item_value
        self.logger.dict(wandb_dict)

    def train_model(self, graphs, labels):
        ITR = Global_Var.get('ITR')
        self.ITR = ITR
        self.logger.info('starting training ITR:{}'.format(ITR), key = 'state')
        itr_self_training = 0
        last_best_val = 0
        while itr_self_training < self.cfg.max_ST_itr:
            self.logger.info('starting itr_self_training:{}'.format(itr_self_training), key = 'state')
            self.logger.dict({'global/ST_itr': itr_self_training, 'global/ITR': ITR})
            graphs, labels, val_acc = self.__do_train__(graphs, labels, itr_self_training)
            if (val_acc > last_best_val):
                last_best_val = val_acc
                itr_self_training += 1
            else:
                self.logger.info('finish self training, ITR:{}, last itr st index:{}'.format(ITR, itr_self_training),
                                 key = 'state')
                break

        return graphs, labels

    def __do_train__(self, graphs, labels, itr_ST):
        assert len(graphs) == len(labels)
        train_dataloader = self.build_train_loader(graphs = graphs, labels = labels)

        self.logger.info('starting self training:{}'.format(itr_ST))

        self.model = build_classifier(self.cfg.classifier)
        self.model = self.model.cuda()
        self.model.train()

        optimizer = self.build_optimizer(self.model)

        loss_func = self.build_loss_func()
        target_itr = self.get_itr_number()
        total_epoch = self.cfg.total_epoch

        meters = MetricLogger(delimiter = "  ")
        end = time.time()

        total_itr = 0
        train_over_flag = False
        key_pre_str = 'cls_{}_{}'.format(self.ITR, itr_ST)

        for epoch in range(total_epoch):
            self.logger.info('total epoch:{}, cur epoch:{}'.format(total_epoch, epoch))
            for iteration, batch in enumerate(train_dataloader):
                total_itr += 1
                data_time = time.time() - end
                batch_graph = batch['batch_graph']
                batch_labels = batch['batch_labels']

                batch_graph = move_to_device(batch_graph)
                batch_labels = move_to_device(batch_labels)

                optimizer.zero_grad()

                output = self.model(batch_graph)
                loss = loss_func(output, batch_labels)
                meters.update(loss = loss)

                loss.backward()
                optimizer.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time = batch_time, data = data_time)
                eta_seconds = meters.time.global_avg * (target_itr - total_itr)
                eta_min = eta_seconds / 60
                # eta_string = str(datetime.timedelta(seconds = int(eta_seconds)))
                if (total_itr % 100 == 0):
                    log_info = meters.get_midian_dict()
                    log_info['eta_min'] = eta_min
                    log_info['itr'] = total_itr
                    log_info['lr'] = optimizer.param_groups[0]["lr"],
                    log_info['epoch'] = epoch
                    self.logger.dict(log_info)
                    self.log_training_info_to_wandb(info_dict = log_info)

                if (total_itr == target_itr):
                    train_over_flag = True
                    break

                if ((total_itr + 1) % self.cfg.eval_interval == 0):
                    result = self.evaler(self.model, key_pre_str)
                    self.logger.dict(result['all'])
                    if (result['val'][self.model_selection_by] > self.global_best):
                        self.global_best = result['val'][self.model_selection_by]
                        self.checkpoint.save_to_file_with_name(model = self.model,
                                                               filename = 'C_ITR_{}'.format(self.ITR))

            if (train_over_flag):
                break
        result = self.evaler(self.model, key_pre_str)
        self.logger.dict(result['all'])
        if (result['val'][self.model_selection_by] > self.global_best):
            self.global_best = result['val'][self.model_selection_by]
            self.checkpoint.save_to_file_with_name(model = self.model,
                                                   filename = 'C_ITR_{}'.format(self.ITR))
            self.logger.info('global_best:{}'.format(self.global_best))

        self.checkpoint.load_from_filename(model = self.model,
                                           filename = 'C_ITR_{}'.format(self.ITR))

        res_dict = self.evaler(self.model, key_pre_str = 'global', return_labeled_samples = True)
        global_dict = res_dict['all']
        global_dict['ITR'] = self.ITR
        self.logger.dict(global_dict)
        train_labels = res_dict['train_labels']
        val_result = res_dict['val'][self.model_selection_by]
        # assert val_result == self.global_best
        train_graphs = self.evaler.dataset_to_eval['train'].graphs_networkx
        return train_graphs, train_labels, val_result
