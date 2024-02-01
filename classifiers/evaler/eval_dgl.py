import torch

from classifiers.build import build_torch_dataset_for_DGL, build_dataloader
from compent import Global_Var
from compent.utils import move_to_device
from dataset import build_dataset
from pseudo_label.metric import cal_metric_multi_class


class Evaler_DGL():
    def __init__(self, cfg):
        self.eval_on = cfg.eval_on
        # self.number_classes = cfg.number_classes
        self.logger = Global_Var.logger()
        self.logger.info('eval on:{}'.format(self.eval_on))
        self.dataset_to_eval = {}
        for item_key in self.eval_on:
            item_cfg = cfg.dataset[item_key]
            item_dataset = build_dataset(item_cfg)
            self.dataset_to_eval[item_key] = item_dataset
        self.split_to_loader = {}
        for item_key in self.eval_on:
            item_set = self.dataset_to_eval[item_key]
            torch_dataset = build_torch_dataset_for_DGL(cfg, item_set.graphs_dgl, item_set.labels, for_train = False)
            torch_loader = build_dataloader(cfg, torch_dataset, for_train = False)
            self.split_to_loader[item_key] = torch_loader

    def __call__(self, model, key_pre_str = '', return_labeled_samples = False):
        model.eval()
        results = {}
        softmax_op = torch.nn.Softmax(dim = 1)
        with torch.no_grad():
            for item_split in self.eval_on:
                all_pred_score = []
                all_pred_classes = []
                all_gt_labels = []
                self.logger.info('starting infer:{}'.format(item_split))
                cur_loader = self.split_to_loader[item_split]
                for batch in cur_loader:
                    batch_graph = batch['batch_graph']
                    batch_labels = batch['batch_labels']

                    batch_graph = move_to_device(batch_graph)
                    output = model(batch_graph)
                    output = output.cpu()

                    _, batch_pred_class = torch.max(output, dim = 1)
                    all_gt_labels.append(batch_labels)
                    all_pred_classes.append(batch_pred_class)
                    output = softmax_op(output)
                    pred_score_cur = output[:, 1]
                    all_pred_score.append(pred_score_cur)

                all_pred_score = torch.cat(all_pred_score, dim = 0)
                all_pred_classes = torch.cat(all_pred_classes, dim = 0)
                all_gt_labels = torch.cat(all_gt_labels, dim = 0)
                # if (self.number_classes > 2):
                metric = cal_metric_multi_class(all_pred_class = all_pred_classes, all_gt_labels = all_gt_labels)
                # else:
                #     metric = cal_metric_binary_class(all_pred_scores = all_pred_score, all_gt_labels = all_gt_labels)
                self.logger.info(metric)
                results[item_split] = metric

                if (item_split == 'train' and return_labeled_samples):
                    results['train_labels'] = all_pred_classes

        model.train()
        combina_key_values = {}
        for item_split, cur_split_metric in results.items():
            if (item_split in self.eval_on):
                for item_key, value in cur_split_metric.items():
                    combina_key = '{}/{}_{}'.format(key_pre_str, item_split, item_key)
                    combina_key_values[combina_key] = value
        results['all'] = combina_key_values
        return results
