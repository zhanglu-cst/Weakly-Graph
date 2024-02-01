import dgl
import numpy
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from compent.global_var import Global_Var
from compent.networkx_ops import batch_networkx_to_DGL
from compent.utils import move_to_device
from trainer.trainer_st_syn import Trainer_ST_SYN


class Dataset_Finetune(Dataset):
    def __init__(self, all_assign_graphs, all_pseudo_labels):
        super(Dataset_Finetune, self).__init__()
        self.graphs_dgl = batch_networkx_to_DGL(all_assign_graphs)
        if (all_pseudo_labels is not None):
            self.all_labels = torch.tensor(all_pseudo_labels).long()
        else:
            self.all_labels = None

    def __getitem__(self, index):
        if (self.all_labels is None):
            return self.graphs_dgl[index]
        else:
            return self.graphs_dgl[index], self.all_labels[index]

    def __len__(self):
        return len(self.graphs_dgl)


def collector_finetune(batch):
    batch_graph, batch_graph_labels = map(list, zip(*batch))
    batch_graph = dgl.batch(batch_graph)
    batch_graph_labels = torch.LongTensor(batch_graph_labels)
    return {'batch_graph': batch_graph, 'batch_graph_labels': batch_graph_labels}


def collector_annotator(batch):
    batch_graph = dgl.batch(batch)
    return batch_graph


class Classifier_Finetuner():
    def __init__(self, cfg_dataloader, cfg_optimizer, cfg_trainer_finetune, classifier):
        super(Classifier_Finetuner, self).__init__()
        self.cfg_dataloader = cfg_dataloader
        self.cfg_optimizer = cfg_optimizer
        self.logger = Global_Var.logger()
        self.classifier = classifier
        self.cfg_trainer_finetune = cfg_trainer_finetune

    def finetune_with_trainer(self, all_assign_graphs, all_pseudo_labels):
        trainer = Trainer_ST_SYN(self.cfg_trainer_finetune)
        all_graphs, all_pred_labels = trainer.train_model(graphs = all_assign_graphs, labels = all_pseudo_labels)
        return all_graphs, all_pred_labels

    def finetune(self, all_assign_graphs, all_pseudo_labels):
        self.logger.info('start finetuning classifier...', key = 'state')
        self.classifier.train()
        dataset_finetune = Dataset_Finetune(all_assign_graphs, all_pseudo_labels)
        dataloader = DataLoader(dataset_finetune, batch_size = self.cfg_dataloader.batch_size, shuffle = True,
                                collate_fn = collector_finetune)

        loss_func = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(self.classifier.parameters(), lr = self.cfg_optimizer.lr)

        max_epoch = self.cfg_dataloader.epoch
        for epoch in range(max_epoch):
            self.logger.dict({'finetune/finetune_epoch': epoch})
            loss_record = []
            for itr, batch in enumerate(dataloader):
                batch = move_to_device(batch)
                batch_graph = batch['batch_graph']
                batch_graph_labels = batch['batch_graph_labels']
                out = self.classifier(batch_graph)
                loss = loss_func(out, batch_graph_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_record.append(loss.data.item())
                if (itr % 100 == 0):
                    mean_loss = numpy.mean(loss_record)
                    self.logger.dict({'finetune/finetune_loss': mean_loss, 'finetune/finetune_itr': itr})
                    loss_record = []
        self.logger.info('finish finetune classifier', key = 'state')

    def do_annotating(self, all_graphs):
        self.logger.info('start do annotating...', key = 'state')
        self.classifier.eval()
        dataset_annotator = Dataset_Finetune(all_graphs, all_pseudo_labels = None)
        dataloader = DataLoader(dataset_annotator, batch_size = self.cfg_dataloader.batch_size, shuffle = False,
                                collate_fn = collector_annotator)
        pred_results = []
        for batch in dataloader:
            batch = move_to_device(batch)
            out = self.classifier(batch)
            pred_class = torch.argmax(out, dim = 1)
            pred_results.append(pred_class)
        pred_results = torch.cat(pred_results, dim = 0)
        pred_results = pred_results.tolist()
        return pred_results
