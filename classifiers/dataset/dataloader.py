import dgl
import torch
from torch.utils.data import DataLoader


def collector_MOL(batch):
    batch_graph, batch_labels, batch_smiles = map(list, zip(*batch))
    batch_graph = dgl.batch(batch_graph)
    batch_labels = torch.tensor(batch_labels).long()
    return {'batch_graph': batch_graph, 'batch_labels': batch_labels, 'batch_smiles': batch_smiles}


def collector_DGL(batch):
    batch_graph, batch_labels = map(list, zip(*batch))
    batch_graph = dgl.batch(batch_graph)
    batch_labels = torch.tensor(batch_labels).long()
    return {'batch_graph': batch_graph, 'batch_labels': batch_labels}


# class Collector():
#     def __init__(self, key_list = ('batch_graph', 'batch_labels', 'batch_smiles'), label_type = 'float'):
#         self.key_list = key_list
#         self.label_type = label_type
#
#     def transform_labels(self, item_data):
#         if (self.label_type == 'float'):
#             item_data = torch.tensor(item_data).float()
#         else:
#             item_data = torch.tensor(item_data).long()
#         return item_data
#
#     def __call__(self, batch):
#         map_res = list(map(list, zip(*batch)))
#         assert len(map_res) == len(self.key_list)
#         res_D = {}
#         for item_key, item_data in zip(self.key_list, map_res):
#             if (item_key == 'batch_graph'):
#                 item_data = dgl.batch(item_data)
#             elif (item_key == 'batch_labels'):
#                 item_data = self.transform_labels(item_data)
#             res_D[item_key] = item_data
#         return res_D


def build_simple_dataloader_mol(cfg, dataset, for_train):
    loader = DataLoader(dataset,
                        batch_size = cfg.dataloader.batch_size,
                        shuffle = for_train,
                        collate_fn = collector_MOL,
                        num_workers = cfg.dataloader.number_workers)
    return loader


def build_simple_dataloader_DGL(cfg, dataset, for_train):
    loader = DataLoader(dataset,
                        batch_size = cfg.dataloader.batch_size,
                        shuffle = for_train,
                        collate_fn = collector_DGL,
                        num_workers = cfg.dataloader.number_workers)
    return loader
