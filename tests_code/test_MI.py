import os

import dgl
import torch

from classifiers import build_classifier
from compent.networkx_ops import batch_networkx_to_DGL
from compent.utils import move_to_device
from dataset import build_dataset
from extract_keysubgraph.enumerate_subgraphs_nx import Enumerate_Subgraphs_NetworkX

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from mmcv import Config

from compent.checkpoint import CheckPointer

cfg = Config.fromfile('config/syn/cfg_syn.py')
cfg.logger.project = 'syn_pipeline'
cfg.logger.name = 'st1_GINdim_512'
cfg.work_dir = os.path.join('./work_dir', cfg.logger.project, cfg.logger.name)
checkpoint = CheckPointer(cfg.work_dir, rank = 0)
model = build_classifier(cfg.classifier)

model = model.cuda()
model.eval()
checkpoint.load_from_filename(model = model, filename = 'C_ITR_{}'.format(0))
syn_dataset = build_dataset(cfg.dataset.train)
enum_subgraph = Enumerate_Subgraphs_NetworkX(min_node_number = 5, max_node_number = 15)

# for index in range(len(mol_dataset)):
index = 300
item_graph = syn_dataset.graphs_dgl[index]
item_label = syn_dataset.labels[index]
item_graph = move_to_device(item_graph)
pred, feature = model(item_graph, return_last_feature = True)
print('pred:{}'.format(pred))
print('label:{}'.format(item_label))
class_pred = torch.argmax(pred)
print('class pred:{}'.format(class_pred))
# print('last feature:{}'.format(feature))
print('dim:{}'.format(feature.shape))
item_graph_nx = syn_dataset.graphs_networkx[index]
all_subgraphs = enum_subgraph(graph_networkx = item_graph_nx)
print('number subgraphs:{}'.format(len(all_subgraphs)))
all_subgraph_dgl = batch_networkx_to_DGL(all_subgraphs)
all_res = []

all_subgraph_dgl = move_to_device(all_subgraph_dgl)
all_subgraph_dgl = dgl.batch(all_subgraph_dgl)
item_pred, item_feature = model(all_subgraph_dgl, return_last_feature = True)
item_feature = item_feature.cpu()
print(item_pred.shape)
print(item_feature.shape)

# for item_subgraph_nx, item_subgraph_dgl in zip(all_subgraphs, all_subgraph_dgl):
#     item_subgraph_dgl = move_to_device(item_subgraph_dgl)
#     item_pred, item_feature = model(item_subgraph_dgl, return_last_feature = True)
#     mi = calc_MI(X = feature, Y = item_feature)
#     all_res.append([item_subgraph_nx, mi])
# all_res = sorted(all_res, key = lambda x: x[1], reverse = True)
# for item_res in all_res:
#     item_sub = item_res[0]
#     mi = item_res[1]
#     print('mi:{}, num nodes:{}'.format(mi, item_sub.number_of_nodes()))
#     show_a_graph(item_sub)
#     xxx = input()
