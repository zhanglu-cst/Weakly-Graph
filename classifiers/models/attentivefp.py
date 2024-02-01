# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.
import torch
from dgllife.model import AttentiveFPGNN as AttentiveFPGNN_DGL, AttentiveFPReadout

from ..build import CLASSIFIER


# @CLASSIFIER.register_module()
# class AttentiveFPGNN(AttentiveFPGNN_DGL):
#     def __init__(self, num_timesteps, get_node_weight = False, output_dim = 1, **kwargs):
#         super(AttentiveFPGNN, self).__init__(**kwargs)
#         feat_size = kwargs.get("graph_feat_size")
#         self.readout = AttentiveFPReadout(
#                 num_timesteps = num_timesteps,
#                 feat_size = feat_size,
#                 dropout = kwargs.get("dropout"))
#         self.fc = nn.Linear(feat_size, output_dim)
#
#     def forward(self, input):
#         node_feats = input.ndata["x"]
#         edge_feats = input.edata["x"]
#         node_feats = super().forward(input, node_feats, edge_feats)
#         graph_feats = self.readout(input, node_feats, False)
#
#         output = self.fc(graph_feats)
#         return output


@CLASSIFIER.register_module()
class AttentiveFPGNN(torch.nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, num_layers, num_timesteps, output_dim, emb_dim = 512):
        super(AttentiveFPGNN, self).__init__()
        self.gnn = AttentiveFPGNN_DGL(node_feat_size = node_feature_size,
                                      edge_feat_size = edge_feature_size,
                                      num_layers = num_layers,
                                      graph_feat_size = emb_dim)
        self.readout = AttentiveFPReadout(num_timesteps = num_timesteps,
                                          feat_size = emb_dim, )
        # self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = torch.nn.Linear(emb_dim, output_dim)

    def forward(self, input, return_last_feature = False, for_pretrain = False):
        # input = dgl.add_self_loop(input)
        node_feats = input.ndata["x"]
        edge_feats = input.edata["x"]
        node_feats = self.gnn(input, node_feats, edge_feats)
        graph_feats_origin = self.readout(input, node_feats, False)
        # print('graph_feats_origin shape:{}'.format(graph_feats_origin.shape))
        graph_feats = self.predict(graph_feats_origin)
        if (return_last_feature == False and for_pretrain == False):
            return graph_feats
        elif (return_last_feature):
            return graph_feats, graph_feats_origin
        elif (for_pretrain):
            return graph_feats, node_feats
