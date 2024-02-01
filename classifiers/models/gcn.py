import dgl
# from dgllife.model.model_zoo.gcn_predictor import GCNPredictor
import torch
from dgl.nn.pytorch import AvgPooling
from dgllife.model.gnn.gcn import GCN

from ..build import CLASSIFIER


@CLASSIFIER.register_module()
class GCN_Predicter(torch.nn.Module):
    def __init__(self, in_feats, output_dim, hidden_feats = None, gnn_norm = None, activation = None,
                 residual = None, batchnorm = None, dropout = None, emb_dim = 512):
        super(GCN_Predicter, self).__init__()
        self.gnn = GCN(in_feats = in_feats,
                       hidden_feats = hidden_feats,
                       gnn_norm = gnn_norm,
                       activation = activation,
                       residual = residual,
                       batchnorm = batchnorm,
                       dropout = dropout)
        # gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = AvgPooling()
        self.predict = torch.nn.Linear(hidden_feats[-1], output_dim)

    def forward(self, input, return_last_feature = False, for_pretrain = False):
        # input = dgl.add_self_loop(input)
        node_feats = input.ndata["x"]
        # edge_feats = g.edata["x"]
        node_feats = self.gnn(input, node_feats)
        graph_feats_origin = self.readout(input, node_feats)
        # print('graph_feats_origin shape:{}'.format(graph_feats_origin.shape))
        graph_feats = self.predict(graph_feats_origin)
        if (return_last_feature == False and for_pretrain == False):
            return graph_feats
        elif (return_last_feature):
            return graph_feats, graph_feats_origin
        elif (for_pretrain):
            return graph_feats, node_feats
