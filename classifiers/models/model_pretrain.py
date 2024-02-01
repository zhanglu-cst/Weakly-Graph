from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from ..build import CLASSIFIER
from ..build import build_classifier


@CLASSIFIER.register_module()
class Pretrain_Wrapper_Model(nn.Module):
    def __init__(self, cfg_classifier, number_classes, lamda_graph = 1):
        super(Pretrain_Wrapper_Model, self).__init__()
        self.classifier = build_classifier(cfg_classifier)
        self.node_linear = nn.Linear(cfg_classifier.emb_dim, number_classes + 1)
        self.lamda_graph = lamda_graph
        self.loss_func_nodes = CrossEntropyLoss()
        self.loss_func_graph = BCEWithLogitsLoss()

    def forward(self, batch):
        batch_graph = batch['batch_graph']
        batch_node_labels = batch['batch_node_labels']
        batch_graph_labels = batch['batch_graph_labels']
        graph_feats, final_node_feats = self.classifier(batch_graph, for_pretrain = True)

        # graph_feats: [2,3]   final_node_feats: [num nodes, 512]
        # batch_graph_labels = batch_graph_labels.unsqueeze(0)
        # print('graph_feats:{}'.format(graph_feats))
        # print('graph_feats shape:{}'.format(graph_feats.shape))
        # print('batch_graph_labels:{}'.format(batch_graph_labels))
        # print('batch_graph_labels shape:{}'.format(batch_graph_labels.shape))
        node_output = self.node_linear(final_node_feats)

        loss_nodes = self.loss_func_nodes(node_output, batch_node_labels)
        loss_graph = self.loss_func_graph(graph_feats, batch_graph_labels)

        loss_all = loss_graph + self.lamda_graph * loss_nodes
        return loss_all


@CLASSIFIER.register_module()
class Pretrain_Wrapper_Only_Node(nn.Module):
    def __init__(self, cfg_classifier, number_classes, lamda_graph = 1):
        super(Pretrain_Wrapper_Only_Node, self).__init__()
        self.classifier = build_classifier(cfg_classifier)
        self.node_linear = nn.Linear(cfg_classifier.emb_dim, number_classes + 1)
        self.loss_func_nodes = CrossEntropyLoss()

    def forward(self, batch):
        batch_graph = batch['batch_graph']
        batch_node_labels = batch['batch_node_labels']
        batch_graph_labels = batch['batch_graph_labels']
        graph_feats, final_node_feats = self.classifier(batch_graph, for_pretrain = True)

        # graph_feats: [2,3]   final_node_feats: [num nodes, 512]
        # batch_graph_labels = batch_graph_labels.unsqueeze(0)
        # print('graph_feats:{}'.format(graph_feats))
        # print('graph_feats shape:{}'.format(graph_feats.shape))
        # print('batch_graph_labels:{}'.format(batch_graph_labels))
        # print('batch_graph_labels shape:{}'.format(batch_graph_labels.shape))
        node_output = self.node_linear(final_node_feats)

        loss_nodes = self.loss_func_nodes(node_output, batch_node_labels)

        loss_all = loss_nodes
        return loss_all


@CLASSIFIER.register_module()
class Pretrain_Wrapper_Only_Graph(nn.Module):
    def __init__(self, cfg_classifier, number_classes, lamda_graph = 1):
        super(Pretrain_Wrapper_Only_Graph, self).__init__()
        self.classifier = build_classifier(cfg_classifier)
        self.loss_func_graph = BCEWithLogitsLoss()

    def forward(self, batch):
        batch_graph = batch['batch_graph']
        batch_node_labels = batch['batch_node_labels']
        batch_graph_labels = batch['batch_graph_labels']
        graph_feats, final_node_feats = self.classifier(batch_graph, for_pretrain = True)

        loss_graph = self.loss_func_graph(graph_feats, batch_graph_labels)

        loss_all = loss_graph
        return loss_all
