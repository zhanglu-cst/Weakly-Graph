from dgllife.model.model_zoo.mpnn_predictor import MPNNPredictor

from ..build import CLASSIFIER


@CLASSIFIER.register_module()
class MPNN(MPNNPredictor):
    def __init__(self, **kwargs):
        self.output_dim = kwargs.pop('output_dim')
        super(MPNN, self).__init__(**kwargs)

    def forward(self, g):
        node_feats = g.ndata["x"]
        edge_feats = g.edata["x"]
        out = super(MPNN, self).forward(g, node_feats, edge_feats)
        return out
