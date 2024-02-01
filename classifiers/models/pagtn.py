from dgllife.model.model_zoo.pagtn_predictor import PAGTNPredictor

from ..build import CLASSIFIER


@CLASSIFIER.register_module()
class PAGTN(PAGTNPredictor):
    def __init__(self, **kwargs):
        self.output_dim = kwargs.pop('output_dim')
        super(PAGTN, self).__init__(**kwargs)

    def forward(self, g):
        # g = dgl.add_self_loop(g)
        node_feats = g.ndata["x"]
        edge_feats = g.edata["x"]
        out = super(PAGTN, self).forward(g, node_feats, edge_feats)
        return out
