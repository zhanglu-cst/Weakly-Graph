from mmcv import Registry

DATASET = Registry('dataset')
KEY_SUBGRAPH = Registry('key_subgraph')


def build_dataset(cfg):
    return DATASET.build(cfg)


def build_key_subgraph(cfg):
    return KEY_SUBGRAPH.build(cfg)
