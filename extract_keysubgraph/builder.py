from mmcv import Registry

TOKEN_MOL = Registry('token_mol')
UPDETAR = Registry('updater')


def build_token_mol(cfg):
    return TOKEN_MOL.build(cfg)


def build_extractor(cfg):
    return UPDETAR.build(cfg)
