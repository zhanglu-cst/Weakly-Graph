from mmcv import Registry

PSEUDO_LABEL_ASSIGNER = Registry('pseudo')


def build_pseudo_label_assigner(cfg):
    return PSEUDO_LABEL_ASSIGNER.build(cfg)
