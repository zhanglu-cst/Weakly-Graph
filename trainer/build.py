from mmcv import Registry

TRAINER = Registry('trainer')


def build_trainer(cfg):
    return TRAINER.build(cfg)
