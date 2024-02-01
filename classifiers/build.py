from mmcv import Registry

from .dataset.dataloader import build_simple_dataloader_mol, build_simple_dataloader_DGL

CLASSIFIER = Registry('classifier')

TRAIN_EVAL_DATASET = Registry('train_dataset')


def build_classifier(cfg):
    return CLASSIFIER.build(cfg)


def build_torch_dataset_for_mol(cfg, smiles, labels, for_train = False):
    config = cfg.train_val_dataset
    config['smiles'] = smiles
    config['labels'] = labels
    config['for_train'] = for_train
    # config = dict(type = type, smiles = smiles, labels = labels, for_train = for_train)
    return TRAIN_EVAL_DATASET.build(config)


def build_torch_dataset_for_DGL(cfg, graphs_dgl, labels, for_train = False):
    config = cfg.train_val_dataset
    config['graphs_dgl'] = graphs_dgl
    config['labels'] = labels
    config['for_train'] = for_train
    return TRAIN_EVAL_DATASET.build(config)


def build_dataloader(cfg, dataset, for_train):
    type = cfg.dataloader.type
    if (type == 'build_simple_dataloader_mol'):
        return build_simple_dataloader_mol(cfg, dataset, for_train)
    elif (type == 'build_simple_dataloader_DGL'):
        return build_simple_dataloader_DGL(cfg, dataset, for_train)
    else:
        raise Exception
