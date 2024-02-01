import numpy

from compent import Global_Var
# from dataset.key_subgraph_mol import Key_SubGraph_Mol
# from dataset.moleculars import Moleculars
from pseudo_label.metric import cal_metric_binary_for_pseudo
from .build import PSEUDO_LABEL_ASSIGNER


@PSEUDO_LABEL_ASSIGNER.register_module()
class Assign_Voter_Mol():
    def __init__(self, number_classes = 2):
        self.number_classes = number_classes
        self.logger = Global_Var.logger()

    def get_one_smile_label(self, item_smile, key_subgraph_mol, mol_dataset):
        tokens_cur_smile = mol_dataset.smile_to_tokens[item_smile]
        prob_each_class = numpy.zeros(self.number_classes)
        map_token_to_label = key_subgraph_mol.key_to_label
        map_token_to_score = key_subgraph_mol.key_to_score
        for item_token in tokens_cur_smile:
            if (item_token in map_token_to_label):
                label = map_token_to_label[item_token]
                score = map_token_to_score[item_token]
                prob_each_class[label] += score
        if (numpy.sum(prob_each_class) == 0):
            return None
        else:
            return numpy.argmax(prob_each_class)

    def __call__(self, key_subgraph_mol, mol_dataset):
        all_asigned_labels = []
        all_asigned_smiles = []
        all_GT_labels = []
        for item_smile in mol_dataset.smiles:
            voter_result = self.get_one_smile_label(item_smile, key_subgraph_mol, mol_dataset)
            if (voter_result is not None):
                all_asigned_labels.append(voter_result)
                all_asigned_smiles.append(item_smile)
                gt_label = mol_dataset.smile_to_label[item_smile]
                all_GT_labels.append(gt_label)

        metric = cal_metric_binary_for_pseudo(all_pseudo_labels = all_asigned_labels, all_gt_labels = all_GT_labels,
                                              mol_dataset = mol_dataset, pre_key = 'pse_vote')
        self.logger.info('cur metric:{}'.format(metric))
        self.logger.dict(metric)
        return {'all_asigned_smiles': all_asigned_smiles, 'all_asigned_labels': all_asigned_labels, 'metric': metric}
