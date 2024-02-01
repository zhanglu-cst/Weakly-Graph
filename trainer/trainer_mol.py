import time

from classifiers.build import build_classifier
from compent.global_var import Global_Var
from trainer.cal_hit_matrix import cal_hit_matrix_func_for_mol
from trainer.denoiser_syn import Denoiser_Syn
from trainer.pretrainer_mol import SSL_Pretrainer_Mol
from trainer.trainer_st_mol import Trainer_ST_MOL
from .build import TRAINER


@TRAINER.register_module()
class Trainer_Mol():
    def __init__(self, cfg_model_pretrain, number_classes, number_of_process, cfg_pretrain_loader,
                 cfg_pretrain_optimizer, cfg_finetune_loader, cfg_finetune_optimizer, cfg_trainer_finetune,
                 cfg_dm, pseudo_label_key):
        super(Trainer_Mol, self).__init__()
        self.logger = Global_Var.logger()
        self.number_classes = number_classes
        self.cfg_model_pretrain = cfg_model_pretrain
        self.number_of_process = number_of_process
        self.cfg_pretrain_loader = cfg_pretrain_loader
        self.cfg_pretrain_optimizer = cfg_pretrain_optimizer
        self.cfg_finetune_loader = cfg_finetune_loader
        self.cfg_finetune_optimizer = cfg_finetune_optimizer
        self.cfg_trainer_finetune = cfg_trainer_finetune
        self.cfg_dm = cfg_dm
        self.pseudo_label_key = pseudo_label_key

    def __call__(self, mol_dataset, key_subgraph):
        # ITR = Global_Var.get('ITR')

        all_cover_graphs, all_label_vector = cal_hit_matrix_func_for_mol(mol_dataset, key_subgraph)

        model_pretrain = build_classifier(self.cfg_model_pretrain)

        s_point_ssl = time.time()
        pretrainer = SSL_Pretrainer_Mol(key_subgraph = key_subgraph,
                                        mol_dataset = mol_dataset,
                                        all_label_vector = all_label_vector,
                                        number_classes = self.number_classes,
                                        cfg_dataloader = self.cfg_pretrain_loader,
                                        cfg_optimizer = self.cfg_pretrain_optimizer)
        classifier = pretrainer.pretrain_annotator(model_pretrain)
        e_point_ssl = time.time()
        self.logger.dict({'time/ssl': e_point_ssl - s_point_ssl})

        denoiser = Denoiser_Syn(number_of_process = self.number_of_process, cfg_dm = self.cfg_dm,
                                number_classes = self.number_classes)
        all_cover_graphs, all_pseudo_labels_dict = denoiser(all_cover_graphs, all_label_vector, key_subgraph)
        all_pseudo_denoise_label = all_pseudo_labels_dict[self.pseudo_label_key]

        mol_dataset.cal_pseudo_quality(pseudo_graphs = all_cover_graphs, pseudo_labels = all_pseudo_denoise_label,
                                       pre_key = 'pseudo_quality_for_finetune')

        for item_pseudo_label_key, pseudo_labels_value in all_pseudo_labels_dict.items():
            mol_dataset.cal_pseudo_quality(pseudo_graphs = all_cover_graphs, pseudo_labels = pseudo_labels_value,
                                           pre_key = item_pseudo_label_key)

        e_point_dm = time.time()
        self.logger.dict({'time/DM': e_point_dm - e_point_ssl})

        # finetuner = Classifier_Finetuner(cfg_dataloader = self.cfg_finetune_loader,
        #                                  cfg_optimizer = self.cfg_finetune_optimizer,
        #                                  classifier = classifier,
        #                                  cfg_trainer_finetune = self.cfg_trainer_finetune)
        finetuner = Trainer_ST_MOL(cfg = self.cfg_trainer_finetune)
        all_pred_graphs, anno_label_result = finetuner.train_model(smiles = all_cover_graphs,
                                                                   labels = all_pseudo_denoise_label)

        # finetuner.finetune(all_cover_graphs, all_pseudo_denoise_label)
        # anno_label_result = finetuner.do_annotating(mol_dataset.graphs_networkx)

        # all_pred_graphs, anno_label_result = finetuner.finetune_with_trainer(all_assign_graphs = all_cover_graphs,
        #                                                                      all_pseudo_labels = all_pseudo_denoise_label)

        e_point_finetune = time.time()
        self.logger.dict({'time/finetune': e_point_finetune - e_point_dm})

        return all_pred_graphs, anno_label_result, classifier
