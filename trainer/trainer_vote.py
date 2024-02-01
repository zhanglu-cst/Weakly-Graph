from compent.global_var import Global_Var
from trainer.cal_hit_matrix import cal_hit_matrix_func
from trainer.denoiser_syn import Denoiser_Syn
from trainer.fintuner import Classifier_Finetuner
from .build import TRAINER


@TRAINER.register_module()
class Trainer_Vote_Syn():
    def __init__(self, number_classes, number_of_process, cfg_finetune_loader, cfg_finetune_optimizer,
                 cfg_trainer_finetune):
        super(Trainer_Vote_Syn, self).__init__()
        self.logger = Global_Var.logger()
        self.number_classes = number_classes
        self.number_of_process = number_of_process

        self.cfg_finetune_loader = cfg_finetune_loader
        self.cfg_finetune_optimizer = cfg_finetune_optimizer
        self.cfg_trainer_finetune = cfg_trainer_finetune

    def __call__(self, syn_dataset, key_subgraph):
        all_cover_graphs, all_label_vector = cal_hit_matrix_func(syn_dataset, key_subgraph, self.number_of_process)

        denoiser = Denoiser_Syn(number_of_process = self.number_of_process, number_classes = self.number_classes)
        all_cover_graphs, all_pseudo_labels_dict = denoiser(all_cover_graphs, all_label_vector, key_subgraph)
        all_pseudo_denoise_label = all_pseudo_labels_dict['all_pseudo_labels_vote']

        finetuner = Classifier_Finetuner(cfg_dataloader = self.cfg_finetune_loader,
                                         cfg_optimizer = self.cfg_finetune_optimizer,
                                         classifier = None,
                                         cfg_trainer_finetune = self.cfg_trainer_finetune)

        all_pred_graphs, anno_label_result = finetuner.finetune_with_trainer(all_assign_graphs = all_cover_graphs,
                                                                             all_pseudo_labels = all_pseudo_denoise_label)

        syn_dataset.cal_pseudo_quality(pseudo_graphs = all_cover_graphs, pseudo_labels = all_pseudo_denoise_label,
                                       pre_key = 'pseudo_quality_for_finetune')

        return all_pred_graphs, anno_label_result, None
