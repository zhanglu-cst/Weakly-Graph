from compent import Global_Var
from pseudo_label.coverter import conbina_score_label_matrix
from trainer.divergence_minimization import Divergence_Minimization


class Denoiser_Syn():
    def __init__(self, cfg_dm, number_classes, number_of_process = 30):
        super(Denoiser_Syn, self).__init__()
        self.logger = Global_Var.logger()
        self.number_of_process = number_of_process
        self.number_classes = number_classes
        self.cfg_dm = cfg_dm

    def combina_scores_add(self, x1, x2):
        return x1 + x2

    def combina_score_mul(self, x1, x2):
        return x1 * x2

    def __call__(self, all_assign_graphs, all_label_vector, key_subgraph):
        # all_graphs, all_label_vector = cal_hit_matrix_func(mol_dataset, key_subgraph, self.number_of_process)

        ensembler = Divergence_Minimization(number_keys_subgraphs = len(key_subgraph), cfg_dm = self.cfg_dm)
        learnt_alpha = ensembler(all_label_vector)
        final_score_list_add = []
        final_score_list_mul = []
        major_voting_score_list = []
        only_alpha_score_list = []
        only_beta_score_list = []

        for index, item_subgraph in enumerate(key_subgraph):
            cur_alpha = learnt_alpha[index]
            origin_score = key_subgraph.subgraph_to_score[item_subgraph]
            score_add = self.combina_scores_add(cur_alpha, origin_score)
            score_mul = self.combina_score_mul(cur_alpha, origin_score)
            final_score_list_add.append(score_add)
            final_score_list_mul.append(score_mul)
            major_voting_score_list.append(1)
            only_alpha_score_list.append(cur_alpha)
            only_beta_score_list.append(origin_score)

        all_pseudo_labels_add = conbina_score_label_matrix(final_score_each_subgraph = final_score_list_add,
                                                           label_matric = all_label_vector,
                                                           number_classes = self.number_classes)
        all_pseudo_labels_mul = conbina_score_label_matrix(final_score_each_subgraph = final_score_list_mul,
                                                           label_matric = all_label_vector,
                                                           number_classes = self.number_classes)
        all_pseudo_labels_vote = conbina_score_label_matrix(final_score_each_subgraph = major_voting_score_list,
                                                            label_matric = all_label_vector,
                                                            number_classes = self.number_classes)
        all_pseudo_labels_alpha = conbina_score_label_matrix(final_score_each_subgraph = only_alpha_score_list,
                                                             label_matric = all_label_vector,
                                                             number_classes = self.number_classes)
        all_pseudo_labels_beta = conbina_score_label_matrix(final_score_each_subgraph = only_beta_score_list,
                                                            label_matric = all_label_vector,
                                                            number_classes = self.number_classes)
        ans = {'all_pseudo_labels_add': all_pseudo_labels_add, 'all_pseudo_labels_mul': all_pseudo_labels_mul,
               'all_pseudo_labels_vote': all_pseudo_labels_vote, 'all_pseudo_labels_alpha': all_pseudo_labels_alpha,
               'all_pseudo_labels_beta': all_pseudo_labels_beta}
        return all_assign_graphs, ans

#
# @PSEUDO_LABEL_ASSIGNER.register_module()
# class Label_Syn2():
#     def __init__(self, number_of_process = 30, number_classes = 3):
#         super(Label_Syn2, self).__init__()
#         self.logger = Global_Var.logger()
#         self.number_of_process = number_of_process
#         self.number_classes = number_classes
#
#     def __call__(self, mol_dataset, key_subgraph):
#         self.logger.info('generating label matrix...')
#         queue_input = multiprocessing.Queue()
#         for index, item_graph in enumerate(mol_dataset.graphs_networkx):
#             queue_input.put((item_graph, index))
#
#         global_ans = multiprocessing.Manager().list()
#         all_process = []
#
#         for i in range(self.number_of_process):
#             p = multiprocessing.Process(target = one_process,
#                                         args = (i, queue_input, global_ans, key_subgraph))
#             all_process.append(p)
#         for item in all_process:
#             item.start()
#         for item in all_process:
#             item.join()
#
#         all_graphs = []
#         all_label_vector = []
#         for item in global_ans:
#             item_graph = mol_dataset.graphs_networkx[item[0]]
#             all_graphs.append(item_graph)
#             all_label_vector.append(item[1])
#         self.logger.info('finish assigning pseudo labels')
#         ensembler = Divergence_Minimization(number_keys_subgraphs = len(key_subgraph))
#         learnt_alpha = ensembler(all_label_vector)
#         final_score_list = []
#         for index, item_subgraph in enumerate(key_subgraph):
#             cur_alpha = learnt_alpha[index]
#             origin_score = key_subgraph.subgraph_to_score[item_subgraph]
#             score = cur_alpha #* origin_score
#             final_score_list.append(score)
#         all_pseudo_labels = conbina_score_label_matrix(final_score_each_subgraph = final_score_list,
#                                                        label_matric = all_label_vector,
#                                                        number_classes = self.number_classes)
#
#         return all_graphs, all_pseudo_labels
