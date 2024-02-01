import multiprocessing

import numpy

from compent.global_var import Global_Var
from compent.networkx_ops import judge_contain_a_subgraph


def one_process(rank, queue_input, global_ans, key_subgraph):
    print('start rank:{}'.format(rank))
    while not queue_input.empty():
        item_graph, index = queue_input.get()
        label_vector_cur_graph = numpy.full(len(key_subgraph), -1)
        for key_index, item_key_subgraph in enumerate(key_subgraph):
            if (judge_contain_a_subgraph(item_graph, item_key_subgraph)):
                class_index = key_subgraph.subgraph_to_class[item_key_subgraph]
                label_vector_cur_graph[key_index] = class_index
        if (numpy.sum(label_vector_cur_graph) > -len(key_subgraph)):
            ans_item = (index, label_vector_cur_graph)
            global_ans.append(ans_item)
            if (len(global_ans) % 100 == 0):
                print('rank:{}, global len:{}, left:{}'.format(rank, len(global_ans), queue_input.qsize()))
    print('rank:{}, finish'.format(rank))


def cal_hit_matrix_func(syn_dataset, key_subgraph, number_of_process = 40):
    logger = Global_Var.logger()
    logger.info('generating label matrix...')
    queue_input = multiprocessing.Queue()
    for index, item_graph in enumerate(syn_dataset.graphs_networkx):
        queue_input.put((item_graph, index))

    global_ans = multiprocessing.Manager().list()
    all_process = []

    for i in range(number_of_process):
        p = multiprocessing.Process(target = one_process,
                                    args = (i, queue_input, global_ans, key_subgraph))
        all_process.append(p)
    for item in all_process:
        item.start()
    for item in all_process:
        item.join()

    all_assign_graphs = []
    all_label_vector = []
    for item in global_ans:
        item_graph = syn_dataset.graphs_networkx[item[0]]
        all_assign_graphs.append(item_graph)
        all_label_vector.append(item[1])
    logger.info('finish assigning pseudo labels')
    return all_assign_graphs, all_label_vector


def cal_hit_matrix_func_for_mol(mol_dataset, key_subgraph):
    logger = Global_Var.logger()
    logger.info('generating label matrix...')
    all_assign_graphs = []
    all_label_vector = []
    for item_smile in mol_dataset:
        label_vector_cur_graph = numpy.full(len(key_subgraph), -1)
        if (item_smile not in mol_dataset.smile_to_subgraphs):
            continue
        all_subgraphs_cur_smile = mol_dataset.smile_to_subgraphs[item_smile]
        for index_key, item_key_subgraph in enumerate(key_subgraph):
            if (item_key_subgraph in all_subgraphs_cur_smile):
                class_index = key_subgraph.subgraph_to_class[item_key_subgraph]
                label_vector_cur_graph[index_key] = class_index
        if (numpy.sum(label_vector_cur_graph) > -len(key_subgraph)):
            all_assign_graphs.append(item_smile)
            all_label_vector.append(label_vector_cur_graph)
    return all_assign_graphs, all_label_vector
