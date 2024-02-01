import os
import pickle
import random

import networkx
import numpy

from compent.networkx_ops import Graphs_Set, get_random_graph, judge_contain_a_subgraph
from compent.utils import get_random_value

NUM_GRAPHS = 1000

DIR_SAVE = './data/syn_good'

if (os.path.exists(DIR_SAVE) == False):
    os.makedirs(DIR_SAVE)
    print('make dir:{}'.format(DIR_SAVE))


def generate_key_subgraph_k_regular(d_list = (4, 5), n_list = (6, 8, 10)):
    ans = []
    for d in d_list:
        for n in n_list:
            item = networkx.random_regular_graph(d = d, n = n)
            # show_a_graph(item)
            ans.append(item)
    return ans


def generate_key_subgraph_k_star(n_list = (5, 6, 7, 8, 9, 10)):
    ans = []
    for n in n_list:
        item = networkx.star_graph(n = n)
        # show_a_graph(item)
        ans.append(item)
    return ans


def generate_key_subgraph_grid(w = (2, 3), h = (3, 4, 5)):
    ans = []
    for item_w in w:
        for item_h in h:
            item = networkx.grid_graph(dim = (item_w, item_h))
            # show_a_graph(item)
            ans.append(item)
    return ans


keywords_class1 = generate_key_subgraph_k_regular()
keywords_class2 = generate_key_subgraph_k_star()
keywords_class3 = generate_key_subgraph_grid()
all_key_subgraphs = [keywords_class1, keywords_class2, keywords_class3]

all_subgraph_set = set()
subgraph_to_class = {}
subgraph_to_index = {}
for class_index, item_class in enumerate(all_key_subgraphs):
    for item in item_class:
        all_subgraph_set.add(item)
        subgraph_to_class[item] = class_index
        subgraph_to_index[item] = len(subgraph_to_index)


# ---------------------------------------------------------

# def pick_random_subgraph():
#     class_indexes = random.sample(population = [0, 1, 2], k = 1)
#     all_subgraphs = []
#     for item_class in class_indexes:
#         subgraph = random.sample(population = all_key_subgraphs[item_class], k = 1)
#         all_subgraphs += subgraph
#     return all_subgraphs, class_indexes
def pick_random_subgraph():
    number_classes = get_random_value(mu = 1, sigma = 0.65, lower_bound = 1, upper_bound = 2.9)
    number_classes = int(number_classes)
    class_indexes = random.sample(population = [0, 1, 2], k = number_classes)
    if (number_classes == 1):
        item_class = class_indexes[0]
        number_subgraphs = get_random_value(mu = 1.3, sigma = 0.7, lower_bound = 1, upper_bound = 2.9)
        number_subgraphs = int(number_subgraphs)
        subgraphs = random.sample(population = all_key_subgraphs[item_class], k = number_subgraphs)
        class_indexes = [item_class] * number_subgraphs
        return subgraphs, class_indexes
    else:
        subgraphs = []
        for item_class in class_indexes:
            item_subgraph = random.sample(population = all_key_subgraphs[item_class], k = 1)
            subgraphs += item_subgraph
        return subgraphs, class_indexes


def remove_all_edges_among_nodes(nodes, graph):
    res = []
    for i in range(len(nodes)):
        s = nodes[i]
        for j in range(i, len(nodes)):
            e = nodes[j]
            res.append((s, e))

    graph.remove_edges_from(res)
    return graph


def combine_subgraph_to_big_graph(big_graph, subgraphs):
    # show_a_graph(big_graph)
    total_number_nodes = big_graph.number_of_nodes()
    number_nodes_each_subgraph = []
    for subgraph in subgraphs:
        # show_a_graph(subgraph)
        num_nodes = subgraph.number_of_nodes()
        number_nodes_each_subgraph.append(num_nodes)
    number_pick_nodes = sum(number_nodes_each_subgraph)
    if (number_pick_nodes > total_number_nodes):
        print('number_pick_nodes > total_number_nodes')
        return None
    all_picked_replace_nodes = random.sample(population = list(big_graph.nodes), k = number_pick_nodes)
    start_index = 0
    picked_nodes_each_subgraph = []
    for num_nodes in number_nodes_each_subgraph:
        end_index = start_index + num_nodes
        picked_nodes_cur_subgraph = all_picked_replace_nodes[start_index:end_index]
        picked_nodes_each_subgraph.append(picked_nodes_cur_subgraph)
        start_index = end_index
    for picked_nodes, subgraph in zip(picked_nodes_each_subgraph, subgraphs):
        # print('picked_nodes:{}'.format(picked_nodes))
        remove_all_edges_among_nodes(nodes = picked_nodes, graph = big_graph)
        origin_edges = subgraph.edges
        map_origin_id_to_picked = {}
        origin_node_ids = subgraph.nodes
        assert len(origin_node_ids) == len(picked_nodes)
        for item_origin, item_picked in zip(origin_node_ids, picked_nodes):
            map_origin_id_to_picked[item_origin] = item_picked
        transfer_edges = []
        # print('map:{}'.format(map_origin_id_to_picked))
        for item_edge in origin_edges:
            s, e = item_edge
            ns = map_origin_id_to_picked[s]
            ne = map_origin_id_to_picked[e]
            n_edge = [ns, ne]
            transfer_edges.append(n_edge)
        big_graph.add_edges_from(transfer_edges)
    # show_a_graph(big_graph)
    return big_graph


# def get_GT_label(subgraphs, class_indexes):
#     score_each_class = numpy.zeros(3)
#     for item_subgraph, item_class in zip(subgraphs, class_indexes):
#         nodes = item_subgraph.number_of_nodes()
#         score_each_class[item_class] += nodes
#     class_res = numpy.argmax(score_each_class)
#     return class_res

def get_GT_label(subgraphs, class_indexes):
    class_res = numpy.max(class_indexes)
    return class_res


def judge_contain_other_subgraph(big_graph, class_indexes):
    exist_class = set(class_indexes)
    other_classes = {0, 1, 2} - exist_class
    need_check_subgraph = []
    for item_class in other_classes:
        need_check_subgraph += all_key_subgraphs[item_class]
    # print('need_check_subgraph len:{}'.format(len(need_check_subgraph)))
    for item_graph in need_check_subgraph:
        res = judge_contain_a_subgraph(big_graph = big_graph, subgraph = item_graph)
        if (res):
            cur_class = subgraph_to_class[item_graph]
            index = subgraph_to_index[item_graph]
            print('find class:{}, index:{}'.format(cur_class, index))
            return True
    return False


all_generated_graphs = Graphs_Set()
all_labels = []
index_finish = 0
while True:
    # print('start a new graph...')
    subgraphs, class_indexes = pick_random_subgraph()
    print('cur class index:{}'.format(class_indexes))
    # for item in subgraphs:
    #     show_a_graph(item)
    # print('class:{}'.format(class_indexes))

    total_number_nodes = 0
    for item_sub in subgraphs:
        total_number_nodes += item_sub.number_of_nodes()
    print('total_number_keysubgraph_nodes:{}'.format(total_number_nodes))
    n = get_random_value(mu = total_number_nodes + 10, sigma = 6, lower_bound = total_number_nodes, upper_bound = 30)
    n = int(n)
    print('total number nodes of graph:{}'.format(n))
    p = get_random_value(mu = 0.15, sigma = 0.1, lower_bound = 0.1, upper_bound = 0.3)
    # print('n={},p={}'.format(n, p))
    big_graph = get_random_graph(n = n, p = p, need_connect = True)

    combine_graph = combine_subgraph_to_big_graph(big_graph, subgraphs)
    if (combine_graph is None):
        print('combine_graph None')
        continue
    if (judge_contain_other_subgraph(big_graph = combine_graph, class_indexes = class_indexes)):
        print('contain other')
        continue

    GT_label = get_GT_label(subgraphs = subgraphs, class_indexes = class_indexes)
    print('gt label:{}'.format(GT_label))
    if (all_generated_graphs.judge_exist(item_graph = combine_graph) == True):
        print('exist combine')
        continue
    all_generated_graphs.insert(item_graph = combine_graph)
    all_labels.append(GT_label)
    #   show_a_graph(all_generated_graphs[index])
    index_finish += 1
    print('finish:{}'.format(index_finish))
    print('-----------------\n\n\n')
    if (index_finish == NUM_GRAPHS):
        break
    # x = input()

# ---------------- key subgraph ---------------
path_keygraph = os.path.join(DIR_SAVE, 'key_subgraph_each_class.pkl')
with open(path_keygraph, 'wb') as f:
    pickle.dump(all_key_subgraphs, f)

# -------------- score info ----------
# path_score_info = os.path.join(DIR_SAVE, 'score_info.json')
# score_subgraph_each_class = [[] for i in range(3)]
# for index, item_s in enumerate(scores_each_subgraph):
#     class_index = index_keygraph_to_class[index]
#     score_subgraph_each_class[class_index].append(item_s)
# with open(path_score_info, 'w') as f:
#     json.dump(score_subgraph_each_class, f)

# -------------- all graphs ----------

all_graphs = all_generated_graphs.graphs
pairs = []
for item_graph, item_label in zip(all_graphs, all_labels):
    pairs.append([item_graph, item_label])

train_number = int(NUM_GRAPHS * 0.6)
val_number = int(NUM_GRAPHS * 0.2)

train = pairs[:train_number]
val = pairs[train_number:train_number + val_number]
test = pairs[train_number + val_number:]

train_only_graphs = [item[0] for item in train]

with open(os.path.join(DIR_SAVE, 'train.pkl'), 'wb') as f:
    pickle.dump(train, f)

with open(os.path.join(DIR_SAVE, 'val.pkl'), 'wb') as f:
    pickle.dump(val, f)

with open(os.path.join(DIR_SAVE, 'test.pkl'), 'wb') as f:
    pickle.dump(test, f)

with open(os.path.join(DIR_SAVE, 'train_unlabel.pkl'), 'wb') as f:
    pickle.dump(train_only_graphs, f)
