import json
import os
import pickle
import random

import numpy

from compent.networkx_ops import Graphs_Set, show_a_graph, get_random_graph
from compent.utils import get_random_value

NUM_GRAPHS = 8000
NUM_CLASSES = 3
NUM_KEYSUB_EACH_CLASS = 3
DIR_SAVE = './data/syn_class3_key3_final_5'

if (os.path.exists(DIR_SAVE) == False):
    os.makedirs(DIR_SAVE)
    print('make dir:{}'.format(DIR_SAVE))

total_number_subgraphs = NUM_CLASSES * NUM_KEYSUB_EACH_CLASS

all_keygraph_set = Graphs_Set()
print('genenrating key subgraph...')
# for i in range(total_number_subgraphs):
while True:
    n = get_random_value(mu = 7, sigma = 6, lower_bound = 5, upper_bound = 15)
    n = int(n)
    p = get_random_value(mu = 0.5, sigma = 1, lower_bound = 0.3, upper_bound = 0.99)
    # if (n * p > 10):
    #     continue
    print('key subgraph, n:{}, p:{}'.format(n, p))
    all_keygraph_set.insert_a_random_graph(n = n, p = p, need_connect = True)
    show_a_graph(all_keygraph_set[-1])
    if (len(all_keygraph_set) == total_number_subgraphs):
        break

keygraph_each_class = [[] for i in range(NUM_CLASSES)]
index_keygraph_each_class = [[] for i in range(NUM_CLASSES)]
index_keygraph_to_class = {}
for i in range(total_number_subgraphs):
    class_index = i // NUM_KEYSUB_EACH_CLASS
    keygraph_each_class[class_index].append(all_keygraph_set[i])
    index_keygraph_each_class[class_index].append(i)
    index_keygraph_to_class[i] = class_index

scores_each_subgraph = []
for i in range(total_number_subgraphs):
    # item_score = random.randint(1, 5)
    node_number = all_keygraph_set[i].number_of_nodes()
    item_score = node_number
    scores_each_subgraph.append(item_score)
print('scores_each_subgraph:{}'.format(scores_each_subgraph))


# for keygraph_cur_class in keygraph_each_class:
#     for item in keygraph_cur_class:
#         show_a_graph(item)

# number = get_random_value(mu = 1, sigma = 1, lower_bound = 1, upper_bound = 3)

# number = int(number)
# choose = random.sample(population = list(range(total_number_subgraphs)), k = number)

def pick_random_keygraph_index():
    number_class = get_random_value(mu = 1, sigma = 0.6, lower_bound = 1, upper_bound = 2.99)
    number_class = int(number_class)
    class_indexes = random.sample(population = list(range(NUM_CLASSES)), k = number_class)
    if (number_class == 1):
        number_keysubgraph = get_random_value(mu = 2, sigma = 0.6, lower_bound = 1,
                                              upper_bound = NUM_KEYSUB_EACH_CLASS + 1)
        number_keysubgraph = int(number_keysubgraph)
        sample_index = random.sample(population = index_keygraph_each_class[class_indexes[0]], k = number_keysubgraph)
    else:
        sample_index = []
        for item_class_index in class_indexes:
            sample_item = random.sample(population = index_keygraph_each_class[item_class_index], k = 1)
            sample_index += sample_item

    print('number_class:{},class_indexes:{},sample_index:{}'.format(number_class, class_indexes, sample_index))
    return sample_index


def remove_all_edges_among_nodes(nodes, graph):
    res = []
    for i in range(len(nodes)):
        s = nodes[i]
        for j in range(i, len(nodes)):
            e = nodes[j]
            res.append((s, e))

    graph.remove_edges_from(res)
    return graph


def combine_subgraph_to_big_graph(big_graph, indexs_subgraph):
    # show_a_graph(big_graph)
    total_number_nodes = big_graph.number_of_nodes()
    number_nodes_each_subgraph = []
    for item_index in indexs_subgraph:
        subgraph = all_keygraph_set[item_index]
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
    for picked_nodes, index_sub in zip(picked_nodes_each_subgraph, indexs_subgraph):
        # print('picked_nodes:{}'.format(picked_nodes))
        subgraph = all_keygraph_set[index_sub]
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


def get_GT_label(indexs_subgraph):
    score_each_class = numpy.zeros(NUM_CLASSES)
    for item_index in indexs_subgraph:
        class_index = index_keygraph_to_class[item_index]
        score = scores_each_subgraph[item_index]
        score_each_class[class_index] += score
    class_res = numpy.argmax(score_each_class)
    return class_res


all_generated_graphs = Graphs_Set()
all_labels = []
index_finish = 0
while True:
    n = get_random_value(mu = 25, sigma = 3, lower_bound = 8, upper_bound = 30)
    n = int(n)
    p = get_random_value(mu = 0.1, sigma = 0.1, lower_bound = 0.1, upper_bound = 0.3)
    print('n={},p={}'.format(n, p))
    big_graph = get_random_graph(n = n, p = p, need_connect = True)
    indexs_subgraph = pick_random_keygraph_index()
    # print('picked indexs_subgraph:{}'.format(indexs_subgraph))
    combine_graph = combine_subgraph_to_big_graph(big_graph, indexs_subgraph)
    if (combine_graph is None):
        continue
    GT_label = get_GT_label(indexs_subgraph = indexs_subgraph)

    if (all_generated_graphs.judge_exist(item_graph = combine_graph) == True):
        continue
    all_generated_graphs.insert(item_graph = combine_graph)
    all_labels.append(GT_label)
    #   show_a_graph(all_generated_graphs[index])
    index_finish += 1
    print('finish:{}'.format(index_finish))
    if (index_finish == NUM_GRAPHS):
        break

# ---------------- key subgraph
path_keygraph = os.path.join(DIR_SAVE, 'key_subgraph_each_class.pkl')
with open(path_keygraph, 'wb') as f:
    pickle.dump(keygraph_each_class, f)

# -------------- score info ----------
path_score_info = os.path.join(DIR_SAVE, 'score_info.json')
score_subgraph_each_class = [[] for i in range(NUM_CLASSES)]
for index, item_s in enumerate(scores_each_subgraph):
    class_index = index_keygraph_to_class[index]
    score_subgraph_each_class[class_index].append(item_s)
with open(path_score_info, 'w') as f:
    json.dump(score_subgraph_each_class, f)

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

# keygraph_each_class
# all_generated_graphs.graphs
# all_labels
