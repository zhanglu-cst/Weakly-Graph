import random

import dgl
import networkx as nx
import torch
from func_timeout import func_set_timeout
from matplotlib import pyplot as plt
from networkx.algorithms import isomorphism


def show_a_graph(G):
    nx.draw_networkx(G, with_labels = True)
    plt.show()


def get_random_graph(n, p, need_connect = True):
    if (need_connect):
        try_times = 0
        while True:
            graph = nx.fast_gnp_random_graph(n = n, p = p)
            conn = nx.is_connected(graph)
            if (conn):
                break
            try_times += 1
            if (try_times == 2000):
                raise Exception('max try times')
        # print('try_times:{}'.format(try_times))
        return graph
    else:
        graph = nx.fast_gnp_random_graph(n = n, p = p)
        return graph


def judge_is_isomorphic(G1, G2):
    return nx.is_isomorphic(G1, G2)


@func_set_timeout(10)
def do_judge_contain_a_subgraph(big_graph, subgraph):
    GM = isomorphism.GraphMatcher(big_graph, subgraph)
    return GM.subgraph_is_isomorphic()


def judge_contain_a_subgraph(big_graph, subgraph):
    try:
        res = do_judge_contain_a_subgraph(big_graph, subgraph)
        return res
    except:
        return False


def networkx_to_DGL(graph_networkX, feature_value = 0.1, feature_dim = 10):
    graph_DGL = dgl.from_networkx(nx_graph = graph_networkX)
    num_nodes = graph_networkX.number_of_nodes()
    num_edges = graph_networkX.number_of_edges() * 2
    graph_DGL.ndata['x'] = torch.full([num_nodes, feature_dim], feature_value)
    graph_DGL.edata['x'] = torch.full([num_edges, feature_dim], feature_value)
    return graph_DGL


def batch_networkx_to_DGL(batch_graphs_networkX, feature_value = 0.1, feature_dim = 10):
    res = []
    for item in batch_graphs_networkX:
        item_res = networkx_to_DGL(item, feature_value, feature_dim)
        res.append(item_res)
    return res


# def networkx_to_DGL(graph_networkX, feature_value = 0.1, feature_dim = 15):
#     graph_DGL = dgl.from_networkx(nx_graph = graph_networkX)
#     num_nodes = graph_networkX.number_of_nodes()
#     num_edges = graph_networkX.number_of_edges() * 2
#     degree_list = graph_networkX.degree
#     degree_list = [item[1] for item in degree_list]
#     feature_node = torch.zeros(num_nodes, feature_dim)
#     for index, item_index in enumerate(degree_list):
#         if (item_index >= feature_dim):
#             item_index = feature_dim - 1
#         feature_node[index][item_index] = 1
#     # print(degree_list)
#     # print(feature_node)
#     graph_DGL.ndata['x'] = feature_node
#     graph_DGL.edata['x'] = torch.full([num_edges, feature_dim], feature_value)
#     return graph_DGL


class Graphs_Set():
    def __init__(self):
        super(Graphs_Set, self).__init__()
        self.graphs = []

    def judge_exist(self, item_graph):
        for item_exist in self.graphs:
            if (judge_is_isomorphic(item_exist, item_graph)):
                return True
        return False

    def insert_set(self, item_graph):
        if (self.judge_exist(item_graph) == False):
            self.insert(item_graph)

    def insert(self, item_graph):
        self.graphs.append(item_graph)

    def insert_a_random_graph(self, n, p, need_connect):
        try_time = 0
        while True:
            g = get_random_graph(n, p, need_connect = need_connect)
            if (self.judge_exist(g) == False):
                self.insert(item_graph = g)
                break
            try_time += 1
            if (try_time == 1000):
                raise Exception('max try time')
        # print('try_time:{}, len:{}'.format(try_time, self.__len__()))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, item):
        return self.graphs[item]


class Graph_Dict():
    def __init__(self):
        super(Graph_Dict, self).__init__()
        self.graph_map = {}

    def __find_a_graph__(self, key):
        for item_graph in self.graph_map:
            if (judge_is_isomorphic(key, item_graph)):
                return item_graph
        return None

    def __contains__(self, graph):
        if (self.__find_a_graph__(graph) is not None):
            return True
        else:
            return False

    def __setitem__(self, key, value):
        for item_graph in self.graph_map:
            if (judge_is_isomorphic(key, item_graph)):
                self.graph_map[item_graph] = value
                return
        self.graph_map[key] = value

    def __getitem__(self, item):
        for item_graph in self.graph_map:
            if (judge_is_isomorphic(item, item_graph)):
                return self.graph_map[item_graph]
        raise KeyError('graph not in dict')

    def __delitem__(self, key):
        inner_graph = self.__find_a_graph__(key)
        if (inner_graph is not None):
            del self.graph_map[inner_graph]
        else:
            raise Exception('graph not in dict')

    def __len__(self):
        return len(self.graph_map)

    def __iter__(self):
        return iter(self.graph_map.items())


def remove_all_edges_among_nodes(nodes, graph):
    res = []
    for i in range(len(nodes)):
        s = nodes[i]
        for j in range(i, len(nodes)):
            e = nodes[j]
            res.append((s, e))

    graph.remove_edges_from(res)
    return graph


def combine_subgraph_to_big_graph(big_graph, subgraphs, return_pick_nodes = False):
    total_number_nodes = big_graph.number_of_nodes()
    number_nodes_each_subgraph = []
    for subgraph in subgraphs:
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

        remove_all_edges_among_nodes(nodes = picked_nodes, graph = big_graph)
        origin_edges = subgraph.edges
        map_origin_id_to_picked = {}
        origin_node_ids = subgraph.nodes
        assert len(origin_node_ids) == len(picked_nodes)
        for item_origin, item_picked in zip(origin_node_ids, picked_nodes):
            map_origin_id_to_picked[item_origin] = item_picked
        transfer_edges = []

        for item_edge in origin_edges:
            s, e = item_edge
            ns = map_origin_id_to_picked[s]
            ne = map_origin_id_to_picked[e]
            n_edge = [ns, ne]
            transfer_edges.append(n_edge)
        big_graph.add_edges_from(transfer_edges)
    if (return_pick_nodes):
        return big_graph, picked_nodes_each_subgraph
    else:
        return big_graph


def filter_too_large_graphs(graph_label_list, filter_radio = 0.01):
    temp = []
    for item in graph_label_list:
        number_nodes = item[0].number_of_nodes()
        item = list(item)
        item.append(number_nodes)
        temp.append(item)
    temp = sorted(temp, key = lambda x: x[-1], reverse = True)
    print('cur graph set, max number node:{}'.format(temp[0][-1]))
    filter_number = int(len(temp) * filter_radio)
    temp = temp[filter_number:]
    print('filter graphs number:{}, keep:{}, cur max number node:{}'.format(filter_number, len(temp), temp[0][-1]))
    ans = []
    for item in temp:
        del item[-1]
        ans.append(item)
    return ans


if __name__ == '__main__':
    graph_dict = Graph_Dict()
    item = nx.star_graph(10)
    graph_dict[item] = 10
    i2 = nx.star_graph(10)
    print(graph_dict[i2])
    i3 = nx.star_graph(10)
    graph_dict[i3] = 1000
    print(graph_dict[item])
    print(len(graph_dict))
    # del graph_dict[item]
    # print(len(graph_dict))
    print(i2 in graph_dict)
    for item in graph_dict:
        print(item)
        # print(graph_dict[item])
