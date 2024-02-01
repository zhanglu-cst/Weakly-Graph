import os

import mmcv

from compent.global_var import Global_Var
from compent.networkx_ops import Graphs_Set, Graph_Dict


class Enumerate_Subgraphs_NetworkX():
    def __init__(self, min_node_number = 5, max_node_number = 15, max_return_subgraph_number_each_graph = None):
        super(Enumerate_Subgraphs_NetworkX, self).__init__()
        self.exist_subgraph = {}
        self.min_node_number = min_node_number
        self.max_node_number = max_node_number
        self.max_return_subgraph_number = max_return_subgraph_number_each_graph
        data_root_dir = Global_Var().get('cfg').root_dir
        dump_filename = os.path.join(data_root_dir, 'dump_enumrate_subgraphs.pkl')
        self.dump_filename = dump_filename
        if (os.path.exists(dump_filename)):
            self.temp_dict = mmcv.load(dump_filename)
        else:
            self.temp_dict = Graph_Dict()

    def clear(self):
        self.exist_subgraph = {}

    def add_to_exist(self, cur_nodes):
        if (len(cur_nodes) < self.min_node_number or len(cur_nodes) > self.max_node_number):
            return
        s_nodes = sorted(cur_nodes)
        str_nodes = '_'.join([str(item) for item in s_nodes])
        self.exist_subgraph[str_nodes] = s_nodes

    def dfs(self, cur_nodes):
        if (len(cur_nodes) > self.max_node_number):
            return
        if (self.max_return_subgraph_number is not None and len(self.exist_subgraph) > self.max_return_subgraph_number):
            return
        self.add_to_exist(cur_nodes)
        for item_node in self.all_nodes:
            if (item_node > cur_nodes[-1]):
                connected = False
                for neighbor in self.graph[item_node]:
                    if (neighbor in cur_nodes):
                        connected = True
                        break
                if (connected):
                    cur_nodes.append(item_node)
                    self.dfs(cur_nodes)
                    cur_nodes.pop()

    def dump_to_temp(self, graph_networkx, all_subgraphs):
        self.temp_dict[graph_networkx] = all_subgraphs
        mmcv.dump(self.temp_dict, file = self.dump_filename)

    def __call__(self, graph_networkx):
        if (graph_networkx in self.temp_dict):
            # print('exist, jump')
            return self.temp_dict[graph_networkx]
        else:
            self.clear()
            self.graph = graph_networkx
            self.all_nodes = graph_networkx.nodes()
            # print('number nodes:{}'.format(len(self.all_nodes)))
            for start_node in self.all_nodes:
                self.dfs(cur_nodes = [start_node])
            all_subgraph_nodes = list(self.exist_subgraph.values())
            all_subgraphs = Graphs_Set()
            for item_node_list in all_subgraph_nodes:
                item_subgraph = graph_networkx.subgraph(item_node_list)
                all_subgraphs.insert_set(item_graph = item_subgraph)
            self.dump_to_temp(graph_networkx, all_subgraphs.graphs)
            return all_subgraphs.graphs


if (__name__ == '__main__'):
    # from compent.networkx_ops import show_a_graph
    import networkx

    subgraph_enumrate = Enumerate_Subgraphs_NetworkX(max_return_subgraph_number_each_graph = 1000)
    g = networkx.star_graph(n = 100)
    all_subs = subgraph_enumrate(g)
    print(len(all_subs))
    # show_a_graph(g)
    # for index, item in enumerate(all_subs):
    #     show_a_graph(item)
    # if (index == 10):
    #     break
