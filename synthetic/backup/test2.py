import networkx as nx

from compent.networkx_ops import show_a_graph

g = nx.fast_gnp_random_graph(n = 5, p = 1, seed = 3)
show_a_graph(g)
print(g.edges)


def remove_all_edges_among_nodes(nodes, graph):
    res = []
    for i in range(len(nodes)):
        s = nodes[i]
        for j in range(i, len(nodes)):
            e = nodes[j]
            res.append((s, e))

    graph.remove_edges_from(res)
    return graph


g = remove_all_edges_among_nodes(nodes = [0, 2, 3, 4], graph = g)
show_a_graph(g)
print(g.edges)
