import networkx

from compent.networkx_ops import show_a_graph


def generate_key_subgraph_k_regular(d_list = (4, 5), n_list = (4, 5, 6, 7, 8)):
    ans = []
    for d in d_list:
        for n in n_list:
            if (n * d % 2 == 0 and n > d):
                item = networkx.random_regular_graph(d = d, n = n)
                # show_a_graph(item)
                ans.append(item)
    print(len(ans))
    return ans


generate_key_subgraph_k_regular()
