import os

import networkx

NUM_GRAPHS = 5000

DIR_SAVE = './data/syn_easy'

if (os.path.exists(DIR_SAVE) == False):
    os.makedirs(DIR_SAVE)
    print('make dir:{}'.format(DIR_SAVE))


def generate_key_subgraph_k_regular(d = 5, n_list = (6, 8, 10)):
    ans = []
    for n in n_list:
        item = networkx.random_regular_graph(d = d, n = n)
        # show_a_graph(item)
        ans.append(item)
    return ans


def generate_key_subgraph_k_star(n_list = (7, 8, 9)):
    ans = []
    for n in n_list:
        item = networkx.star_graph(n = n)
        # show_a_graph(item)
        ans.append(item)
    return ans


def generate_key_subgraph_grid(w = 2, h = (5, 6, 7)):
    ans = []
    for item_h in h:
        item = networkx.grid_graph(dim = (w, item_h))
        # show_a_graph(item)
        ans.append(item)
    return ans


keywords_class1 = generate_key_subgraph_k_regular()
keywords_class2 = generate_key_subgraph_k_star()
keywords_class3 = generate_key_subgraph_grid()

key_set = set(keywords_class3)
print(keywords_class1[0] in key_set)
print(keywords_class3[0] in key_set)

