import time

import networkx

from compent.networkx_ops import judge_contain_a_subgraph

g = networkx.star_graph(n = 20)
sub = networkx.fast_gnp_random_graph(n = 10, p = 0.1)
s = time.time()
for i in range(1000):
    res = judge_contain_a_subgraph(g, sub)
e = time.time()
delta = e - s
avg = delta / 1000
print(avg)
