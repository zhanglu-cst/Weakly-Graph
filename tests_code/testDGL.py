import dgl
import torch as th
import rdkit

edges = th.tensor([2, 5, 3]), th.tensor([3, 5, 0])

e = edges[0].to('cuda')

g64 = dgl.graph(edges)
g64 = g64.to('cuda:0')
print(g64)
