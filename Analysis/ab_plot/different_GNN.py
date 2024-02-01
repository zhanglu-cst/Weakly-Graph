import matplotlib.pyplot as plt

x = list(range(1, 11))
GIN = [0.81, 0.915, 0.95, 0.94, 0.955, 0.945, 0.955, 0.945, 0.95, 0.955]
GCN = [0.78, 0.884, 0.92, 0.90, 0.89, 0.92, 0.91, 0.92, 0.901, 0.91]
GraphSAGE = [0.80, 0.92, 0.94, 0.93, 0.92, 0.945, 0.931, 0.922, 0.935, 0.939]
GAT = [0.79, 0.905, 0.911, 0.925, 0.915, 0.895, 0.925, 0.901, 0.899, 0.919]

plt.figure(dpi = 600, figsize = (8, 4))

p1 = plt.plot(x, GIN, marker = 'o')
p2 = plt.plot(x, GCN, marker = 'v')
p3 = plt.plot(x, GraphSAGE, marker = '^')
p4 = plt.plot(x, GAT, marker = '+')
plt.xticks(list(range(1, 11)))
plt.legend([p1[0], p2[0], p3[0], p4[0]], ['GIN', 'GCN', 'GraphSAGE', 'GAT'], loc = 'best')
plt.ylabel('Classifier Performance')
plt.xlabel('Iterations')
# plt.show()

plt.savefig('different_GNN.eps', dpi = 600, format = 'eps')
