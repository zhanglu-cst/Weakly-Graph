from compent.networkx_ops import show_a_graph
import networkx

wheel_graph = networkx.ring_of_cliques(2,4)
show_a_graph(wheel_graph)

