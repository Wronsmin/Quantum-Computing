import dwave_networkx as dnx
import matplotlib.pyplot as plt 

G = dnx.pegasus_graph(4)

dnx.draw_pegasus(G, crosses=True, node_size=20)

plt.show()