import networkx as nx
import matplotlib.pyplot as plt
from utils import *
import numpy as np 

mu0 = 1
var0 = 0.5 

x = np.random.normal(mu0, var0, 10)

N_qbits = 9
Nodes = 9 * 2

C_G = nx.complete_graph(Nodes)
W_G = nx.Graph()

Nodes_category = {a: b for a, b in zip(range(Nodes), ['a'] * (Nodes//2) + ['b'] * (Nodes//2))}
Nodes_labels = {a: b for a, b in zip(range(Nodes), list(range(-7, 2)) * 2)}

"""
nx.draw(G)
plt.show()
"""


        


for edge in C_G.edges:
    node1, node2 = edge
    if Nodes_category[node1] == Nodes_category[node2] == 'a':
        pi = Nodes_labels[node1]
        pj = Nodes_labels[node2]
        w = - 2**(pi+pj)*fAA(var0)
        
        W_G.add_edge(node1, node2, bij=w)
    elif Nodes_category[node1] == Nodes_category[node2] == 'b':
        pi = Nodes_labels[node1]
        pj = Nodes_labels[node2]
        w = - np.sum(2**(pi+pj)*fBB(x, mu0, var0))
        
        W_G.add_edge(node1, node2, bij=w)
    else:
        pi = Nodes_labels[node1]
        pj = Nodes_labels[node2]
        w = - np.sum(2**(pi+pj)*fAB(x, mu0, var0))
        
        W_G.add_edge(node1, node2, bij=w)


node_weights = np.zeros(W_G.number_of_nodes())
for node in C_G.nodes:
    if Nodes_category[node] == 'a':
        p = Nodes_labels[node]
        w = -np.sum(2**p * fA(x, mu0, var0) + (2**(2*p - 1) - 2**p*mu0)*fAA(var0) - 2**p*var0*fAB(x, mu0, var0))
        
        node_weights[node] = w
    else:
        p = Nodes_labels[node]
        w = -np.sum(2**p * fB(x, mu0, var0) + (2**(2*p - 1) - 2**p*var0)*fBB(x, mu0, var0) - 2**p*var0*fAB(x, mu0, var0))
        
        node_weights[node] = w

nx.set_node_attributes(W_G, {a:{'ai':b} for a, b in zip(C_G.nodes, node_weights)})


