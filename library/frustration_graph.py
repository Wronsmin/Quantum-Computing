import networkx as nx
import dimod

def bqm_frustration(L, const):
    Lattice_Size = (L, L)
    C_G = nx.grid_graph(dim=Lattice_Size, periodic=False)
    
    h = 0.
    J1 = -1
    J2 = const * J1

    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

    for node in C_G.nodes:
        i, j = node # i and j represents the matrix indices i-> rows j->columns
        node_name = f"{i}-{j}"
        bqm.add_variable(v=node_name, bias=h)
        

    for x in range(1, L-1):
        for y in range(1, L-1):
            # diagonals on right
            x1, y1 = x+1, y+1
            bqm.add_interaction(f"{x}-{y}", f"{x1}-{y1}", J2)
            
            x1, y1 = x-1, y+1
            bqm.add_interaction(f"{x}-{y}", f"{x1}-{y1}", J2)
            
            # diagonals on left
            x1, y1 = x+1, y-1
            bqm.add_interaction(f"{x}-{y}", f"{x1}-{y1}", J2)
            
            x1, y1 = x-1, y-1
            bqm.add_interaction(f"{x}-{y}", f"{x1}-{y1}", J2)


    for edge in C_G.edges:
        node1, node2 = edge
        i, j = node1
        node1 = f"{i}-{j}"
        
        i, j = node2
        node2 = f"{i}-{j}"
        bqm.add_interaction(node1, node2, J1)
    
    return bqm
