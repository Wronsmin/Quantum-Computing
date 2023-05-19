import networkx as nx
import numpy as np
from tqdm import tqdm
import dimod
import time

def wall1(lattice):
    L = lattice.shape[0]
    wl=0
    for i in range (L):
        for j in range (L):
            s = lattice[i, j]
            right = lattice[(i + 1)%L, j]
            up = lattice[i, (j + 1)%L]
            if right != s:
                wl += 1
            if up != s:
                wl += 1
    return wl


def wall(lattice, n=4):
    """
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      lattice : A boolean image with values -1 or 1
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """
    
    lattice[lattice == -1] = 0  # convert -ones into zeros

    if n not in (4,8):
        raise ValueError('contour: n must be 4 or 8')
    rows,cols = lattice.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = lattice[1:,:]
    south[1:,:]  = lattice[:-1,:]
    west[:,:-1]  = lattice[:,1:]
    east[:,1:]   = lattice[:,:-1]
    idx = (north == lattice) & \
          (south == lattice) & \
          (west  == lattice) & \
          (east  == lattice)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = lattice[1:, :-1]
        north_west[:-1, :-1] = lattice[1:, 1:]
        south_east[1:, 1:]     = lattice[:-1, :-1]
        south_west[1:, :-1]   = lattice[:-1, 1:]
        idx &= (north_east == lattice) & \
               (south_east == lattice) & \
               (south_west == lattice) & \
               (north_west == lattice)

    return np.sum(~idx)


def bqm_frustration(L: int, ratio:  float, h: float =0.0) -> dimod.BinaryQuadraticModel:
    """Function used to build the model described in Park and Lee's article[1]
    
    [1] Park, Hayun and Lee, Hunpyo. "Phase transition of Frustrated Ising model 
        via D-wave Quantum Annealing Machine". Quantum Physics. 
        :DOI: '10.48550/ARXIV.2110.05124'

    Args:
        L (int): Lattice size
        const (float): constant which represents the ratio J_2/J_1

    Returns:
        dimod.BinaryQuadraticModel: model's representation on the D-Wave system
    """
    
    
    Lattice_Size = (L, L)
    C_G = nx.grid_graph(dim=Lattice_Size)
    
    J1 = -1
    J2 = ratio * np.abs(J1)

    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

    for node in C_G.nodes:
        i, j = node # i and j represents the matrix indices i-> rows j->columns
        node_name = f"{i}-{j}"
        bqm.add_variable(v=node_name, bias=h) # h=0
        

    for x in range(L-1):
        for y in range(L-1):
            # diagonals on right
            bqm.add_interaction(f"{x}-{y}", f"{x+1}-{y+1}", J2)
            
            bqm.add_interaction(f"{x + 1}-{y}", f"{x}-{y + 1}", J2)


    for edge in C_G.edges:
        node1, node2 = edge

        i, j = node1
        node1 = f"{i}-{j}"
        
        i, j = node2
        node2 = f"{i}-{j}"
        
        bqm.add_interaction(node1, node2, J1)


    return bqm


def phase_transition(L: int, ratios: np.ndarray, sampler, num_reads: int =100, h: int =0.0):
    results = []
    lattice = np.zeros((L, L))
    i = 1
   
    for const in tqdm(ratios):
        bqm = bqm_frustration(L, const, h)
        sampleset = sampler.sample(bqm, num_reads=num_reads, annealing_time=900,
                                   label=f'Ising Frustrated {i}/{ratios.size}') #chain_strenght=5
        i += 1
        
        M = []#, W = [], W1 = []
        for sample in sampleset.lowest():
            for node, value in sample.items():
                split = node.split('-')
                x, y = int(split[0]), int(split[1])
                
                lattice[x, y] = value
            
            M.append(np.abs(lattice.mean()))#, W.append(wall(config)), W1.append(wall1(config))
            
        results.append([np.mean(M), np.std(M)])
        #time.sleep(4)
        
    return results


def results_analysis(M: list, F: list, E: list):
    n = len(M)

    Mag, Chi, Ens = [], [], []

    for i in range(n):
        Magn = np.array(M[i])
        Freq = np.array(F[i])
        En = np.array(E[i])
        
        M_mean = (Magn * Freq).sum() / Freq.sum()
        En_mean = (En * Freq).sum() / Freq.sum()
        C = ((Magn - M_mean)**2 * Freq).sum() / Freq.sum()
        Mag.append(M_mean)
        Chi.append(C)
        Ens.append(En_mean)
    
    return Mag, Chi, Ens


def h_transition(L: int, ratios: np.ndarray, H: np.ndarray, sampler, num_reads=100):
    Ms, Cs, Es = [], [], []
    for h in H:
        print(f"Computing Phase Transition for B={h}")
        M, F, E = phase_transition(L, ratios, sampler, num_reads, h)
        M, C, E = results_analysis(M, F, E)

        Ms.append(M), Cs.append(C), Es.append(E)
    
    return Ms, Cs, Es