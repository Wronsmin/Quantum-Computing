import networkx as nx
import numpy as np
from tqdm import tqdm
import dimod
import time


def bqm_frustration(L: int, const:  float, h: float =0.0) -> dimod.BinaryQuadraticModel:
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
    C_G = nx.grid_graph(dim=Lattice_Size, periodic=False)
    
    J1 = -1
    J2 = const * np.abs(J1)

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


def phase_transition(L: int, ratios: np.ndarray, sampler, num_reads: int =100, h: int =0.0):
    results = []
    i = 1
    for const in tqdm(ratios):
        bqm = bqm_frustration(L, const, h)
        sampleset = sampler.sample(bqm, num_reads=num_reads, 
                                   label=f'Ising Frustrated {i}/{ratios.size}') #chain_strenght=5
        i += 1
        results.append(sampleset)
        time.sleep(5)
        
    
    Magnetizations = []
    Frequencies = []
    Energies = []

    for result in results: 
        M, f, E = [], [], []
        for record in result.record:
            M_mean = np.abs(record[0].mean())
            M.append(M_mean)
            f.append(record[2])
            E.append(record[1])
        Magnetizations.append(M)
        Frequencies.append(f)
        Energies.append(E)
    
    return Magnetizations, Frequencies, Energies


def results_analysis(M: list, F: list, E: list):
    n = len(M)

    Mag, Chi, En = [], [], []

    for i in range(n):
        Magn = np.array(M[i])
        Freq = np.array(F[i])
        En = np.array(E[i])
        
        M_mean = (Magn * Freq).sum() / Freq.sum()
        En_mean = (En * Freq).sum() / Freq.sum()
        C = ((Magn - M_mean)**2 * Freq).sum() / Freq.sum()
        Mag.append(M_mean)
        Chi.append(C)
        En.append(En_mean)
    
    return Mag, Chi, En


def h_transition(L: int, ratios: np.ndarray, H: np.ndarray, sampler, num_reads=100):
    Ms, Cs, Es = [], [], []
    for h in H:
        print(f"Computing Phase Transition for B={h}")
        M, F, E = phase_transition(L, ratios, sampler, num_reads, h)
        M, C, E = results_analysis(M, F, E)

        Ms.append(M), Cs.append(C), Es.append(E)
    
    return Ms, Cs, Es