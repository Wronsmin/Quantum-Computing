import networkx as nx
import numpy as np
from tqdm import tqdm
import dimod
import time
import graph_tool.all as gt 
import matplotlib.pyplot as plt


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


class FIM_cpu():
    def __init__(self, L: int, T: float, ratios: np.ndarray, 
                 H: float, periodic: bool=False, num_sim=100) -> None:
        self.L = L
        self.T = T
        self.ratios = ratios
        self.H = H
        self.periodic = periodic
        
        self.num_sim = num_sim
        self.epochs = 2000
        
        
    def _graph_building(self, ratio, periodic):
        g = gt.lattice([self.L, self.L], periodic=periodic)
        
        J1 = 1
        J2 = -(ratio * J1)
        
        edges = [(*edge, J1) for edge in g.get_edges().tolist()]

        indices = g.get_vertices().reshape([self.L, self.L])
        for x in range(self.L-1):
            for y in range(self.L-1):
                # diagonals on right
                edges.append((*indices[[x, x+1], [y, y+1]], J2))
                edges.append((*indices[[x+1, x], [y, y+1]], J2))
        
        
        g = gt.Graph(directed=False)
        self.weights = g.new_ep("double")

        g.add_edge_list(edges, eprops=[self.weights])
        
        pos = []
        for x in range(self.L):
            for y in range(self.L):
                pos.append([x, y])
                
        pos = np.array(pos)
        self.pos = g.new_vertex_property("vector<double>", pos)
        
        return g
    
    
    def sim_run(self):
        
        self.M = np.zeros([self.num_sim, len(self.ratios)])
        self.X = np.zeros([self.num_sim, len(self.ratios)])
        
    
        for ratio in tqdm(self.ratios): # transition loop
            self.G = self._graph_building(ratio, self.periodic)
            
            num_nodes = self.G.num_vertices()
            Mag = np.zeros(self.num_sim)
            X = np.zeros(self.num_sim)
            
            for i in range(self.num_sim):
                M = []
                state = gt.IsingMetropolisState(self.G, beta=1/self.T, w=self.weights)
                state.iterate_async(niter=100 * num_nodes)
                for e in range(self.epochs):
                    state.iterate_async(niter=num_nodes)
                    M.append(np.abs(state.s.fa.mean()))
                
                Mag[i] = np.mean(M)
                X[i] = np.var(M)
            
            idx = np.where(self.ratios == ratio)[0][0]
            
            self.M[:, idx] = Mag[:]
            self.X[:, idx] = X[:] 
            
    def plot_results(self):
        mean = self.M.mean(axis=0)
        std = self.M.std(axis=0)
        
        plt.plot(self.ratios, mean, label='B=0')
        plt.fill_between(self.ratios, mean - std, mean + std, alpha=0.5)
        plt.grid()
        plt.xlabel(r'$J_2$/$J_1$', fontsize=18)
        plt.ylabel('|M| [arb. unit]', fontsize=18)
        plt.tight_layout()
        plt.legend()
        
        mean = self.X.mean(axis=0)
        std = self.X.std(axis=0)
        
        plt.plot(self.ratios, mean, label='B=0')
        plt.fill_between(self.ratios, mean - std, mean + std, alpha=0.5)
        plt.grid()
        plt.xlabel(r'$J_2$/$J_1$', fontsize=18)
        plt.ylabel('X [arb. unit]', fontsize=18)
        plt.tight_layout()
        plt.legend()
                