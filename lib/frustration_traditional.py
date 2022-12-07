import numpy as np
import pandas as pd 
from numba import jit
from tqdm import tqdm
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count


def dstack_product(x, y):
    return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

@jit
def MC_step(config, beta, J, H):
    '''
    Monte Carlo move using Metropolis algorithm
    '''
    L = config.shape[0]
    rng = np.random.rand(L, L)
    J1, J2 = J
    for i in range(L):
        for j in range(L):
            s = config[i, j]
            
            # considering nearest neighbors
            n_nb = config[(i + 1) % L,   j] + \
                    config[          i,  (j + 1) % L] + \
                    config[(i - 1) % L,  j] + \
                    config[          i,  (j - 1) % L]
            
            # considering next nearest neighbors  
            nn_nb = config[(i + 1) % L, (j + 1) % L] + \
                    config[(i - 1) % L,  (j + 1) % L] + \
                    config[(i + 1) % L,  (j - 1) % L] + \
                    config[(i - 1) % L,  (j - 1) % L]
                    
            del_E = 2 * s * (J1 * n_nb - J2 * nn_nb +  H)
            
            if (del_E < 0) or (rng[i, j] < np.exp(-del_E * beta)):
                config[i, j] = -s


@jit   
def E_dimensionless(config, J, H):
    total_energy = 0
    L = config.shape[0]
    J1, J2 = J
    for i in range(L):
        for j in range(L):
            S = config[i, j]
            n_nb = config[(i + 1) % L, j] + \
                    config[i, (j + 1) % L] + \
                    config[(i - 1) % L, j] + \
                    config[i, (j - 1) % L]
                    
            # considering next nearest neighbors  
            nn_nb = config[(i + 1) % L, (j + 1) % L] + \
                    config[(i - 1) % L,  (j + 1) % L] + \
                    config[(i + 1) % L,  (j - 1) % L] + \
                    config[(i - 1) % L,  (j - 1) % L]

            total_energy += - S * (J1 * n_nb - J2 * nn_nb +  H)
    return total_energy / 2.



@dataclass
class Ising():
    L: int
    
    def hot_config(self):
        return np.tile([[-1, 1], [1, -1]], 
                       (self.L, self.L))[:self.L, :self.L]
                    
    
    def thermalization(self, T, J, H=0, err_runs=1):
        # L is the length of the lattice

        # number of temperature points
        eqSteps = 100
        mcSteps = 1000

        # initialization of all variables
    
        E, E_std = 0, 0
        M, M_std = 0, 0
        C, C_std = 0, 0
        X, X_std = 0, 0
        
        config = self.hot_config()

        # initialize total energy and mag
        beta = 1. / T
        # evolve the system to equilibrium
        for i in range(eqSteps):
            MC_step(config, beta, J, H)
        # list of ten macroscopic properties
        Ez = []
        Cz = []
        Mz = []
        Xz = []

        for j in range(err_runs):
            E = np.zeros(mcSteps)
            M = np.zeros(mcSteps)
            for i in range(mcSteps):
                MC_step(config, beta, J, H)
                E[i] = E_dimensionless(config, J, H)  # calculate the energy at time stamp
                M[i] = abs(np.mean(config))  # calculate the abs total mag. at time stamp


            # calculate macroscopic properties (divide by # sites) and append
            Energy = E.mean() / self.L ** 2
            SpecificHeat = beta ** 2 * E.var() / self.L**2
            Magnetization = M.mean()
            Susceptibility = beta * M.var() * (self.L ** 2)

            Ez.append(Energy)
            Cz.append(SpecificHeat)
            Mz.append(Magnetization)
            Xz.append(Susceptibility)

        E = np.mean(np.array(Ez))
        E_std = np.std(np.array(Ez))

        M = np.mean(np.array(Mz))
        M_std = np.std(np.array(Mz))

        C = np.mean(np.array(Cz))
        C_std = np.std(np.array(Cz))

        X = np.mean(np.array(Xz))
        X_std = np.std(np.array(Xz))
        
        return np.array([T, np.divide(*J[::-1]), 
                         E, E_std, 
                         M, M_std, 
                         C, C_std, 
                         X, X_std]), config
    
    def transition(self, Ts, ratios, H=0, err_runs=1):
        params = dstack_product(Ts, ratios)
        
        columns = ['T', 'ratio', 'E', 'E_std', 'M', 'M_std', 'C', 'C_std', 'X', 'X_std']
        res = []
        
        for param in tqdm(params):
            T, ratio = param
            J = np.array([1, ratio])
            
            res.append(self.thermalization(T, J, H, err_runs)[0])
        
        return pd.DataFrame(res, columns=columns)