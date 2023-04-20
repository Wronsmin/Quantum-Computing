import numpy as np
import pandas as pd 
import networkx as nx
from numba import jit
from itertools import product
import matplotlib.pyplot as plt
from multiprocessing import Pool

def cartesian_product(*arrays):
    return list(product(*arrays))

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
def hot_config(L):
    config = np.ones(shape=(L, L))
    config[1::2, ::2] = -1
    config[::2, 1::2] = -1
    return config.astype("int8")


def thermalization(*param): #L, T, J, H=0, err_runs=1
    # L is the length of the lattice
    index = param[0]
    L, T, ratio, H, err_runs = param[1]
    
    print(param[1])

    J = np.array([1, ratio])

    # number of temperature points
    eqSteps = 1000
    mcSteps = 2000
    
    config = hot_config(L)

    # initialize total energy and mag
    beta = 1. / T
    # evolve the system to equilibrium
    for i in range(eqSteps):
        MC_step(config, beta, J, H)
    
    lattices = []
    for j in range(err_runs):
        configs = []
        for i in range(mcSteps):
            MC_step(config, beta, J, H)
            configs.append(np.copy(config))

        lattices.append(configs)
        
    configs = np.stack(lattices).astype(np.int8)
    
    np.save(f"../../Results/Ising_Frustrated/Classical/Configs/{L}x{L}/Lattices/param_{index}.npy", configs)
            
    return None


def transition(L, Ts, ratios, err_runs=1, workers=1, H=0):
    
    params = cartesian_product([L], Ts, ratios, [H], [err_runs])
    
    np.save(f"../../Results/Ising_Frustrated/Classical/Configs/{L}x{L}/params.npy", params)
    
    params_indexed = []
    for i, param in enumerate(params):
        params_indexed.append([i, param])
        
    pool = Pool(processes=workers)
    res = pool.starmap(thermalization, params_indexed)
    pool.close()