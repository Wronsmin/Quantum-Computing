import numpy as np
import pandas as pd 
from numba import jit
from itertools import product
from multiprocessing import Pool, cpu_count
from scipy.stats import kurtosis


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

@jit
def hot_config(L):
    config = np.ones(shape=(L, L))
    config[1::2, ::2] = -1
    config[::2, 1::2] = -1
    return config
                

def thermalization(*param): #L, T, J, H=0, err_runs=1
    # L is the length of the lattice
    L, T, ratio, H, err_runs = param

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
    # list of ten macroscopic properties
    #Ez = np.zeros(err_runs)
    #Cz = np.zeros(err_runs)
    Mz = np.zeros(err_runs)
    Xz = np.zeros(err_runs)
    Uz = np.zeros(err_runs)

    for j in range(err_runs):
        #E = np.zeros(mcSteps)
        M = np.zeros(mcSteps)
        for i in range(mcSteps):
            MC_step(config, beta, J, H)
            #E[i] = E_dimensionless(config, J, H)  # calculate the energy at time stamp
            M[i] = abs(np.mean(config))  # calculate the abs total mag. at time stamp


        # calculate macroscopic properties (divide by # sites) and append
        #Energy = E.mean() / L ** 2
        #SpecificHeat = beta ** 2 * E.var() / L**2
        Magnetization = M.mean()
        Susceptibility = beta * M.var() * (L ** 2)
        

        #Ez[j] = Energy
        #Cz[j] = SpecificHeat
        Mz[j] = Magnetization
        Xz[j] = Susceptibility
        m2 = np.mean(M ** 2)
        if m2 != 0:
            Uz[j] = 1 - (np.mean(M ** 4)) / (3 * m2 ** 2)
        else: Uz[j] = -1000

    #E = Ez.mean()
    #E_std = Ez.std()

    M = Mz.mean()
    M_std = Mz.std()

    #C = Cz.mean()
    #C_std = Cz.std()

    X = Xz.mean()
    X_std = Xz.std()
    
    #[T, ratio, E, E_std, M, M_std, C, C_std, X, X_std, Uz.mean()]
    return [T, ratio, M, M_std, X, X_std, Uz.mean()]#, config


def transition(L, Ts, ratios, err_runs=1, workers=1, H=0):
    
    params = cartesian_product([L], Ts, ratios, [H], [err_runs])
    # ['T', 'ratio', 'E', 'E_std', 'M', 'M_std', 'C', 'C_std', 'X', 'X_std', 'U']
    columns = ['T', 'ratio', 'M', 'M_std', 'X', 'X_std', 'U']
    
    pool = Pool(processes=workers)
    res = pool.starmap(thermalization, params)
    pool.close()
    
    res = np.array(res)
    
    return pd.DataFrame(res, columns=columns)
