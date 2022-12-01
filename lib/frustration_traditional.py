import numpy as np
from dataclasses import dataclass
from numpy.random import rand
import time
from numba import jit, prange
from multiprocessing import Pool, cpu_count
import pandas as pd 
from tqdm import tqdm

@dataclass
def Ising():
    L: int
    
    def hot_config(self):
        return np.tile([[-1, 1], [1, -1]], 
                       (self.N // 2, self.N // 2))
    
    @jit
    def MC_step(self, config, beta, J, H):
        '''
        Monte Carlo move using Metropolis algorithm
        '''
        rng = np.random.rand(self.L, self.L)
        for i in range(self.L):
            for j in range(self.L):
                s = config[i, j]
                neighbors = config[(i + 1) % L, j] + config[i, (j + 1) % L] + config[(i - 1) % L, j] + config[
                    i, (j - 1) % L]
                del_E = 2 * J * s * neighbors + 2 * H * s
                
                if (del_E < 0) or (rng[i, j] < np.exp(-del_E * beta)):
                    config[i, j] = -s

