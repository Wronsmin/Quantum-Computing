import os
import sys
sys.path.append('lib/')

import time
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from frustration_graph import phase_transition
from dwave.system import DWaveSampler, EmbeddingComposite


res_path = "Results/Ising_Frustrated/"
start = len(os.listdir(res_path + "CI/"))

L = 20
N = 100
ratios = np.linspace(0, 1, N)

qpu = DWaveSampler(profile='defaults') #'CINECA'
sampler = EmbeddingComposite(qpu)

for i in range(start, start+6):
    filename = f"CI/{i}_{ratios.size}_ratio_points.pickle"
    
    if not os.path.isfile(res_path + filename):
        Magnetizations, Frequencies, Energies = phase_transition(L, ratios, sampler, 
                                    num_reads=800)

        with open(res_path + filename, "wb") as file:
            pickle.dump([Magnetizations, Frequencies, Energies] , file)
    else:
        with open(res_path + filename, "rb") as file:
            Magnetizations, Frequencies, Energies = pickle.load(file)
    
    time.sleep(10)