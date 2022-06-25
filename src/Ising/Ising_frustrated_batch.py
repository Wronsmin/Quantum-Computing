import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('lib/')
from frustration_graph import phase_transition
import pickle
import os
import copy
import time
from dwave.system import DWaveSampler, EmbeddingComposite

res_path = "Results/Ising_Frustrated/"
start = len(os.listdir(res_path + "CI/"))

L = 20
ratios = np.concatenate((np.arange(0, 0.3, 0.01), 
                         np.arange(0.3, 0.7, 0.02), 
                         np.arange(0.7, 1, 0.1)))

qpu = DWaveSampler(profile='CINECA') #'defaults'
sampler = EmbeddingComposite(qpu)

for i in range(start, start+6):
    filename = f"CI/{i}_{ratios.size}_ratio_points.pickle"

    if not os.path.isfile(res_path + filename):
        Magnetizations, Frequencies, Energies = phase_transition(L, ratios, sampler, 
                                    num_reads=1000)

        with open(res_path + filename, "wb") as file:
            pickle.dump([Magnetizations, Frequencies, Energies] , file)
    else:
        with open(res_path + filename, "rb") as file:
            Magnetizations, Frequencies, Energies = pickle.load(file)