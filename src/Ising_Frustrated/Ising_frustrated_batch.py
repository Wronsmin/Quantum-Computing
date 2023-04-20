import os
import sys
sys.path.append('lib/')

import time
import dimod
import pickle
import minorminer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from frustration_graph import *
from dwave.system import DWaveSampler, FixedEmbeddingComposite


res_path = "Results/Ising_Frustrated/"
start = len(os.listdir(res_path + "CI/"))

L = 20
N = 100
ratios = np.linspace(0, 1, N)

qpu = DWaveSampler(profile='defaults') #'CINECA'
bqm = bqm_frustration(L, 1, 0)
emb = minorminer.find_embedding(dimod.to_networkx_graph(bqm), qpu.to_networkx_graph(), threads=12)
sampler = FixedEmbeddingComposite(qpu, embedding=emb)

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