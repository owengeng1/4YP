import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm


world = World(5000)

kmin = 1
kmax = 1000
stepsize = 20
repeats = 20

results_orthog = []
results_mom = []
results_adapt = []
results_ana = []
results_gs1 = []
results_proj1 = []
results_proj2 = []
results_proj3 = []

karr = []



for k in range(kmin, kmax, stepsize):

    tot_orthog = 0
    tot_mom = 0
    tot_adapt = 0
    tot_ana = 0

    tot_gs1 = 0
    tot_proj1 = 0
    tot_proj2 = 0
    tot_proj3 = 0


    for rep in range(repeats):
        vect1 = world.generate(k)
        vect2 = world.generate(k)

        start = time.time()
        _, _, _, gs_time_1, proj_time_1 = world.orthogonal_descent_basis(vect1, vect2)
        end = time.time()
        tot_orthog = tot_orthog + (end-start)/repeats

        start = time.time()
        _, _, _, _, proj_time_2 = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.5)
        end = time.time()
        tot_mom = tot_mom + (end-start)/repeats

        start = time.time()
        _, _, _, _, proj_time_3 = world.orthogonal_descent_momentum_basis_adaptive(vect1, vect2, 1.5)
        end = time.time()
        tot_adapt = tot_adapt + (end-start)/repeats

        start = time.time()
        analytical_sol(vect1, vect2)
        end = time.time()
        tot_ana = tot_ana + (end-start)/repeats

        tot_gs1 = tot_gs1 + gs_time_1/repeats
        tot_proj1 = tot_proj1 + proj_time_1/repeats
        tot_proj2 = tot_proj2 + proj_time_2/repeats
        tot_proj3 = tot_proj3 + proj_time_3/repeats


    

    results_orthog.append(tot_orthog)
    results_mom.append(tot_mom)
    results_adapt.append(tot_adapt)
    results_ana.append(tot_ana)

    results_gs1.append(tot_gs1)
    results_proj1.append(tot_proj1)
    results_proj2.append(tot_proj2)
    results_proj3.append(tot_proj3)

    karr.append(k)

    print(f"Current k = {k}")


plt.plot(karr, results_orthog, label="POCS")
plt.plot(karr, results_mom, label="POCS with Momentum")
plt.plot(karr, results_adapt, label="POCS with Adaptive Momentum")
plt.plot(karr, results_ana, label="Analytical Solution")

plt.xlabel("k", fontsize=18)
plt.ylabel("Time (s)", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize=18)

plt.show()
    
        
plt.plot(karr, results_ana, label="Analytical Solution")
#plt.plot(karr, results_gs1, label="MGS")
plt.plot(karr, results_proj1, label="POCS")
plt.plot(karr, results_proj2, label="Momentum")
plt.plot(karr, results_proj3, label="Adaptive Momentum")


plt.xlabel("k", fontsize=18)
plt.ylabel("Time (s)", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize=18)

plt.show()