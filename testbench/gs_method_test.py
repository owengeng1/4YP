import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


kmin = 1
kmax = 300

stepsize = 4

world = World(1000)

repeats = 10

default_m = []
gs_m = []

k_list = []

for i in range(kmin, kmax, stepsize):
    
    default_tot = 0
    gs_tot = 0
    momentum_tot = 0
    both_tot = 0
    energy_tot = 0

    for rep in range(repeats):
    
        v1 = world.generate(i)
        v2 = world.generate(i)

        start = time.time()
        world.orthogonal_descent(v1, v2)
        end = time.time()

        default_tot = default_tot + (end - start)

        start = time.time()
        world.orthogonal_descent_basis(v1, v2)
        end = time.time()

        gs_tot = gs_tot + (end - start)

        #start = time.time()
        #world.orthogonal_descent_momentum(v1, v2, 0.2)
        #end = time.time()

        #momentum_tot = momentum_tot + end - start

        #start = time.time()
        #world.orthogonal_descent_momentum_basis(v1, v2, 0.2)
        #end = time.time()

        #both_tot = both_tot + end - start

        #start = time.time()
        #world.energy_descent_2(v1, v2, 0.8, 0.8)
        #end = time.time()

        #energy_tot = energy_tot + end - start
    
    default_m.append(default_tot/repeats)
    gs_m.append(gs_tot/repeats)
    #momentum_m.append(momentum_tot/repeats)
    #both_m.append(both_tot/repeats)
    #energy_m.append(energy_tot/repeats)

    k_list.append(i)

    print(f"Currently at: {i}")


plt.plot(k_list, default_m, label = "POCS")
plt.plot(k_list, gs_m, label = "With MGS")
#plt.plot(k_list, momentum_m, label = "With momentum")
#plt.plot(k_list, both_m, label = "Combined")
#plt.plot(k_list, energy_m, label = "Energy method")

plt.xlabel("Dimensionality k", fontsize=18)
plt.ylabel("Time (s)", fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=18)

plt.show()

plt.plot(np.log(k_list), np.log(default_m), label= "POCS")
plt.plot(np.log(k_list), np.log(gs_m), label = "With MGS")

plt.xlabel("log(k)", fontsize=18)
plt.ylabel("log(t)", fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=18)

plt.show()