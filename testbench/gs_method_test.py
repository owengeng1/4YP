import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


kmin = 230
kmax = 250

stepsize = 4

world = World(501)

repeats = 10

default_m = []
gs_m = []
momentum_m = []
both_m = []
energy_m = []
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
        #world.orthogonal_descent(v1, v2)
        end = time.time()

        #default_tot = default_tot + end - start

        start = time.time()
        world.orthogonal_descent_basis(v1, v2)
        end = time.time()

        gs_tot = gs_tot + end - start

        start = time.time()
        #world.orthogonal_descent_momentum(v1, v2, 0.2)
        end = time.time()

        #momentum_tot = momentum_tot + end - start

        start = time.time()
        world.orthogonal_descent_momentum_basis(v1, v2, 0.2)
        end = time.time()

        both_tot = both_tot + end - start

        start = time.time()
        world.energy_descent_2(v1, v2, 0.8, 0.8)
        end = time.time()

        energy_tot = energy_tot + end - start
    
    default_m.append(default_tot/repeats)
    gs_m.append(gs_tot/repeats)
    momentum_m.append(momentum_tot/repeats)
    both_m.append(both_tot/repeats)
    energy_m.append(energy_tot/repeats)

    k_list.append(i)

    print(f"Currently at: {i}")


plt.plot(k_list, default_m, label = "Default method")
plt.plot(k_list, gs_m, label = "With basis orthogonalisation")
plt.plot(k_list, momentum_m, label = "With momentum")
plt.plot(k_list, both_m, label = "Combined")
plt.plot(k_list, energy_m, label = "Energy method")

plt.xlabel("Dimensionality k")
plt.ylabel("Time (s)")

plt.legend()

plt.show()