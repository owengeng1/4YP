import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

world = World(100)

repeats = 10

kmin = 1
kmax = 40
step_size = 5

results_default = []
results_momentum = []
x = []

for ind in range(kmin, kmax, step_size):

    tot_def = 0
    tot_mom = 0

    x.append(ind)

    for i in range(repeats):

        
        vect1 = world.generate(ind)
        vect2 = world.generate(ind)
        
        #start = time.time()
        _1, _2, ctr = world.orthogonal_descent(vect1, vect2)
        #end = time.time()

        tot_def = tot_def + ctr

        #start = time.time()
        _1, _2, ctr = world.orthogonal_descent_momentum(vect1, vect2, 0.3)
        #end = time.time()

        tot_mom = tot_mom + ctr
        
    results_default.append(tot_def/repeats)
    results_momentum.append(tot_mom/repeats)

    print(f"Currently at {ind}")




plt.plot(x, results_default, label = "Without momentum")
plt.plot(x, results_momentum, label = "With momentum, beta = 0.5")

plt.xlabel('Dimensionality k', fontsize = 10)

plt.ylabel('Iterations', fontsize = 10)

plt.legend(loc="upper left")

plt.show()
