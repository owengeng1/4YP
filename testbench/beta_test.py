import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

world = World(500)

repeats = 10

kmin = 1
kmax = 249
bmin = 0.2
bmax = 0.9
step_size = 20

#results_default = []
results_momentum = []
x = []

for ind in range(kmin, kmax, step_size):

    tot_def = 0
    tot_mom = [0, 0, 0, 0, 0]

    x.append(ind)

    for i in range(repeats):

        
        vect1 = world.generate(ind)
        vect2 = world.generate(ind)
        
        #start = time.time()
        #_1, _2, ctr = world.orthogonal_descent(vect1, vect2)
        #end = time.time()

        #tot_def = tot_def + ctr

        #start = time.time()

        ctr_arr = []
        for beta in range(0, 8):
            beta = beta/10
            _1, _2, ctr = world.orthogonal_descent_momentum(vect1, vect2, beta)
            if ctr == 801:
                ctr_arr.append(np.nan)
            else:    
                ctr_arr.append(ctr)
        #end = time.time()

        tot_mom = np.add(tot_mom, ctr_arr)
        
    #results_default.append(tot_def/repeats)
    results_momentum.append(tot_mom/repeats)

    print(f"Currently at {ind}")


print(results_momentum)

#plt.plot(x, results_default)
#plt.legend(loc="upper left")

lines = plt.plot(x, results_momentum)

plt.xlabel('Dimensionality k', fontsize = 18)
plt.ylabel('Iterations', fontsize = 18)
plt.legend(lines, (r'$ \beta $ = 0', r'$ \beta $ = 0.1', r'$ \beta $ = 0.2', r'$ \beta $ = 0.3', r'$ \beta $ = 0.4', r'$ \beta $ = 0.5', r'$ \beta $ = 0.6', r'$ \beta $ = 0.7'), loc="upper left", fontsize=15)
plt.show()
plt.clf()

results_0 = [row[0] for row in results_momentum]
results_min = [min(row) for row in results_momentum]

plt.plot(x, results_0, label = "Without momentum")
plt.plot(x, results_min, label = r'With best $\beta$')
plt.xlabel('Dimensionality k', fontsize = 10)
plt.ylabel('Iterations', fontsize = 10)
plt.legend(loc = "upper left")
plt.show()