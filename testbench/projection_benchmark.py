#Compare the different projection calculation speeds.

#Check if they both give the same answer.


import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

n = 5000
world = World(n)

repeats = 5
highest_dim = 1500
stepsize = 50

results_method1 = []
results_gs = []
results_method2 = []
results_diff = []
k_size = []

pt = np.ones(n)
for dim in range(2, highest_dim, stepsize):

    method1_tot = 0
    gram_schmidt_tot = 0
    lin_proj_tot = 0
    for rep in range(repeats):
        vect = world.generate(dim)
        method1_start = time.time()
        world.calculate_orthogonal(pt, vect)
        method1_end = time.time()
        method1_tot = method1_tot + method1_end - method1_start

        gs_start = time.time()
        world.gram_schmidt(vect)

        lin_proj_start = time.time()
        world.sum_linear_projections(pt, vect)
        gs_end = time.time()
        gram_schmidt_tot = gram_schmidt_tot + gs_end - gs_start
        lin_proj_tot = lin_proj_tot + gs_end - lin_proj_start
    
    results_gs.append(gram_schmidt_tot/repeats)
    results_method1.append(method1_tot/repeats)
    results_method2.append(lin_proj_tot/repeats)
    results_diff.append((gram_schmidt_tot/repeats) - (method1_tot/repeats))
    k_size.append(dim)
    print(f"Current loop: {dim}")



plt.plot(k_size, results_method1, label="Method 1")

plt.plot(k_size, results_gs, label="Gram_Schmidt")

plt.plot(k_size, results_method2, label="Linear Projections")

plt.legend(loc="upper left")

plt.show()

plt.clf()

plt.plot(k_size, results_diff, label = "GS - M1")

plt.legend(loc="upper left")

plt.show()