import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
import scipy as sp



results1 = []
dim_arr = []

repeats = 10

for dim in range(3, 100000, 1000):

    for rep in range(repeats):
        world = World(dim)
        line1 = world.generate(1)
        line2 = world.generate(1)
        
        start = time.time()
        world.quadratic_solve(line1, line2)
        end = time.time()

        time_taken = end-start

    results1.append(time_taken/repeats)
    dim_arr.append(dim)
    print(f"Currently at loop: {dim}")

plt.plot(dim_arr, results1)
plt.xlabel("Dimensionality n")
plt.ylabel("Time (seconds)")
plt.savefig("quad_solve_test.png")
plt.show()