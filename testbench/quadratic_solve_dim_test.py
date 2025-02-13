import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

world = World(3)

results1 = []
dim_arr = []

for dim in range(3, 10000, 100):
    line1 = world.generate(1)
    line2 = world.generate(1)
    
    start = time.time()
    world.quadratic_solve(line1, line2)
    end = time.time()

    time_taken = end-start

    results1.append(time_taken)
    dim_arr.append(dim)

plt.plot(dim_arr, results1)
plt.show()