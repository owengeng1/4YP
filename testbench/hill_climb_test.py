import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *



repeats = 100
itarr = []
narr = []
nmin = 100
nmax = 1001
step_size = 100

for n in range(nmin, nmax, step_size):

    world = World(n)
    vect1 = world.generate(1)
    vect2 = world.generate(1)
    totit = 0
    for rep in range(repeats):
      _, _, it = world.step_descent_greedy(vect1, vect2)
      totit = totit + it/repeats
    itarr.append(totit)
    narr.append(n)
    print(f"Current n: {n}")

plt.plot(narr, itarr)

plt.show()