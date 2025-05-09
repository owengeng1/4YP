import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(42)
world = World(10000)

vect1 = world.generate(1000)
vect2 = world.generate(1000)

start = time.time()
k1, k2 = analytical_sol(vect1, vect2)
end = time.time()

print(np.shape(k1))
print(np.shape(k2))

print(end-start)

start = time.time()
a, b, ctr = world.orthogonal_descent_basis(vect1, vect2)
end = time.time()


print(end-start)
print(ctr)