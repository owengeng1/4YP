from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


random.seed(44)

results = []
nstore = []

for n in range(2, 1000000, 100000):
    world = World(n)
    line = world.generate(1)
    pt = np.ones(n)
    start = time.time()
    world.calculate_orthogonal(pt, line)
    end = time.time()
    results.append(end - start)
    nstore.append(n)
    print("looped")

plt.plot(nstore, results)
print(nstore)
print(results)
plt.show()