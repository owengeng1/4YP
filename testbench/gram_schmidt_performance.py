from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


random.seed(43)
repeats = 10
highest_dim = 550
stepsize = 50

results = []
k_size = []

for iter in range(3, highest_dim, stepsize):

    k_size.append(iter)
    current_result = 0

    for rep in range(repeats):

        world = World(1000)
        vect = world.generate(iter)
        start = time.time()
        world.gram_schmidt(vect)
        end = time.time()
        current_result = current_result + end-start
    
    current_result = current_result/repeats
    results.append(current_result)
    print(f"current iter: {iter}")



plt.plot(k_size, results)
coeff = np.polyfit(k_size, results, 2)
p = np.poly1d(coeff)
plt.plot(k_size, p(k_size))
plt.show()