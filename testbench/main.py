'''
Testbench for Vector NPA (Nearest Point of Approach) 4YP
By: Owen Geng, Wesley Armour

A testbench environment to generate and interact with vectors which may or may not intersect.
Should provide details about intersections, nearest points of approach and distances involved.
'''

import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


random.seed(43)

results = []
results_iter = []

world = World(100)
line1 = world.generate(1)
line2 = world.generate(1)

ansK1, ansK2, ctr = world.orthogonal_descent(line1, line2)

print("Answers: ")
print(ansK1)
print(ansK2)
print("============")

'''
K1 = np.array([0])
dK1 = np.array([1])
y1 = world.calc_l2(line1.calculate_point(K1), line2.calculate_point(world.calculate_orthogonal(line1.calculate_point(K1), line2)))
y2 = world.calc_l2(line1.calculate_point(dK1), line2.calculate_point(world.calculate_orthogonal(line1.calculate_point(dK1), line2)))
print(y1)
print(y2)
x = y1/(y1-y2)
print(x)
'''
#'''


print(world.quadratic_solve(line1, line2))

#'''

'''
for dim in range(1, 10000, 5):

    total = 0
    avg_iter = 0
    for counts in range (1):
        world = World(1000000)
        line1 = world.generate(1)
        line2 = world.generate(dim)

        start = time.time()
        iter = world.orthogonal_descent(line1, line2)
        end = time.time()
        total = total + ((end - start)/1)
        avg_iter = iter
    results.append(total)
    results_iter.append(avg_iter)
    print(f"Current dim: {dim}")

'''
'''
with open('dat.txt', 'w') as file:
    json.dump(results, file)
file.close()

with open('dat_iter.txt', 'w') as file2:
    json.dump(results_iter, file2)
file2.close()

xpoints = range(np.size(results))
plt.plot(xpoints, results, 'o')
plt.show()
'''