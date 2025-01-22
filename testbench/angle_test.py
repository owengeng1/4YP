import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)

world = World(10000)

repeat_count = 10

results1 = []
results2 = []
angle_arr = []

for angle in range(100, 785, 1):
    angle_fix = angle/1000
    total1 = 0
    total2 = 0
    for repeats in range(0, repeat_count):
        line1 = world.generate(1)
        line2 = world.generate_angle(line1, angle_fix)

        start = time.time()
        world.orthogonal_descent(line1, line2)
        end = time.time()
        total1 = total1 + (end-start)/repeat_count

        start = time.time()
        world.quadratic_solve(line1, line2)
        end = time.time()
        total2 = total2 + (end-start)/repeat_count

    results1.append(total1)
    results2.append(total2)
    angle_arr.append(angle_fix)

plt.plot(angle_arr, results1)
plt.plot(angle_arr, results2)
plt.show()