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

repeat_count = 100

results1 = []
angle_arr = []

for angle in range(100, 785, 100):
    angle_fix = angle/1000
    total1 = 0
    total2 = 0
    for repeats in range(0, repeat_count):
        line1 = world.generate(1)
        line2 = world.generate_angle(line1, angle_fix)

        start = time.time()
        a, b, ctr = world.orthogonal_descent(line1, line2)
        end = time.time()
        total1 = total1 + ctr/repeat_count

    results1.append(total1)
    angle_arr.append(angle_fix)
    print(f"Current angle: {angle}")

plt.plot(angle_arr, results1, label="Measured iterations")
plt.xlabel("Angle (radians)")
plt.ylabel("Iterations")

x = np.arange(0.1, 0.785, 0.1)
y = 1/pow(np.sin(x),2)
y=y*4
plt.plot(x, y, label="Model function")
plt.legend()
plt.show()