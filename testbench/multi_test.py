import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm

random.seed(42)
world = World(2)
l1 = world.generate(1)
l2 = world.generate(1)
l3 = world.generate(1)


larr = [l1, l2, l3]

xmin = -200
xmax = 200
ymin = -200
ymax = 200

dat = []

min = 1000
xminloc = 0
yminloc = 0

for x in range(xmin, xmax+1):
    row = []
    for y in range(ymin, ymax+1):
        pt = np.array([x, y])
        tot = 0
        for i in range(np.size(larr)):
            tot = tot + calc_l2_dist(pt, larr[i].calculate_point(world.calculate_orthogonal(pt, larr[i])))
        
        row.append(tot)
        if tot < min:
            min = tot
            xminloc = x
            yminloc = y
    dat.append(row)
    print("Running")

x = np.linspace(xmin, xmax, (xmax-xmin + 1))
y = np.linspace(ymin, ymax, (ymax-ymin + 1))

print("Found:")
print(xminloc)
print(yminloc)
print(min)
tot = 0
for i in range(np.size(larr)):
    tot = tot + calculate_magnitude(np.subtract(larr[i].calculate_point(world.calculate_orthogonal(np.array([xminloc, yminloc]), larr[i])), np.array([xminloc, yminloc])))
print(tot)

print("Calculated:")
tot = 0
calcpt = world.avg_projections(larr)
print(calcpt)
for i in range(np.size(larr)):
    tot = tot + calculate_magnitude(np.subtract(larr[i].calculate_point(world.calculate_orthogonal(calcpt, larr[i])), calcpt))
    print(tot)
print(tot)

x, y = np.meshgrid(x, y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(x, y, np.array(dat), cmap=cm.hsv,linewidth=0, antialiased=False)

ax.set_zlim(0, 500)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.scatter(np.array([61.75697735]), np.array([10.0978788]), np.array([15.000397274293489]), marker = 'o', color='g')
ax.scatter(np.array([61]), np.array([15]), np.array([12.285115770798178]), marker = 'o', color='b')

plt.show()


