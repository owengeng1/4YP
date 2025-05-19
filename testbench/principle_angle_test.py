import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm

random.seed(48)

nmin = 3
nmax = 50

kmin = 1
kmax = 49
repeats = 10

results = []
krange = []

for n in range(nmin, nmax):

    world = World(n)
    temp = []
    for k in range(kmin, np.min([kmax, n])):
        tot = 0
        for rep in range(repeats):
            vect1 = world.generate(k)
            vect2 = world.generate(k)
            tot = tot + np.mean(calc_principle_angles(vect1, vect2))/repeats
        temp.append(tot)
    
    for k in range(np.min([kmax, n]), kmax):
        temp.append(0)
    
    results.append(temp)
    print(f"n = {n}")

    


nrange = []


for n in range(nmin, nmax):
    nrange.append(n)

for k in range(kmin, kmax):
    krange.append(k)



x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(results)
print(dat)
print(nrange)
print(krange)
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*dat.max())
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Angle, radians")

plt.show()



