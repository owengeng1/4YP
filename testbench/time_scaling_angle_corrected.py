import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm



results = []
results_iter = []


#TEST 1

#TESTING n, k
#param
nmin = 500
nmax = 1000
kmin = 1
kmax = 150
repeats = 5


#GENERATION

for n in range(nmin, nmax, 20):
    world = World(nmin)

    temp = []
    temp_iter = []
    for k in range(kmin, np.min([n, kmax])):
        
        tot = 0
        tot_per_it = 0
        for rep in range(repeats):
            line1 = world.generate(k)
            line2 = world.generate(k)
            a, b, it, _, _ = world.orthogonal_descent_basis(line1, line2)
            theta = calc_principle_angles(line1, line2)
            theta = np.mean(theta)
            if (pow(np.sin(theta),2) < 0.001):
                correction = 10000
            else:
                correction = 1/pow(np.sin(theta),2)
            tot = tot + (it)/(repeats*correction)
        
        temp.append(tot)

        if (k%10 == 0):
            print(f"Currently at n = {n}, k = {k}")
    
    results.append(temp)

    

    

nrange = []
for n in range(nmin, nmax, 20):
    nrange.append(n)

krange = []
for k in range(kmin, kmax):
    krange.append(k)

x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(results)
print(dat)
print(nrange)
print(krange)
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*np.max(dat))
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Iterations")

plt.show()


#GENERATION

"""
world = World(1000)

for k in range(kmin, kmax):
        
    tot = 0
    tot_per_it = 0
    for rep in range(repeats):
        line1 = world.generate(k)
        line2 = world.generate(k)
        a, b, it, _, _ = world.orthogonal_descent_basis(line1, line2)
        theta = calc_principle_angles(line1, line2)
        theta = np.mean(theta)
        if (pow(np.sin(theta),2) < 0.001):
            correction = 1000
        else:
            correction = 1/pow(np.sin(theta),2)
        tot = tot + (it)/(repeats*correction)
    
    results.append(tot)

    if (k%10 == 0):
        print(f"k = {k}")


krange = []
for k in range(kmin, kmax):
    krange.append(k)


plt.plot(krange, results)
plt.show()

print(krange)
print(results)

logkrange = np.log(krange)
plt.xlabel("log(k)", fontsize=18)
logresults = np.log(results)

print(logkrange)
print(logresults)
plt.ylabel("log(iter)", fontsize=18)
plt.title("log-log plot, angle adjusted iterations against k, fixed n = 1000", fontsize=18)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.plot(logkrange, logresults)
plt.show()

"""
