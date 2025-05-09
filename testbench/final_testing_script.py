import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm


random.seed(42)

results = []
results_iter = []


#TEST 1

#TESTING n, k
#param
nmin = 2
nmax = 40
kmin = 1
kmax = nmax-1
repeats = 10
nrange = []
krange = []


#GENERATION

for n in range(nmin, nmax):
    world = World(nmin)

    temp = []
    temp_iter = []
    for k in range(kmin, n):
        
        tot = 0
        tot_per_it = 0
        for rep in range(repeats):
            line1 = world.generate(1)
            line2 = world.generate(k)
            start = time.time()
            a, b, it = world.step_descent_greedy(line1, line2)
            end = time.time()
            tot = tot + (it)/repeats
            tot_per_it = tot_per_it + ((end-start)/it)/repeats

        temp.append(tot)
        temp_iter.append(tot_per_it)

        if ((k-1)%100 == 0):
            print(f"n = {n}, k = {k}")
        
        for k in range(n, nmax-1):
            temp.append(0)
            temp_iter.append(0)

    
    results.append(temp)
    results_iter.append(temp_iter)
    nrange.append(n)

    

for k in range(kmin, nmax-1):
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
ax.set_zlabel("Iterations")

plt.show()


x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(results_iter)
print(dat)
print(dat.max())
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*dat.max())
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Time per iteration (s)")

plt.show()


"""

#TEST 2

n = 30
repeats = 10

world = World(n)

karr = []

for k in range(1, n):
    l1 = world.generate(1)
    l2 = world.generate(k)
    tot_per_it = 0
    for rep in range(repeats):
        start = time.time()
        a, b, it = world.step_descent(l1, l2)
        end = time.time()
        tot_per_it = tot_per_it + ((end-start)/it)/repeats
    results.append(tot_per_it)
    karr.append(k)
    print(f"k = {k}")

plt.plot(karr, results)
plt.xlabel("k")
plt.ylabel("Time per iteration (s)")
plt.show()

logresults = np.log(results)
logkarr = np.log(karr)
plt.clf()

plt.plot(logkarr, logresults)
plt.xlabel("log(k)")
plt.ylabel("log(t)")
plt.show()


print("Log-results")
print(np.array(logresults).max())
print(np.array(logresults).min())

print("karr")
print(np.array(karr).max())
print(np.array(karr).min())

print("Gradient approx")
print((np.array(logresults).max()-np.array(logresults).min())/(np.array(logkarr).max()-np.array(logkarr).min()))

"""