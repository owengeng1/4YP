import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm
import math


random.seed(45)

results = []
results_iter = []


#TEST 1

#TESTING n, k
#param
nmin = 1000
nmax = 2000
kmin = 1
kmax = 300
repeats = 10
nrange = []
krange = []
kerrarr = []
disterrarr = []
perrarr = []
res3 = []


#GENERATION

for n in range(nmin, nmax+1, 50):
    world = World(nmin)
    temp = []
    kerrarrtemp = []
    disterrarrtemp = []
    perrarrtemp = []
    for k in range(kmin, kmax, 20):

        print(f"n = {n}, k = {k}")
        
        tot = 0
        tot_per_it = 0
        kerr = 0
        disterr = 0
        perr = 0
        
        for rep in range(repeats):
            line1 = world.generate(k)
            line2 = world.generate(k)

            start = time.time()
            world.orthogonal_descent_basis(line1, line2)
            end = time.time()

            tot = tot + (end-start)/repeats

            """
            k1a, k2a, _, _ = analytical_sol_lstsq(line1, line2)

            

            k1diff = np.subtract(k1a, k1p)
            k2diff = np.subtract(k2a, k2p)

            kerr = kerr + (calculate_magnitude(k1diff) + calculate_magnitude(k2diff))/repeats


            p1 = line1.calculate_point(k1a)
            p2 = line1.calculate_point(k1p)

            p3 = line2.calculate_point(k2a)
            p4 = line2.calculate_point(k2p)

            distdiff = np.abs(calc_l2_dist(p1, p3) - calc_l2_dist(p2, p4))
            perrdiff = np.abs(calc_l2_dist(p1, p2) - calc_l2_dist(p3, p4))

            disterr = disterr + distdiff/repeats
            perr = perr + perrdiff/repeats

            """

        
        """
        kerrarrtemp.append(kerr)
        disterrarrtemp.append(disterr)
        perrarrtemp.append(perr)

        """
        temp.append(tot)


        
    """
    for k in range((n//2)+1, (nmax//2)+1):
        kerrarrtemp.append(0)
        perrarrtemp.append(0)
        disterrarrtemp.append(0)
        results.append(0)
    """
    
    results.append(temp)
    results_iter.append(perrarrtemp)
    res3.append(disterrarrtemp)

    nrange.append(n)

    

for k in range(kmin, kmax, 20):
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
ax.set_zlabel("Time (s)")

plt.show()

"""
x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(results_iter)
print(dat)
print(dat.max())
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*dat.max())
ax.set_xlim(kmin, nmax//2)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Point error")

plt.show()

x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(res3)
print(dat)
print(dat.max())
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*dat.max())
ax.set_xlim(kmin, nmax//2)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Objective Function Error")

plt.show()

"""
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