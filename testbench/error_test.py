import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm

nmin = 20
nmax = 40
kmin = 1
kmax = 10
repeats = 20

spacing = 10


avg_kerr = []
avg_disterr = []
avg_perr = []
narr = []


for n in range(nmin, nmax, 10):

    avg_disterrtemp = []
    avg_kerrtemp = []
    avg_perrtemp = []

    for k in range(kmin, kmax):

        world = World(n)
        vect1 = world.generate(k)
        vect2 = world.generate(k)

        k1p, k2p, d = world.orthogonal_descent_basis(vect1, vect2)
        k1a, k2a, a, b = analytical_sol(vect1, vect2)

        k1diff = np.subtract(k1a, k1p)
        k2diff = np.subtract(k2a, k2p)

        kerr = calculate_magnitude(k1diff) + calculate_magnitude(k2diff)

        avg_kerrtemp.append(kerr)

        p1 = vect1.calculate_point(k1a)
        p2 = vect1.calculate_point(k1p)

        p3 = vect2.calculate_point(k2a)
        p4 = vect2.calculate_point(k2p)

        distdiff = np.abs(calc_l2_dist(p1, p3) - calc_l2_dist(p2, p4))
        perrdiff = np.abs(calc_l2_dist(p1, p2) - calc_l2_dist(p3, p4))

        avg_disterrtemp.append(distdiff)
        avg_perrtemp.append(perrdiff)

        if n > 700 and k > 130:
            print(distdiff)
            print(perrdiff)
            print(kerr)
    
    avg_disterr.append(avg_disterrtemp)
    avg_kerr.append(avg_kerrtemp)
    avg_perr.append(avg_perrtemp)

    print(f"n = {n}")
    narr.append(n)
karr = []
for i in range(kmin, kmax):
    karr.append(i)
nrange = np.array(narr)
krange = np.array(karr)

x, y = np.meshgrid(krange, nrange)
    
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(avg_disterr)
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*np.max(dat))
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Distance Error")

plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(avg_kerr)
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*np.max(dat))
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("K Coefficients Total Error")

plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dat = np.array(avg_perr)
surf = ax.plot_surface(x, y, dat, cmap=cm.twilight,linewidth=0, antialiased=False)

ax.set_zlim(0, 1.5*np.max(dat))
ax.set_xlim(kmin, kmax)
ax.set_ylim(nmin, nmax)

ax.invert_yaxis()

ax.set_xlabel("k")
ax.set_ylabel("n")
ax.set_zlabel("Total Distance to Optimal Point")

plt.show()

"""
world = World(1000)

repeats = 20
anglemin = 5
anglemax = 76
resultsk = []
resultsdist = []
angles = []


for i in range(anglemin, anglemax):


    angle = i/100  
    
    totk = 0
    totdist = 0

    for rep in range(repeats):
        l1 = world.generate(1)
        l2 = world.generate_angle(l1, angle)

        k1it, k2it, it = world.orthogonal_descent_basis(l1, l2)
        k1ana, k2ana, _, _ = analytical_sol(l1, l2)

        k1err = np.abs(np.subtract(k1ana, k1it))
        k2err = np.abs(np.subtract(k2ana, k2it))

        p1 = l1.calculate_point(k1it)
        p2 = l1.calculate_point(k1ana)

        p3 = l2.calculate_point(k2it)
        p4 = l2.calculate_point(k2ana)

        disterr = calc_l2_dist(p3, p1) - calc_l2_dist(p4, p2)


        totk = totk + (k1err + k2err)/repeats
        totdist = totdist + disterr/repeats


    resultsk.append(totk)
    resultsdist.append(totdist)
    angles.append(angle)

    print(f"Currently at it = {i}")


plt.plot(angles, resultsk)
plt.xlabel("Angle (rad)", fontsize = 18)
plt.ylabel("K error", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.show()

plt.plot(angles, resultsdist)
plt.xlabel("Angle (rad)", fontsize = 18)
plt.ylabel("Distance error", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.show()
"""