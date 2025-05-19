import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm

n = 1000

world = World(n)

res0 = []
res1 = []
res2 = []
res3 = []
res4 = []
res5 = []
res6 = []
res7 = []

karr = []
kmin = 100
kmax = 200
kstep = 10
repeats = 10


betaarr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

for k in range(kmin, kmax, kstep):
    karr.append(k)

    res0tot = 0
    res1tot = 0
    res2tot = 0
    res3tot = 0
    res4tot = 0
    res5tot = 0
    res6tot = 0
    res7tot = 0

    for rep in range(repeats):

        vect1 = world.generate(k)
        vect2 = world.generate(k)

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0)
        if it >= 10000:
            it = res7tot
        res0tot = res0tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.1)
        if it >= 10000:
            it = res7tot
        res1tot = res1tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.2)
        if it >= 10000:
            it = res7tot
        res2tot = res2tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.3)
        if it >= 10000:
            it = res7tot
        res3tot = res3tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.4)
        if it >= 10000:
            it = res7tot
        res4tot = res4tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.5)
        if it >= 10000:
            it = res7tot
        res5tot = res5tot + it/repeats

        _, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.6)
        if it >= 10000:
            it = res7tot
        res6tot = res6tot + it/repeats
        
        #_, _, it, _, _ = world.orthogonal_descent_momentum_basis(vect1, vect2, 0.7)
        #if it >= 10000:
            #it = res7tot
        #res7tot = res7tot + it/repeats
    
    res0.append(res0tot)
    res1.append(res1tot)
    res2.append(res2tot)
    res3.append(res3tot)
    res4.append(res4tot)
    res5.append(res5tot)
    res6.append(res6tot)
    #res7.append(res7tot)

    print(f"k = {k}")


plt.plot(karr, res0, label="beta = 0")
plt.plot(karr, res1, label="beta = 0.1")
plt.plot(karr, res2, label="beta = 0.2")
plt.plot(karr, res3, label="beta = 0.3")
plt.plot(karr, res4, label="beta = 0.4")
plt.plot(karr, res5, label="beta = 0.5")
plt.plot(karr, res6, label="beta = 0.6")
#plt.plot(karr, res7, label="beta = 0.7")

plt.xlabel("k", fontsize=18)
plt.ylabel("Iterations", fontsize = 18)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize=18)

plt.show()