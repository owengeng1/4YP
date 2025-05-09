import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

world = World(10000)

kmin = 3500
kmax = 4000
repeats = 5

results = []
karr = []
res_ana = []
karr_ana = []
mult_ana = []
sol_ana = []

"""
for i in range(kmin, kmax, 10):

    vect = world.generate(i)


    tot = 0
    for rep in range(repeats):
        start = time.time()
        world.gram_schmidt(vect)
        end = time.time()
        tot = tot + (end - start)
    results.append(tot/repeats)
    karr.append(i)
    print(f"iteration: {i}")

plt.plot(karr, results)
plt.show()

logkarr= np.log(karr)
logresults = np.log(results)
plt.plot(logkarr,logresults)
plt.show()
"""

for i in range(kmin, kmax, 10):
    
    vect = world.generate(i)
    vect2 = world.generate(i)


    tot = 0
    sol_tot = 0
    mult_tot = 0
    for rep in range(repeats):
        start = time.time()
        a,b,mult_t,sol_t = analytical_sol(vect, vect2)
        end = time.time()
        tot = tot + (end - start)
        mult_tot = mult_tot + mult_t
        sol_tot = sol_tot + sol_t
    res_ana.append(tot/repeats)
    mult_ana.append(mult_tot/repeats)
    sol_ana.append(sol_tot/repeats)
    karr_ana.append(i)
    print(f"iteration: {i}")


plt.plot(karr_ana, res_ana)
plt.xlabel("k")
plt.ylabel("Total time (seconds)")
plt.show()

logkarr= np.log(karr_ana)
logresults = np.log(res_ana)
plt.plot(logkarr,logresults)
plt.xlabel("log(k)")
plt.ylabel("log(t)")
plt.show()


plt.plot(karr_ana, sol_ana)
plt.xlabel("k")
plt.ylabel("Time taken to solve linear system (seconds)")
plt.show()

logresults = np.log(sol_ana)

plt.plot(logkarr,logresults)
plt.xlabel("log(k)")
plt.ylabel("log(t)")
plt.show()

plt.plot(karr_ana, mult_ana)
plt.xlabel("k")
plt.ylabel("Time taken to formulate linear system (seconds)")
plt.show()

logresults = np.log(mult_ana)

plt.plot(logkarr,logresults)
plt.xlabel("log(k)")
plt.ylabel("log(t)")
plt.show()

plt.plot(karr_ana, mult_ana, label="Formulation of linear system")
plt.plot(karr_ana, sol_ana, label = "Solution of linear system")
plt.xlabel("k")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()

