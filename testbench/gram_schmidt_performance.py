from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


random.seed(43)

def formulate_gs_mat(A):

    k, _ = np.shape(A)
    start = time.time()
    for i in range(k):
        for j in range(i+1, k):
            proj = (A[j]@A[i])*A[i]
            A[j] = np.subtract(A[j], proj)
            A[j] = normalise(A[j])
    end = time.time()
    return A, end-start

repeats = 10
highest_dim = 500
stepsize = 50

results = []
k_size = []

for iter in range(3, highest_dim, stepsize):

    k_size.append(iter)
    current_result = 0

    for rep in range(repeats):

        world = World(10000)
        vect = world.generate(iter)
        A = np.delete(vect.dirarr, 0, axis=0)
        #start = time.time()
        _, t = formulate_gs_mat(A)
        #end = time.time()
        current_result = current_result + t
    
    current_result = current_result/repeats
    results.append(current_result)
    print(f"current iter: {iter}")



plt.plot(k_size, results, label="Measured time")
coeff = np.polyfit(k_size, results, 2)
plt.xlabel("k", fontsize=18)
plt.ylabel("Time (s)", fontsize = 18)
p = np.poly1d(coeff)
plt.plot(k_size, p(k_size), label="Fitted quadratic curve")
plt.legend(fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

print(p)


"""

repeats = 20
highest_dim = 5000
start_dim = 1000
stepsize = 50
kconst = 300

n_size = []
results = []

for n in range(start_dim, highest_dim,stepsize):
    n_size.append(n)

    current_result = 0

    for rep in range(repeats):
        world = World(n)
        vect = world.generate(kconst)
        start = time.time()
        world.gram_schmidt(vect)
        end = time.time()
        current_result = current_result + end-start
    
    current_result = current_result/repeats
    results.append(current_result)
    print(f"current iter: {n}")


plt.plot(n_size, results)
plt.xlabel("n", fontsize=18)
plt.ylabel("Time (s)", fontsize = 18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

"""