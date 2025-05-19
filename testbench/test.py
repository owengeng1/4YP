import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)



def formulate_gs_mat(A):

    k, n = np.shape(A)
    start = time.time()
    for i in range(k):
        for j in range(i+1, k):
            proj = (A[j],A[i])*A[i]
            A[j] = np.subtract(A[j], proj)
            A[j] = normalise(A[j])
    end = time.time()
    return A, end-start

world = World(100)

vect = world.generate(30)


A = np.delete(vect.dirarr, 0, axis=0)
_, t2 = formulate_gs_mat(A)
start = time.time()
world.gram_schmidt(vect)
end = time.time()