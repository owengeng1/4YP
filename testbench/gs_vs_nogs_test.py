"""
import os

# Limit all thread pools to 1 thread
os.environ["OMP_NUM_THREADS"] = "1"     # For OpenMP (used by MKL, OpenBLAS)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For macOS Accelerate framework
os.environ["BLIS_NUM_THREADS"] = "1"

"""

import numpy as np #imports
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from scipy import linalg

def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

def matmul(A, B):
    B = transpose(B)
    return [[dot(row, col) for col in B] for row in A]

def identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

def inverse(matrix):
    n = len(matrix)
    AM = [row[:] for row in matrix]  # Copy of matrix
    IM = identity(n)

    for fd in range(n):  # fd: focus diagonal
        if AM[fd][fd] == 0:
            # Swap with a row below that has a non-zero in this column
            for i in range(fd+1, n):
                if AM[i][fd] != 0:
                    AM[fd], AM[i] = AM[i], AM[fd]
                    IM[fd], IM[i] = IM[i], IM[fd]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Scale row to make pivot == 1
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n):
            AM[fd][j] *= fdScaler
            IM[fd][j] *= fdScaler

        # Eliminate all other entries in this column
        for i in range(n):
            if i == fd:
                continue
            crScaler = AM[i][fd]
            for j in range(n):
                AM[i][j] -= crScaler * AM[fd][j]
                IM[i][j] -= crScaler * IM[fd][j]

    return IM



def formulate_pocs_mat(A):
    A = A.T 
    start = time.time()
    newA = matmul(A, linalg.inv((matmul(transpose(A),A))))
    end = time.time()
    return newA, end-start

def formulate_gs_mat(A):

    k, n = np.shape(A)
    start = time.time()
    for i in range(k):
        for j in range(i+1, k):
            proj = (dot(A[j],A[i]))*A[i]
            A[j] = np.subtract(A[j], proj)
            A[j] = normalise(A[j])
    end = time.time()
    return A, end-start

world = World(1000)

kmin = 1
kmax = 50
repeats = 50

results_1 = []
results_2 = []
karr = []

for k in range(kmin, kmax):

    tot_t1 = 0
    tot_t2 = 0

    for rep in range(repeats):
        vect = world.generate(k)
        A = np.delete(vect.dirarr, 0, axis=0)
        _, t1 = formulate_pocs_mat(A)
        _, t2 = formulate_gs_mat(A)
        tot_t1 = tot_t1 + t1/repeats
        tot_t2 = tot_t2 + t2/repeats

    print(f"Currently at k: {k}")
    results_1.append(tot_t1)
    results_2.append(tot_t2)
    karr.append(k)

plt.plot(karr, results_1, label="Standard")
plt.plot(karr, results_2, label="MGS")
plt.xlabel("k", fontsize=18)
plt.ylabel("Time (s)", fontsize=18)
plt.legend(fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()


