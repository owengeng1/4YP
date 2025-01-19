import numpy as np
from vect_classes import World, Vector
from math_operations import *

p1 = np.array([1, 3])
p2 = np.array([2, 5])
p3 = np.array([5, 9])

l1 = np.array([1, 2, 5])
l2 = np.array([3, 5, 9])

print(solve_quadratic(p1, p2, p3))
print(solve_quadratic_2(l1, l2))