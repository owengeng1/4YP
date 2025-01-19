from vect_classes import Vector, World
import numpy as np
from math_operations import *
import random
import linalg

random.seed(42)

world = World(3)

vect = Vector(2, np.array([[0, 0, 0], [-1, 3, -1], [2, -7, 5]]))

pt = np.array([1,2,3])

vect.print_vector()

kvals = world.calculate_orthogonal(pt, vect)

print(vect.calculate_point(kvals))