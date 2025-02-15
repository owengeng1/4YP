from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


world = World(100)

line1 = world.generate(30)

pt = np.ones(100)

print(line1.calculate_point(world.calculate_orthogonal(pt, line1)))

world.gram_schmidt(line1)

print(line1.calculate_point(world.sum_linear_projections(pt, line1)))
