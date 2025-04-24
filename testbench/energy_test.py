import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)
world = World(3)

v1 = world.generate(1)
v2 = world.generate_angle(v1, 0.4)

print(world.energy_descent_2(v1, v2))
print(world.orthogonal_descent_momentum_basis(v1, v2, 0.3))
print(world.orthogonal_descent_basis(v1, v2))