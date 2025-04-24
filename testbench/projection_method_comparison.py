import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *


random.seed(43)

world = World(10)

v1 = world.generate(3)
v2 = world.generate(3)

print(world.orthogonal_descent(v1, v2))
print(world.orthogonal_descent_basis(v1, v2))

