from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)

#Ensure both versions still represent the same plane

world = World(3)
vect1 = world.generate(2)

line1 = world.generate(1)

print(vect1.calculate_point(world.orthogonal_descent(vect1, line1)[0]))

vect1 = world.gram_schmidt(vect1)

print(vect1.calculate_point(world.orthogonal_descent(vect1, line1)[0]))


