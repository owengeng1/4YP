from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)

world = World(100)
vect = world.generate(50)

vect.print_vector()

vect = world.gram_schmidt(vect)

vect.print_vector()
