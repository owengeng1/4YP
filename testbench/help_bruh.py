import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *
from matplotlib import cm

world = World(3)

vect1 = world.generate(2)
vect2 = world.generate(1)


print(calc_principle_angles(vect1, vect2))