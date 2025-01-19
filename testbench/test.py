import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

random.seed(43)

world = World(100)
line1 = world.generate(1)