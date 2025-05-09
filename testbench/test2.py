import numpy as np
from vect_classes import World, Vector
import random
import cmath
import time
import matplotlib.pyplot as plt
import json
from math_operations import *

imba = 0
tests = 1000

for i in range(tests):
    world1 = World(100)

    world2 = World(10000)

    v1 = world1.generate(1)
    v2 = world1.generate(1)

    v3 = world2.generate(1)
    v4 = world2.generate(1)


    _, _, it1 = world1.step_descent(v1, v2)
    _, _, it2 = world2.step_descent(v3, v4)

    if it2 > it1:
        imba = imba + 1

    if it1 > it2:
        imba = imba - 1
    
    print(f"Current test: {i}")
    print(f"Current imba: {imba}")

print(imba)