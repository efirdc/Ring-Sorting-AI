import math
import numpy as np
from .heuristics import *

# Formula for how many possible states there are for air berlin when you have n types of small disks
# Just gives us an idea of how big of a space we are exploring for each n
def num_possible_states(n):
    return math.factorial(n * n + 1) // (math.factorial(n) ** n)


# https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


# figures out what jumps were performed
def get_actions(path):
    zeros = np.where(path == 0)[1]
    m = path.shape[1]
    jumps = signed_mod_distance(zeros[:-1], zeros[1:], m)
    return jumps


def bezier(p0, p1, p2, p3, t):
    omt = 1 - t
    p = omt*omt*omt*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + t*t*t*p3
    return p