import math
import numpy as np


# Formula for how many possible states there are for air berlin when you have n types of small disks
# Just gives us an idea of how big of a space we are exploring for each n
def num_possible_states(n):
    return math.factorial(n * n + 1) // (math.factorial(n) ** n)


# https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)
