import numpy as np
from .game import *


# Given an input array x with shape (..., m)
# returns an array E with shape (..., m, m)
# such that E[i, j] is 1 if x[i] == x[j], 0 otherwise
def pairwise_equality(x):
    out = np.expand_dims(x, -1) == np.expand_dims(x, -2)
    out = out.astype(np.int32)
    return out


# distance between x1 and x2 in mod m arithmetic
def mod_distance(x1, x2, m):
    d = np.abs(x1 - x2)
    d = np.minimum(d, m - d)
    return d


# signed distance between x1 and x2 in mod m arithmetic
# the sign tells us if x2 is positioned "clockwise" relative to x1
def signed_mod_distance(x1, x2, m):
    d = x2 - x1
    half_m = m // 2
    d = (d + half_m) % m - half_m
    return d


# returns a matrix D with shape (m, m) such that D[i][j] = mod_distance(i, j, m)
# For example, m=5 returns:
# [[0 1 2 2 1]
#  [1 0 1 2 2]
#  [2 1 0 1 2]
#  [2 2 1 0 1]
#  [1 2 2 1 0]]
def mod_distance_matrix(m):
    x = np.arange(m)
    D = mod_distance(np.expand_dims(x, -2), np.expand_dims(x, -1), m)
    return D


# returns a matrix D with shape (m, m) such that D[i][j] = signed_mod_distance(i, j, m)
# For example, m=4 returns:
# [[ 0 -1 -2  2  1]
#  [ 1  0 -1 -2  2]
#  [ 2  1  0 -1 -2]
#  [-2  2  1  0 -1]
#  [-1 -2  2  1  0]]
def signed_mod_distance_matrix(m):
    x = np.arange(m)
    D = signed_mod_distance(np.expand_dims(x, -2), np.expand_dims(x, -1), m)
    return D


# returns a matrix D with shape (m, m) such that D[i][j] = (j - i) % m
# For example, m=5 returns:
# [[0 1 2 3 4]
#  [4 0 1 2 3]
#  [3 4 0 1 2]
#  [2 3 4 0 1]
#  [1 2 3 4 0]]
def mod_difference_matrix(m):
    x = np.arange(m)
    D = np.expand_dims(x, -2) - np.expand_dims(x, -1)
    D = D % m
    return D


class PairwiseDistanceHeuristic:
    def __init__(self, n, scale=1):
        self.m = n * n
        self.n = n
        self.scale = scale

        self.mod_difference_matrix = mod_difference_matrix(n*n)
        self.solved = basic_solved_state(1, n)[0]

    def __call__(self, x, xvals):
        N = x.shape[0]

        # Filter out the zero, so x only has tiles
        x0 = x
        x = x[x != 0].reshape(N, self.m) - 1

        diff = (np.expand_dims(x, -2) - np.expand_dims(x, -1)) % self.n
        diff *= self.n

        diff = (diff - self.mod_difference_matrix) % self.m
        diff = np.minimum(self.m - np.abs(diff), np.abs(diff))
        diff = np.maximum(0, diff - self.n + 1)

        h = diff.sum(axis=(-1, -2)) * self.scale

        # The zero does not participate in the heuristic until all small tiles are in the correct order.
        # Then the heuristic is just 1 if placed correctly and 0 if not
        x_zero_shifted = np.stack([np.roll(row, -zero) for row, zero in zip(list(x0), xvals["zero_pos"])])
        h = h + 1 - np.all(x_zero_shifted == self.solved, axis=1)

        return h


# Gets the signed_mod_distance between adjacent elements
# Distances of 1 or 0 are okay since the elements are allowed to increase around the ring
# However if the distance is >1 or <0 then this is summed into the heuristic.
class AdjacentDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.scale = scale

    def __call__(self, x):
        d = signed_mod_distance(x, np.roll(x, -1, axis=1), self.n + 1)
        d[d > 0] -= 1
        h = np.abs(d).sum(axis=1)
        return h * self.scale


# Computes hamming distance between x and all possible solved states
# Then the heuristic is the minimum possible hamming distance.
# This doesn't work very well but could easily be changed to compute a different distance measure.
class TestAllHeuristic:
    def __init__(self, n):
        self.all = all_solved_states(n)
        self.all = np.expand_dims(self.all, 0)

    def __call__(self, x):
        x = np.expand_dims(x, 1)
        E = x != self.all
        h = E.sum(axis=2).min(axis=1)
        return h


# NOTE: This was designed for the game where it is not required to have sequential elements
# i.e [2 2 1 1 1 0 3 3 3 2] is a valid solution with this heuristic
# Distance based heuristic for an AB state with n tokens
# In short this heuristic is: sum(max(M(x) - n + 1, 0)) * scale
# where M(x) matrix with shape (n^2 + 1, n^2 + 1)
#   such that if x[i] == x[j] then M[i][j] = min(|i - j|, n - |i - j|) else 0
class MaskedDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.scale = scale

        self.D = mod_distance_matrix(n * n + 1)
        self.D = np.maximum(0, self.D - n + 1)
        self.D = self.D * scale

    def __call__(self, x):
        E = pairwise_equality(x)
        h = (self.D * E).sum(axis=(-1, -2))
        return h
