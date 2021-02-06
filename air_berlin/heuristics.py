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


# Given large tokens X, returns the adjacency matrix of a directed graph representing the moves that can occur
# A[i][j] = 1 if small token at vertex i could move to vertex j if the zero was at vertex j, else 0
def adjacency_matrix(X):
    m = X.shape[0]
    i = np.arange(m).reshape(m, 1)
    j = np.arange(m).reshape(1, m)

    A = np.zeros((m, m), dtype=np.int)

    # Add the edges for the main cycle
    A[(i == ((j + 1) % m)) | (i == ((j - 1) % m))] = 1

    # Add the jumps
    A[(i == ((j + X) % m)) | (i == ((j - X) % m))] = 1

    return A


# Given the adjacency matrix A, returns a matrix P where P[i][j] is the length of the shortest path from i to j
# This function exploits the fact that A^k[i][j] contains the number of walks from i to j with length at least k
def shortest_path(A):
    P = A.copy()
    Ak = A.copy()
    for k in range(2, A.shape[0] // 2 + 1):
        Ak = Ak @ A
        P[(P == 0) & (Ak != 0)] = k
    np.fill_diagonal(P, 0)
    return P


# Measures how much small disk i dislikes the position of small disk j according to the rules of Air Berlin
# small disk i does not want to move, and it has a list of valid positions for small disk j
# if small disk j is at one of those positions, then small disk i likes that (returns 0)
# Otherwise, the length of the shortest path from position j to a valid position for small disk j is returned
# Parameters:
#   i, j - position of small tokens in the ring
#   ival, jval - values of the small tokens at position i and j, should be in [1, n]
#   Note: i, j, ival, jval can be single values or arrays with broadcastable shapes
#   n - Air Berlin n value
#   P - must be shortest_path(adjacency_matrix(X)) where X is the large tile values for the game
def air_berlin_distance(i, j, ival, jval, n, P):
    m = n * n + 1

    # Convert inputs to numpy arrays, broadcast their shapes, and add a trailing dimension
    vals = (i, j, ival, jval)
    arrays = [val if isinstance(val, np.ndarray) else np.array(val) for val in vals]
    arrays = np.broadcast_arrays(*arrays)
    arrays = [array[..., np.newaxis] for array in arrays]
    i, j, ival, jval = arrays

    # Get valid j positions
    val_diff = signed_mod_distance(ival - 1, jval - 1, n)
    valid_j = (i + val_diff * n) + np.arange(-n + 1, n)

    # Shift the positions to account for the zero
    roll_over = np.broadcast_to((ival + val_diff) > n, valid_j.shape)
    roll_back = np.broadcast_to((ival + val_diff) <= 0, valid_j.shape)
    valid_j[roll_over] += 1
    valid_j[roll_back] -= 1
    valid_j = valid_j % m

    all_paths_to_valid_j = P[j, valid_j]
    shortest_path_to_valid_j = all_paths_to_valid_j.min(axis=-1)
    return shortest_path_to_valid_j


class AirBerlinDistanceHeuristic:
    def __init__(self, n, X, scale=1):
        self.n = n
        self.scale = scale

        P = shortest_path(adjacency_matrix(X))

        m = n * n + 1
        self.m = m

        # Run all possible input combinations for i, j, ival, jval
        # Results are stored in a 4D lookup table
        i = np.arange(m).reshape(m, 1, 1, 1)
        j = np.arange(m).reshape(1, m, 1, 1)
        ivals = np.arange(1, n + 1).reshape(1, 1, n, 1)
        jvals = np.arange(1, n + 1).reshape(1, 1, 1, n)
        self.D = air_berlin_distance(i, j, ivals, jvals, n, P)

    def __call__(self, X, x, xvals):
        N = x.shape[0]
        indices = np.arange(self.m).reshape(1, self.m)
        indices = np.broadcast_to(indices, (N, self.m))
        indices = indices[x != 0].reshape(N, self.m - 1)
        values = x[x != 0].reshape(N, self.m - 1)

        i = indices[..., np.newaxis]
        j = indices[..., np.newaxis, :]
        ivals = values[..., np.newaxis] - 1
        jvals = values[..., np.newaxis, :] - 1

        distances = self.D[i, j, ivals, jvals]
        h = distances.sum(axis=(-1, -2))

        return h * self.scale


# Simplest heuristic, returns 0 if solved 1 if not
class SolvedHeuristic:
    def __init__(self, n):
        self.n = n

    def __call__(self, X, x, xvals):
        d = x - np.roll(x, 1, axis=1)
        num_decreasing_elements = (d < 0).sum(axis=1)
        solved = num_decreasing_elements == 1
        return 1 - solved


class PairwiseDistanceHeuristic:
    def __init__(self, n, scale=1):
        self.m = n * n
        self.n = n
        self.scale = scale

        self.mod_difference_matrix = mod_difference_matrix(n*n)

    def __call__(self, X, x, xvals):
        N = x.shape[0]

        # Filter out the zero, so x only has tiles
        x = x[x != 0].reshape(N, self.m) - 1

        diff = (np.expand_dims(x, -2) - np.expand_dims(x, -1)) % self.n
        diff *= self.n

        diff = (diff - self.mod_difference_matrix) % self.m
        diff = np.minimum(self.m - np.abs(diff), np.abs(diff))
        diff = np.maximum(0, diff - self.n + 1)

        h = diff.sum(axis=(-1, -2)) * self.scale

        return h


# Gets the signed_mod_distance between adjacent elements
# Distances of 1 or 0 are okay since the elements are allowed to increase around the ring
# However if the distance is >1 or <0 then this is summed into the heuristic.
class AdjacentDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.m = n * n
        self.scale = scale
        self.solved_heuristic = SolvedHeuristic(n)

    def __call__(self, X, x, xvals):

        N = x.shape[0]
        x = x[x != 0].reshape(N, self.m) - 1

        d = signed_mod_distance(x, np.roll(x, -1, axis=1), self.n)
        d[d > 0] -= 1
        h = np.abs(d).sum(axis=1) * self.scale

        return h


class HammingHeuristic:
    def __init__(self, n):
        self.n = n

    def __call__(self, X, x, xvals):
        d = signed_mod_distance(x, np.roll(x, -1, axis=1), self.n + 1)
        d = (d != 0) & (d != 1)
        h = d.sum(axis=1)
        return h


# Upgrades any heuristic by doing a few iterations of breadth first search
class BreadthHeuristic:
    def __init__(self, n, depth, h_breadth):
        self.n = n
        self.depth = depth
        self.h_breadth = h_breadth

    def __call__(self, X, x, xvals):
        x = x.copy()
        xvals = xvals.copy()
        xvals["g"] = 0

        N = x.shape[0]
        hvals = []
        for i in range(N):
            breadth = [(x[i:i+1], xvals[i:i+1])]
            for d in range(self.depth):
                breadth.append(expand(X, *breadth[d], compute_hash=False))
            x_breadth = np.concatenate([elem[0] for elem in breadth])
            xvals_breadth = np.concatenate([elem[1] for elem in breadth])
            hvals_breadth = self.h_breadth(X, x_breadth, xvals_breadth)
            fvals_breadth = hvals_breadth + xvals_breadth["g"]
            hvals.append(np.min(fvals_breadth))
        hvals = np.stack(hvals)

        return hvals


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
