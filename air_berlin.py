import numpy as np
import math

# Returns a basic solved states with shape (N, n^2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n):
    x = np.repeat(np.arange(n) + 1, n)
    x = np.concatenate((np.array([0]), x))
    x = np.stack((x,) * N)
    return x


# returns matrix of size (n*n + 1, n*n + 1) with every possible solved state as a row
def all_solved_states(n):
    m = n * n + 1
    x = basic_solved_state(1, n)[0]
    x = [np.roll(x, i) for i in range(m)]
    x = np.stack(x)
    return x


# https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


# Returns uniform distributed array of integers {1, 2, 3, 4} with length n*n + 1
def random_large_discs(n):
    return np.random.uniform(1, 5, size=(n * n + 1)).astype(np.int)


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   action_ids - int, or array of length (N,) representing which action to perform
#       0 -> jump CW 1 | 1 -> jump CC 1 | 2 -> jump CW by large token | 3 -> jump CC by large token
# Returns:
#   x with actions applied, same shape (N, m)
def apply_action(action_ids, X, x):
    N = range(x.shape[0])

    zero_pos = np.where(x == 0)[1]

    token_jump = X[zero_pos]
    actions = np.stack((zero_pos,) * 4)
    actions = actions
    actions[0] += 1
    actions[1] -= 1
    actions[2] += token_jump
    actions[3] -= token_jump
    chosen_actions = actions[action_ids, N]
    chosen_actions = chosen_actions % X.shape[0]

    x = x.copy()
    x[N, chosen_actions], x[N, zero_pos] = x[N, zero_pos], x[N, chosen_actions]

    return x


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
# Returns:
#   jagged array of shape (N, *), where the second dimension contains the valid action ids for each x[i]
def valid_actions(X, x):
    zero_pos = np.where(x == 0)[1]
    actions = []
    num_actions = (X[zero_pos] != 1) * 2 + 2
    for num in num_actions:
        actions.append(np.arange(num))
    return np.array(actions)


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
# Returns: (x_new, x_parent)
#   x_new - expanded x of shape (M, m), 2N <= M <= 4N depending on how many actions are possible per state in x
#   x_parent - parents of x_new shape (M, m), it has the same rows of the input x, but they are repeated so
#              each row is a parent of the same row in x_new
def expand(X, x):
    actions = valid_actions(X, x)
    actions_per_state = [a.shape[0] for a in actions]
    x_parent = x.repeat(actions_per_state, axis=0)
    x_new = apply_action(np.concatenate(actions), X, x_parent)
    return x_new, x_parent


# Given an input array x with shape (..., m)
# returns an array E with shape (..., m, m)
# such that E[i, j] is 1 if x[i] == x[j], 0 otherwise
def equality_matrix(x):
    out = np.expand_dims(x, -1) == np.expand_dims(x, -2)
    out = out.astype(np.int32)
    return out


# returns a matrix D with shape (m, m) such that D[i][j] = min(|i - j|, n - |i - j|)
# For example, m=4 returns:
# [[0, 1, 2, 1],
#  [1, 0, 1, 2],
#  [2, 1, 0, 1],
#  [1, 2, 1, 0]]
def distance_matrix(m):
    x = np.arange(m)
    x = np.expand_dims(x, -2) - np.expand_dims(x, -1)
    x = np.abs(x)
    x = np.minimum(x, m - x)
    return x


def ring_matrix(m):
    x = np.arange(m)
    x = np.expand_dims(x, -2) - np.expand_dims(x, -1)
    x = x % m
    return x



# Get the distance between x1 and x2 moving clockwise around a ring with m elements
def clockwise_distance(x1, x2, m):
    x = x2 - x1
    half_m = m // 2
    x = (x + half_m) % m - half_m
    return x


def clockwise_distance_matrix(m):
    x = np.arange(m)
    x = clockwise_distance(np.expand_dims(x, -1), np.expand_dims(x, -2), m)
    return x


# Distance based heuristic for an AB state with n tokens
# In short this heuristic is: sum(max(M(x) - n + 1, 0)) * scale
# where M(x) matrix with shape (n^2 + 1, n^2 + 1)
#   such that if x[i] == x[j] then M[i][j] = min(|i - j|, n - |i - j|) else 0
# NOTE: This was designed for the game when it is not required to have sequential elements
# i.e [2 2 1 1 1 0 3 3 3 2] is a valid solution with this heuristic
class MaskedDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.scale = scale

        self.D = distance_matrix(n * n + 1)
        self.D = np.maximum(0, self.D - n + 1)
        self.D = self.D * scale

    def __call__(self, x):
        E = equality_matrix(x)
        h = (self.D * E).sum(axis=(-1, -2))
        return h


# Gets the clockwise distance between adjacent elements and sums them together
class AscendingHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.scale = scale

    def __call__(self, x):
        d = clockwise_distance(x, np.roll(x, -1, axis=1), self.n + 1)
        d[d > 0] -= 1
        h = np.abs(d).sum(axis=1)
        return h * self.scale


# Experiment, broken
class AscendingDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.scale = scale
        self.cw_target = clockwise_distance_matrix(n * n) / n
        self.cw_target += np.sign(self.cw_target) - 1e-3
        self.cw_target = np.trunc(self.cw_target).astype(np.int32)

    def __call__(self, x):
        N = x.shape[0]
        x = x[x != 0].reshape(N, self.n * self.n)
        cw_actual = clockwise_distance(np.expand_dims(x, -1), np.expand_dims(x, -2), self.n)
        cw_diff = clockwise_distance(self.cw_target, cw_actual, self.n)
        cw_diff[cw_diff == 1] = 0
        cw_diff[cw_diff == -1] = 0
        h = np.abs(cw_diff).sum(axis=(1, 2))
        return h * self.scale


# Works like the MaskedDistanceHeuristic, except it is applied multiple times while rolling the ring
# i.e it rolls one group of tokens
# slightly broken (doesnt care about position of 0)
class AscendingMaskedDistanceHeuristic:
    def __init__(self, n, scale):
        self.n = n
        self.m = n * n + 1
        self.scale = scale

        self.D = distance_matrix(self.m)
        self.D = np.maximum(0, self.D - n + 1)
        self.D = self.D * scale

        self.solved = basic_solved_state(1, n)[0]

    def __call__(self, x):
        N = x.shape[0]

        E = np.zeros((N, self.m, self.m))

        #zero_pos = np.where(x == 0)[1]
        #x_zero = np.stack([np.roll(sol, -zero) for sol, zero in zip(list(x), zero_pos)])
        #x = x[x != 0].reshape(N, self.n * self.n)

        E += np.expand_dims(x, -1) == np.expand_dims(x, -2)

        for i in range(1, self.n):
            x_cc_roll = np.roll(x, -i * self.n, axis=1) + i
            x_cw_roll = self.n + i - np.roll(x, i * self.n + 1, axis=1)
            print(x[0])
            print(x_cc_roll[0])
            print(x_cw_roll[0])

            E += np.expand_dims(x, -1) == np.expand_dims(x_cc_roll, -2)
            E += np.expand_dims(x, -1) == np.expand_dims(x_cw_roll, -2)

        h = (self.D * E).sum(axis=(-1, -2))

        return h


class TestAllHeuristic:
    def __init__(self, n):
        self.all = all_solved_states(n)
        self.all = np.expand_dims(self.all, 0)
        print(self.all)

    def __call__(self, x):
        x = np.expand_dims(x, 1)
        #print(x.shape)
        #print(self.all.shape)
        E = x != self.all
        h = E.sum(axis=2).min(axis=1)
        return h


class YetAnotherAscendingDistanceHeuristic:
    def __init__(self, n, weak_zero=False):
        self.m = n * n
        self.n = n
        self.ring_matrix = ring_matrix(n*n)
        self.weak_zero = weak_zero

        self.solved = basic_solved_state(1, n)[0]

    def __call__(self, x):
        zero_pos = np.where(x == 0)[1]

        x_zero_shifted = np.stack([np.roll(x0, -zero) for x0, zero in zip(list(x), zero_pos)])
        diff_zero = (x_zero_shifted - self.solved) % (self.n)
        diff_zero = np.minimum(self.n - np.abs(diff_zero), np.abs(diff_zero))
        h_zero = diff_zero.sum(axis=-1)
        if self.weak_zero:
            h_zero = h_zero > 0

        N = x.shape[0]
        x = x[x != 0].reshape(N, self.m) - 1

        diff = (np.expand_dims(x, -2) - np.expand_dims(x, -1)) % self.n
        diff *= self.n

        diff = (diff - self.ring_matrix) % self.m
        diff = np.minimum(self.m - np.abs(diff), np.abs(diff))
        diff = np.maximum(0, diff - self.n + 1)

        h = diff.sum(axis=(-1, -2))
        return h + h_zero


# Keeps track of the explored tree using a hash based graph
# The self.vertices dictionary maps hash(x) -> x
# while the self.edges dictionary maps hash(x) -> hash(x_parent)
class Expanded:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def get_hash(self, x):
        return [hash(elem.tobytes()) for elem in list(x)]

    def add(self, x, x_parent):
        x_hash = self.get_hash(x)
        new_vertices = {h: xrow for h, xrow in zip(x_hash, list(x))}
        self.vertices.update(new_vertices)

        if x_parent is not None:
            x_parent_hash = self.get_hash(x_parent)
            new_edges = {h: h_parent for h, h_parent in zip(x_hash, x_parent_hash)}
            self.edges.update(new_edges)

    def contains(self, x):
        if isinstance(x, np.ndarray):
            hashes = self.get_hash(x)
        else:
            hashes = x
        return np.array([h in self.vertices for h in hashes])

    def path_to_root(self, x, max_depth=1000):
        h = self.get_hash(x)[0]
        x = x[0]
        out = [x]

        depth = 0
        while True:
            depth += 1
            if depth >= max_depth:
                raise TimeoutError(f"Reached max depth of {max_depth}. Probably a cycle in the graph.")
            if h not in self.edges:
                break
            h = self.edges[h]
            x_parent = self.vertices[h]
            out.append(x_parent)

        out.reverse()
        return np.stack(out)


# Formula for how many possible states there are for air berlin when you have n types of small disks
# Just gives us an idea of how big of a space we are exploring for each n
def num_possible_states(n):
    return math.factorial(n * n + 1) // (math.factorial(n) ** n)
