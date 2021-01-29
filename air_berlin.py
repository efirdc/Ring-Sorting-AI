import numpy as np
import math
import heapq

# Returns a basic solved states with shape (N, n^2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n, dtype=np.uint8):
    x = np.arange(n, dtype=dtype)
    x = np.repeat(x + 1, n)
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


# Given an array x with shape (N, m)
# returns a structured array xvals with shape (N, 6)
# xvals consists of 6 values that stores information for each node in x
xvals_type = [('zero_pos', 'i4'), ('prev_action', 'i4'), ('g', 'i4'),
              ('h', 'f4'), ('hash', 'i8'), ('parent_hash', 'i8')]
def get_xvals(x):
    xvals = np.zeros(x.shape[0], dtype=xvals_type)
    xvals['zero_pos'] = np.where(x == 0)[1]
    xvals['h'] = np.nan
    xvals['hash'] = np.array([hash(elem.tobytes()) for elem in list(x)])
    return xvals


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - structured array of values (N, 6)
#   actions - array of valid jumps that can be performed
#       m -> jump CW by m | -m -> jump CC by m
# Returns:
#   x with actions applied (N, m)
#   xvals  with updated information for each node (N, 6)
def apply_action(X, x, xvals, actions):
    jumps = xvals['zero_pos'] + actions
    jumps = jumps % X.shape[0]

    N = range(x.shape[0])
    x[N, jumps], x[N, xvals['zero_pos']] = x[N, xvals['zero_pos']], x[N, jumps]
    xvals['zero_pos'] = jumps
    xvals['prev_action'] = actions
    xvals['g'] = xvals['g'] + 1
    xvals['h'] = np.nan
    xvals['parent_hash'] = xvals['hash']
    xvals['hash'] = np.array([hash(elem.tobytes()) for elem in list(x)])

    return x, xvals


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - structured array of values (N, 6)
# Returns:
#   array of valid jump distances (N, 4) y
#       first two indices are 1 and -1
#       last two indices are X[zero_pos]
#       jumps that are the opposite of the prev_jump are set to 0
def valid_actions(X, x, xvals):
    token_jump = X[xvals['zero_pos']]
    token_jump[token_jump == 1] = 0
    ones = np.ones(x.shape[0], dtype='i4')
    actions = np.stack((ones, -ones, token_jump, -token_jump), axis=1)
    actions[actions == -xvals['prev_action']] = 0
    return actions


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - structured array of values (N, 6)
# Returns: (x_new, x_parent)
#   x - expanded x of shape (M, m), 2N <= M <= 4N depending on how many actions are possible per state in x
#   xvals - expanded xvals of shape (M, 6)
def expand(X, x, xvals):
    actions = valid_actions(X, x, xvals)
    action_mask = actions != 0
    actions_per_state = action_mask.sum(axis=1)
    x = x.repeat(actions_per_state, axis=0)
    xvals = xvals.repeat(actions_per_state)
    x, xvals = apply_action(X, x, xvals, actions[action_mask])
    return x, xvals


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
    def __init__(self, n, scale=1, weak_zero=False):
        self.m = n * n
        self.n = n
        self.scale = scale

        self.ring_matrix = ring_matrix(n*n)
        self.weak_zero = weak_zero

        self.solved = basic_solved_state(1, n)[0]

    def __call__(self, x, xvals):
        if x.shape[0] == 0:
            raise ValueError("x is empty")

        zero_pos = xvals["zero_pos"]

        x_zero_shifted = np.stack([np.roll(x0, -zero) for x0, zero in zip(list(x), zero_pos)])
        if self.weak_zero:
            h_zero = 1 - np.all(x_zero_shifted == self.solved, axis=1)
        else:
            diff_zero = (x_zero_shifted - self.solved) % (self.n)
            diff_zero = np.minimum(self.n - np.abs(diff_zero), np.abs(diff_zero))
            h_zero = diff_zero.sum(axis=-1)


        N = x.shape[0]
        x = x[x != 0].reshape(N, self.m) - 1

        diff = (np.expand_dims(x, -2) - np.expand_dims(x, -1)) % self.n
        diff *= self.n

        diff = (diff - self.ring_matrix) % self.m
        diff = np.minimum(self.m - np.abs(diff), np.abs(diff))
        diff = np.maximum(0, diff - self.n + 1)

        h = diff.sum(axis=(-1, -2))

        return h * self.scale + h_zero


# Keeps track of the explored tree using a hash based graph
# The self.vertices dictionary maps hash(x) -> x
# while the self.edges dictionary maps hash(x) -> hash(x_parent)
class Expanded:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def __len__(self):
        return len(self.vertices)

    def add(self, xvals, is_root=False):
        new_vertices = {xv["hash"]: xv["prev_action"] for xv in xvals}
        self.vertices.update(new_vertices)

        if not is_root:
            new_edges = {xv["hash"]: xv["parent_hash"] for xv in xvals}
            self.edges.update(new_edges)

    def contains(self, xvals):
        return np.array([xv["hash"] in self.vertices for xv in xvals])

    def path_to_root(self, X, x, xvals, max_depth=10000000):
        out = [x]

        h = xvals[0]["parent_hash"]
        a = xvals[0]["prev_action"]

        depth = 0
        while True:
            depth += 1
            if depth >= max_depth:
                raise TimeoutError(f"Reached max depth of {max_depth}. Probably a cycle in the graph.")

            x, xvals = apply_action(X, x.copy(), xvals, -a)
            out.append(x)

            h = self.edges[h]
            if h not in self.edges or h == 0:
                break
            a = self.vertices[h]

        out.reverse()
        return np.concatenate(out)


# Formula for how many possible states there are for air berlin when you have n types of small disks
# Just gives us an idea of how big of a space we are exploring for each n
def num_possible_states(n):
    return math.factorial(n * n + 1) // (math.factorial(n) ** n)


# Keeps track of the Fringe in a priority queue
# Priority value based on cost from root plus the hueristic value
# Tie breakers are dealt with using self.counter
class Fringe:
    def __init__(self):
        self.fringe = []
        self.counter = 0

    def __len__(self):
        return len(self.fringe)

    def push(self, x, xvals):
        N = x.shape[0]
        for i in range(N):
            heapq.heappush(self.fringe, (xvals['g'][i] + xvals['h'][i], (self.counter, x[i], xvals[i])))
            self.counter += 1

    def pop(self, num):
        xs = []
        xvals = []
        for i in range(num):
            if self.fringe:
                _, x, xval = heapq.heappop(self.fringe)[1]
                xs.append(x)
                xvals.append(xval)
        xs = np.stack(xs)
        xvals = np.stack(xvals)
        return xs, xvals
