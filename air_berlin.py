import numpy as np
import heapq


# Returns a basic solved states with shape (N, n*2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n, dtype=np.uint8):
    x = np.arange(n, dtype=dtype)
    x = np.repeat(x + 1, n)
    x = np.concatenate((np.array([0]), x))
    x = np.stack((x,) * N)
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


# returns a matrix D with shape (m, m) such that D[i][j] = |i - j|
# For example, m=4 returns:
# [[0 1 2 3]
#  [1 0 1 2]
#  [2 1 0 1]
#  [3 2 1 0]]
def distance_matrix(m):
    x = np.arange(m)
    x = np.expand_dims(x, -1) - np.expand_dims(x, -2)
    x = np.abs(x)
    return x


# Distance based heuristic for an AB state with n tokens
# In short this heuristic is: sum(max(M(x) - n + 1, 0)) * scale
# where M(x) matrix with shape (n^2 + 1, n^2 + 1) such that if x[i] == x[j] then M[i][j] = |i - j| else 0
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


# Keeps track of the explored tree using a hash based graph
# The self.vertices dictionary maps hash(x) -> x
# while the self.edges dictionary maps hash(x) -> hash(x_parent)
class Expanded:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def add(self, xvals, is_root=False):
        new_vertices = {xv["hash"]: xv["prev_action"] for xv in xvals}
        self.vertices.update(new_vertices)

        if not is_root:
            new_edges = {xv["hash"]: xv["parent_hash"] for xv in xvals}
            self.edges.update(new_edges)

    def contains(self, xvals):
        return np.array([xv["hash"] in self.vertices for xv in xvals])

    def path_to_root(self, X, x, xvals, max_depth=1000):
        out = [x]

        h = xvals[0]["hash"]
        a = xvals[0]["prev_action"]

        depth = 0
        while True:
            depth += 1
            if depth >= max_depth:
                raise TimeoutError(f"Reached max depth of {max_depth}. Probably a cycle in the graph.")
            x, _ = apply_action(X, x.copy(), xvals, -a)
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
    return np.factorial(n * n + 1) // (np.factorial(n) ** n)


# Keeps track of the Fringe in a priority queue
# Priority value based on cost from root plus the hueristic value
# Tie breakers are dealt with using self.counter
class Fringe:
    def __init__(self):
        self.fringe = []
        self.counter = 0

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
