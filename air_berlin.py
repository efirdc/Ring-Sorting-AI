import numpy as np


# Returns a basic solved states with shape (N, n*2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n):
    x = np.repeat(np.arange(n) + 1, n)
    x = np.concatenate((np.array([0]), x))
    x = np.stack((x,) * N)
    return x

# Replaces basic_solved_state on the student server
# as np.stack is not available on the server.
def basic_solved_state_2(N, n):
    x = np.repeat(np.arange(n) + 1, n)
    x = np.concatenate((np.array([0]), x))
    x_stack = np.vstack([x,])
    for i in range(1, N):
        x_stack = np.vstack([x_stack, x])
    return x_stack

# https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

# https://gist.github.com/AidySun/bb2b90a993d74400ababb8c8bdbf1d40
def shuffle_2D_matrix(matrix, axis = 0):
    np.random.seed(10)
    if axis == 0: # by column
        m = matrix.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_matrix = matrix[:, permutation]
    else:          # by row
        m = matrix.shape[0]
        permutation = list(np.random.permutation(m))
        shuffled_matrix = matrix[permutation, :]
    return shuffled_matrix

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

    #actions = np.stack((zero_pos,) * 4)
    actions = np.vstack([zero_pos,])
    for i in range(1,4):
        actions = np.vstack([actions, zero_pos])
    
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

    def get_hash(self, x):
        #return [hash(elem.tobytes()) for elem in list(x)]
        return [hash(bytes(elem)) for elem in list(x)]

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
                raise TimeoutError("Reached max depth.") #(f"Reached max depth of {max_depth}. Probably a cycle in the graph.")
            if h not in self.edges:
                break
            h = self.edges[h]
            x_parent = self.vertices[h]
            out.append(x_parent)

        out.reverse()
        #return np.stack(out)
        return out

# Formula for how many possible states there are for air berlin when you have n types of small disks
# Just gives us an idea of how big of a space we are exploring for each n
def num_possible_states(n):
    return np.factorial(n * n + 1) // (np.factorial(n) ** n)
