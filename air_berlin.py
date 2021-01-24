import numpy as np

# Returns a basic solved states with shape (N, n*2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n):
    x = np.repeat(np.arange(n) + 1, n)
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
# Returns:
#   expanded x of shape (M, m), 2N <= M <= 4N depending on how many actions are possible per state in x
def expand(X, x):
    actions = valid_actions(X, x)
    actions_per_state = [a.shape[0] for a in actions]
    x = x.repeat(actions_per_state, axis=0)
    x = apply_action(np.concatenate(actions), X, x)
    return x


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


class Expanded:
    def __init__(self):
        self.hashed = set()

    def get_hash(self, x):
        return [hash(elem.tobytes()) for elem in list(x)]

    def add(self, x):
        self.add_hashes(self.get_hash(x))

    def add_hashes(self, hashes):
        self.hashed.update(hashes)

    def contains(self, x):
        if isinstance(x, np.ndarray):
            hashes = self.get_hash(x)
        else:
            hashes = x
        return np.array([h in self.hashed for h in hashes])
