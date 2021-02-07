import numpy as np


# Returns basic solved states with shape (N, n^2 + 1)
# i.e for N=1, n=3 returns [[0 1 1 1 2 2 2 3 3 3]]
def basic_solved_state(N, n, dtype=np.int8):
    x = np.arange(n, dtype=dtype)
    x = np.repeat(x + 1, n)
    x = np.concatenate((np.array([0]), x))
    x = np.stack((x,) * N)
    return x


# returns matrix of size (n^2 + 1, n^2 + 1) with every possible solved state in the rows
def all_solved_states(n):
    m = n * n + 1
    x = basic_solved_state(1, n)[0]
    x = [np.roll(x, i) for i in range(m)]
    x = np.stack(x)
    return x


# Returns uniform distributed array of integers {1, 2, 3, 4} with length n*n + 1
def random_large_discs(n, low=1, high=4):
    return np.random.randint(low, high + 1, size=(n * n + 1))


# Data structure for additional information that is stored with each set of small tiles
xvals_type = [
    ('zero_pos', np.int32),     # index of the zero in x
    ('prev_action', np.int32),  # previous action applied to x
    ('g', np.int32),            # cost so far
    ('h', np.float32),          # heuristic
    ('parent_h', np.float32),   # heuristic of the parent
    ('hash', np.int64),         # hash of small tiles
    ('parent_hash', np.int64)   # hash of the parent small tiles
]


# Given an array of small tiles x with shape (N, n*n + 1), initializes an xvals_type array with shape (N,)
def get_xvals(x):
    xvals = np.zeros(x.shape[0], dtype=xvals_type)
    xvals['zero_pos'] = np.where(x == 0)[1]
    xvals['h'] = np.nan
    xvals['parent_h'] = np.nan
    xvals['hash'] = np.array([hash(elem.tobytes()) for elem in list(x)])
    return xvals


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - array of xvals_type (N,)
#   actions - array of actions of shape (N,), where each action is a distance that the zero will jump
#             positive distances are clockwise and negative are counter-clockwise
# Returns:
#   x, xvals with actions applied.
def apply_action(X, x, xvals, actions, compute_hash=True):
    jumps = xvals['zero_pos'] + actions
    jumps = jumps % X.shape[0]

    N = range(x.shape[0])
    x[N, jumps], x[N, xvals['zero_pos']] = x[N, xvals['zero_pos']], x[N, jumps]

    xvals['zero_pos'] = jumps
    xvals['prev_action'] = actions
    xvals['g'] = xvals['g'] + 1
    xvals['parent_h'] = xvals['h']
    xvals['h'] = np.nan
    if compute_hash:
        xvals['parent_hash'] = xvals['hash']
        xvals['hash'] = np.array([hash(elem.tobytes()) for elem in list(x)])

    return x, xvals


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - array of xvals_type (N,)
# Returns:
#   array of valid jump distances (N, 4)
#       first two indices are 1 and -1
#       last two indices are X[zero_pos]
#       jumps that are the opposite of the prev_jump are set to 0
def valid_actions(X, x, xvals):
    token_jump = X[xvals['zero_pos']]
    token_jump[token_jump == 1] = 0
    ones = np.ones(x.shape[0], dtype=np.int32)
    actions = np.stack((ones, -ones, token_jump, -token_jump), axis=1)
    actions[actions == -xvals['prev_action']] = 0
    return actions


# Parameters:
#   X - array of large discs of shape (m,)
#   x - array of small discs of shape (N, m)
#   xvals - structured array of values (N, 6)
# Returns: (x_new, x_parent)
#   x - expanded x of shape (M, m), N <= M <= 4N depending on how many actions are possible per state in x
#   xvals - expanded xvals of shape (M, 6)
def expand(X, x, xvals, compute_hash=True):
    actions = valid_actions(X, x, xvals)
    action_mask = actions != 0
    actions_per_state = action_mask.sum(axis=1)
    x = x.repeat(actions_per_state, axis=0)
    xvals = xvals.repeat(actions_per_state)
    x, xvals = apply_action(X, x, xvals, actions[action_mask], compute_hash=compute_hash)
    return x, xvals


# Tests if x is solved
# Works by testing if x decreases exactly once
def is_solved(x):
    d = x - np.roll(x, 1, axis=-1)
    num_decreasing_elements = (d < 0).sum(axis=-1)
    solved = num_decreasing_elements == 1
    return solved