from .utils import *
from .search import *


if __name__ == '__main__':
    n = 3
    N = 1
    x = basic_solved_state(N, n)
    x = shuffle_along_axis(x, 1)
    X = random_large_discs(n)
    heuristic = PairwiseDistanceHeuristic(n, scale=1 / (n*n - 1))

    print("Initial state:")
    print("Large Disks:", X)
    print("Small Disks:", x)

    search(X, x, heuristic)
