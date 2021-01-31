from air_berlin import *


if __name__ == '__main__':
    n = 4
    N = 1
    x = basic_solved_state(N, n)
    x = shuffle_along_axis(x, 1)
    X = random_large_discs(n)
    heuristic = PairwiseDistanceHeuristic(n, scale=1 / (n*n - 1))

    print("Initial state:")
    print("Large Disks:", X)
    print("Small Disks:", x)

    path = search(X, x, heuristic, search_width=1, log_interval=1000, cost_scale=1)
