from search import *

if __name__ == '__main__':
    n = 3
    N = 1
    x = basic_solved_state(N, n)
    x = shuffle_along_axis(x, 1)
    X = random_large_discs(n)
    heuristic = MaskedDistanceHeuristic(n, scale=(1 / (8 * (n - 1))))

    print("Initial state:")
    print("Large Disks:", X)
    print("Small Disks:", x)

    search(X, x, heuristic)
