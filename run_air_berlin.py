from air_berlin import *


if __name__ == "__main__":
    # X = input("Large disks:\n")
    # x = input("Small disks:\n")

    # X = np.array([int(disc) for disc in X.split(" ")], dtype=np.int32)
    # x = np.array([[int(disc) for disc in x.split(" ")]], dtype=np.int8)

    # n = int(np.sqrt(X.shape[0] - 1))

    n = 3
    N = 1
    x = basic_solved_state(N, n)
    x = shuffle_along_axis(x, 1)
    X = random_large_discs(n)

    # heuristic = AirBerlinDistanceHeuristic(n, X, scale=0.25)
    heuristic = PairwiseDistanceHeuristic(n, scale=1 / (n*n))
    
    path = ida_star(X, x, heuristic)

    if path is None:
        print("No solution")
    else:
        print("Solution is")
        for row in path:
            print(str(row)[1:-1])
