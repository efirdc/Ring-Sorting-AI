from search import *


if __name__ == "__main__":
    X = input("Large disks:\n")
    x = input("Small disks:\n")

    X = np.array([int(disc) for disc in X.split(" ")], dtype=np.int32)
    x = np.array([[int(disc) for disc in x.split(" ")]], dtype=np.uint8)

    n = int(np.sqrt(X.shape[0] - 1))

    heuristic = YetAnotherAscendingDistanceHeuristic(n, scale=(1 / (8 * (n - 1))))

    path = search(X, x, heuristic, verbose=False)

    print("Solution is")
    for row in path:
        print(str(row)[1:-1])
