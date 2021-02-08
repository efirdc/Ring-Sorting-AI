from air_berlin import *


if __name__ == "__main__":
    X = input()
    x = input()

    X = np.array([int(disc) for disc in X.split(" ")], dtype=np.int32)
    x = np.array([[int(disc) for disc in x.split(" ")]], dtype=np.int8)

    if np.all(X == 1):
        print("No solution")
        exit()

    n = int(np.sqrt(X.shape[0] - 1))

    heuristic = PairwiseDistanceHeuristic(n, scale=1 / (n*n))

    if n <= 5:
        N = 2 ** 16
    elif n >= 10:
        N = 2 ** 6
    else:
        N = {6: 2**14, 7: 2**12, 8: 2**10, 9: 2**8}[n]

    path = search(X, x, heuristic, BeamFringe(), Expanded(), search_width=N, verbose=False)

    if path is None:
        print("No solution")
    else:
        print("Solution is")
        for row in path:
            print(str(row)[1:-1])
