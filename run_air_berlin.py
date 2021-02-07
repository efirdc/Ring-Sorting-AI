#!/usr/bin/env python
from air_berlin import *
import sys


def main():
    # X = input("Large disks:\n")
    # x = input("Small disks:\n")
    with open("AB.sample") as f:
        X = f.readline()
        x = f.readline()

    X = np.array([int(disc) for disc in X.split(" ")], dtype=np.int32)
    x = np.array([[int(disc) for disc in x.split(" ")]], dtype=np.int8)

    n = int(np.sqrt(X.shape[0] - 1))

    heuristic = PairwiseDistanceHeuristic(n, scale=(1 / (8 * (n - 1))))
    fringe = Fringe()
    expanded = Expanded()

    path = search(X, x, heuristic, fringe, expanded, verbose=False)

    f = open(sys.argv[2], "w")
    if path is None:
        f.write("No solution\n")
    else:
        f.write("Solution is\n")
        for row in path:
            f.write(str(row)[1:-1] + "\n")


if __name__ == "__main__":
    main()
