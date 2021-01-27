from air_berlin import *
import argparse


def check_positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value given must be a positive number")
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('groups', type=int)
    args = parser.parse_args()

    n = args.groups
    x = basic_solved_state(1, n)
    x = shuffle_along_axis(x, 1)
    X = random_large_discs(n)

    print("Initial state:")
    print("Large Disks:", X)
    print("Small Disks:", x)

    heuristic = MaskedDistanceHeuristic(n, scale=(1 / (8*(n-1))))
    expanded = Expanded()
    expanded.add(x, x_parent=None)

    fringe = Fringe()
    cost_to_root = 0

    for i in range(5000):
        x, x_parent = expand(X, x)
        backtracked = expanded.contains(x)
        x = x[~backtracked]
        x_parent = x_parent[~backtracked]
        expanded.add(x, x_parent)
        
        hvals = heuristic(x)
        cost_to_root += 1
        
        for idx in range(len(x)):
            # cost_to_root = g(x), h[idx] = h(x), x[idx] = state
            fringe.push(cost_to_root + hvals[idx], (cost_to_root, hvals[idx], x[idx]))

        current = fringe.pop()
        g,h,x = current[2]
        print("Turn: ", i)
        print("Cost: ", g+h)
        print("Heuristic: ", h)
        print("Large disks:  ", X)
        print("Popped State: ", x)
        print()
        x = np.expand_dims(x, 0)
        cost_to_root = g
        
        print((x != np.roll(x, 1, axis=1)).sum(axis=1))
        if (x != np.roll(x, 1, axis=1)).sum(axis=1) == n:
            print("Done.")
            print("Large Disks:")
            print(X)
            path = expanded.path_to_root(x[:1])
            print("Solution")
            print(path)
            break

main()