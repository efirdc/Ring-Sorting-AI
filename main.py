from air_berlin import *

def main():
    n = 3
    #x = basic_solved_state(1, n)
    x = basic_solved_state_2(1, n)
    #x = shuffle_along_axis(x, 1)
    x = shuffle_2D_matrix(x, axis=0)
    X = random_large_discs(n)

    print("Initial state:")
    print("Large Disks:", X)
    print("Small Disks:", x)

    h = MaskedDistanceHeuristic(n, scale=(1 / (8*(n-1))))
    expanded = Expanded()
    expanded.add(x, x_parent=None)

    search_width = 4

    for i in range(100):
        x, x_parent = expand(X, x)
        backtracked = expanded.contains(x)
        x = x[~backtracked]
        x_parent = x_parent[~backtracked]
        expanded.add(x, x_parent)
        
        hvals = h(x)
        arg_best = hvals.argsort()
        arg_best = arg_best[:min(search_width, arg_best.shape[0])]
        
        x = x[arg_best]
        hvals = hvals[arg_best]
        
        print()
        print("h(x):\n", hvals)
        print("Nodes:")
        print(x)
        if hvals[0] < 1e-2:
            print("Done.")
            print("Large Disks:")
            print(X)
            path = expanded.path_to_root(x[:1])
            print("Solution")
            print(path)
            break

main()