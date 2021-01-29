from air_berlin import *


def search(X, x, h, search_width=1):
    xvals = get_xvals(x)
    xvals['h'] = h(x)

    expanded = Expanded()
    expanded.add(xvals, is_root=True)

    fringe = Fringe()

    for i in range(5000):
        x, xvals = expand(X, x, xvals)
        backtracked = expanded.contains(xvals)
        x = x[~backtracked]
        xvals = xvals[~backtracked]
        expanded.add(xvals)

        xvals['h'] = h(x)
        fringe.push(x, xvals)

        x, xvals = fringe.pop(search_width)
        print("Turn: ", i)
        print("Cost: ", xvals['g'] + xvals['h'])
        print("Heuristic: ", xvals['h'])
        print("Large disks:  ", X)
        print("Popped State: ", x)
        print()

        if xvals[0]["h"] < 1e-5:
            print("Done.")
            print("Large Disks:")
            print(X)
            path = expanded.path_to_root(X, x[:1], xvals)
            print("Solution")
            print(path)
            break
