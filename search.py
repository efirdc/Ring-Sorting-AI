from air_berlin import *


def search(X, x, h, search_width=1, log_interval=None, print_solution=False, cost_scale=None):
    if log_interval is None:
        log_interval = x.shape[1]

    expanded = Expanded()
    fringe = Fringe()

    xvals = get_xvals(x)
    xvals['h'] = h(x, xvals)

    fringe.push(x, xvals)

    for i in range(10000000):
        x, xvals = fringe.pop(search_width)

        best_xval = xvals[0]

        if i % log_interval == 0:
            print(f"Turn: {i}, expanded: {len(expanded)}, fringe: {len(fringe)}")
            print(f"Best g(x) + h(x) = {best_xval['g']} + {round(best_xval['h'], 2)}")
            print(f"State: {x[0]}\n")

        if best_xval["h"] < 1e-5:
            print("Done.")
            path = expanded.path_to_root(X, x[:1], xvals[:1])
            print(f"Solution length: {path.shape[0]}")

            if print_solution:
                print("Large Disks:")
                print(X)
                print("Solution")
                print(path)

            return path

        backtracked = expanded.contains(xvals)
        x = x[~backtracked]
        xvals = xvals[~backtracked]

        if x.shape[0] == 0:
            continue

        expanded.add(xvals, is_root=i == 0)
        x, xvals = expand(X, x, xvals)
        if cost_scale is not None:
            xvals["g"] *= cost_scale

        if x.shape[0] == 0:
            continue

        xvals['h'] = h(x, xvals)
        fringe.push(x, xvals)


def heuristic_search(X, x, h, search_width=1, log_interval=None, print_solution=False):
    if log_interval is None:
        log_interval = x.shape[1]

    expanded = Expanded()

    xvals = get_xvals(x)
    xvals['h'] = h(x, xvals)

    for i in range(10000000):
        best_xval = xvals[0]

        if i % log_interval == 0:
            print(f"Turn: {i}, expanded: {len(expanded)}")
            print(f"Best h(x) = {round(best_xval['h'], 2)}")
            #print(f"State: {x[0]}\n")

        if best_xval["h"] < 1e-5:
            print("Done.")
            path = expanded.path_to_root(X, x[:1], xvals[:1], max_depth=i*2)
            print(f"Solution length: {path.shape[0]}")

            if print_solution:
                print("Large Disks:")
                print(X)
                print("Solution")
                print(path)

            return path

        backtracked = expanded.contains(xvals)
        x = x[~backtracked]
        xvals = xvals[~backtracked]

        expanded.add(xvals, is_root=i == 0)
        x, xvals = expand(X, x, xvals)

        xvals['h'] = h(x, xvals)

        arg_best = xvals['h'].argsort()
        arg_best = arg_best[:min(search_width, arg_best.shape[0])]

        x = x[arg_best]
        xvals = xvals[arg_best]
