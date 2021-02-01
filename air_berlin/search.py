from .game import *
from .expanded import *
from .fringe import *
from .heuristics import *
import numpy as np


def search(X, x, h, search_width=1, log_interval=None, print_solution=False, cost_scale=None, verbose=True):
    if log_interval is None:
        log_interval = x.shape[1]

    expanded = Expanded()
    fringe = Fringe()

    xvals = get_xvals(x)
    xvals['h'] = h(x, xvals)

    fringe.push(x, xvals)

    for i in range(10000000):
        if len(fringe) == 0:
            return None

        x, xvals = fringe.pop(search_width)

        best_xval = xvals[0]

        if verbose and i % log_interval == 0:
            print(f"Turn: {i}, expanded: {len(expanded)}, fringe: {len(fringe)}")
            print(f"Best g(x) + h(x) = {best_xval['g']} + {round(best_xval['h'], 2)}")
            print(f"State: {x[0]}\n")

        if best_xval["h"] < 1e-5:
            path = expanded.path_to_root(X, x[:1], xvals[:1])

            if verbose:
                print("Done.")
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

        # This maybe shouldn't be necessary.
        # Try raising a ValueError here and debugging.
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


def ida_star(X, x, h):
    xvals = get_xvals(x)
    bound = h(x, xvals)
    path = [x[0]]

    solutions = all_solved_states(len(x[0])-1)

    while True:
        threshold = ida_search(X, path, 0, bound, h, solutions)
        
        if threshold is "FOUND":
            return path
        
        if threshold is np.inf:
            return None

        bound = threshold
            

def ida_search(X, path, g, bound, h, solutions):
    x = path[-1]
    x = np.resize(x, (1, len(x)))
    print("Large:  ", X)
    print("Small: ", x)

    xvals = get_xvals(x)
    f = g + h(x, xvals)
    print("f: ", f)
    print("bound: ", bound)

    if f > bound:
        return f

    if is_goal(solutions, x):
        return "FOUND"
    
    min = np.inf
    children, _ = expand(X, x, xvals)

    for child in children:
        if not any(np.array_equal(child, elem) for elem in path):
            path.append(child)
            threshold = ida_search(X, path, g + 1, bound, h, solutions)

            if threshold is "FOUND":
                return "FOUND"

            if threshold < min:
                min = threshold
            
            path = path[:-1]

    return min


def is_goal(solutions, x):
    return any(np.array_equal(x, solution) for solution in solutions)