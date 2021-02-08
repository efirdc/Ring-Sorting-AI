import time

from .game import *
from .expanded import *
from .fringe import *
from .heuristics import *
import numpy as np
import time


def search(X, x, h, fringe, expanded, search_width=1, cost_scale=None,
           verbose=True, log_interval=None, print_solution=False, print_state=False):
    if log_interval is None:
        log_interval = x.shape[1]

    xvals = get_xvals(x)
    xvals['h'] = h(X, x, xvals)

    fringe.push(x, xvals)

    start_time = time.time()

    i = 0
    while True:
        i += 1
        if len(fringe) == 0:
            raise ValueError("Ran out of nodes!")

        x, xvals = fringe.pop(search_width)

        best_xval = xvals[0]

        if verbose and i % log_interval == 0:
            time_elapsed = time.time() - start_time
            print(f"Turn: {i}, expanded: {len(expanded)}, fringe: {len(fringe)}, time elapsed: {time_elapsed}")
            print(f"Best g(x) + h(x) = {best_xval['g']} + {round(best_xval['h'], 2)}")
            if print_state:
                print(f"State: {x[0]}\n")

        candidate_x = xvals["h"] < 1e-5
        solved_x = is_solved(x[candidate_x])
        if np.any(solved_x):
            id = np.where(solved_x)[0][0]
            path = expanded.path_to_root(X, x[candidate_x][id:id + 1], xvals[candidate_x][id:id + 1])

            if verbose:
                print("Done.")
                print(f"Solution length: {path.shape[0] - 1}")
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

        expanded.add(xvals)
        x, xvals = expand(X, x, xvals)
        if cost_scale is not None:
            xvals["g"] *= cost_scale

        # This maybe shouldn't be necessary.
        if x.shape[0] == 0:
            continue

        xvals['h'] = h(X, x, xvals)
        fringe.push(x, xvals)


def ida_star(X, x, h, verbose=True, log_interval=None, print_solution=False):
    xvals = get_xvals(x)
    bound = h(X, x, xvals)
    path = [x[0]]

    while True:
        threshold = ida_search(X, path, 0, bound, h)
        
        if threshold is "FOUND":
            path = np.stack(path)
            if verbose:
                print("Done.")
                print(f"Solution length: {path.shape[0] - 1}")
                if print_solution:
                    print("Large Disks:")
                    print(X)
                    print("Solution")
                    print(path)

            return np.stack(path)
        
        if threshold is np.inf:
            return None

        bound = threshold
        print("New bound:", bound)


def ida_search(X, path, g, bound, h):
    x = path[-1]
    x = np.resize(x, (1, len(x)))
    
    xvals = get_xvals(x)
    huer = h(X, x, xvals)
    f = g + huer
    #print("Turn:", g)
    #print("h:", huer[0])
    #print("f:", f)
    #print(len(path))
    #print(np.stack(path))
    #print(path)

    if f > bound:
        return f

    if is_solved(path[-1]):
        return "FOUND"
    
    min = np.inf
    children, _ = expand(X, x, xvals)

    for child in children:
        if not any(np.array_equal(child, elem) for elem in path):
            path.append(child)

            threshold = ida_search(X, path, g + 1, bound, h)

            if threshold is "FOUND":
                return "FOUND"

            if threshold < min:
                min = threshold
            
            path.pop()

    return min


def is_goal(x):
    solutions = all_solved_states(len(x)-1)
    return any(np.array_equal(x, solution) for solution in solutions)
