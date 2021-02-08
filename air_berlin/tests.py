from air_berlin import *

from os import listdir
from itertools import permutations


# Tries to show the heuristic is not admissible or not consistent with a counter-example
# Compares against optimal solutions found with breadth first search
# Parameters:
#    constructor - string so that heuristic can be constructed with eval(constructor)
def test_heuristic(constructor):
    optimal_path = "solutions/breadth_first_optimal/"
    optimal_solutions = listdir(optimal_path)

    for optimal_solution in optimal_solutions:
        load = np.load(optimal_path + optimal_solution, allow_pickle=True)
        X = load[0]
        path = load[1]

        n = int(np.sqrt(path.shape[1] - 1))
        h = eval(constructor)
        xvals = get_xvals(path)

        solution_length = path.shape[0] - 1
        hvals = h(X, path, xvals)

        true_costs = solution_length - np.arange(solution_length + 1)
        not_admissible = hvals > true_costs
        if np.any(not_admissible):
            for i in np.where(not_admissible)[0]:
                print(f"Heuristic is not admissible for state:\n{path[i]}")
                print(f"True cost to solution: {true_costs[i]} h(x): {hvals[i]}\n")

        not_consistent = hvals[:-1] > 1 + hvals[1:]
        if np.any(not_consistent):
            for i in np.where(not_consistent)[0]:
                print(f"Heuristic is not consistent for action:\n{path[i:i+2]}")
                print(f"The heuristic decreased from {hvals[i]} to {hvals[i+1]}\n")


# Test for any hash collisions in Expanded
def test_expanded():
    x = list(basic_solved_state(1, 3)[0])

    normal_set = set()
    expanded = Expanded()

    iterations = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for p in permutations(x):
        iterations += 1

        x = np.array(p)[np.newaxis]

        if p in normal_set:
            if expanded.contains(x)[0]:
                TN += 1
            else:
                FN += 1
        else:
            if not expanded.contains(x)[0]:
                TP += 1
            else:
                print(p)
                FP += 1

        normal_set.add(p)
        expanded.add(x)

        if iterations % 50000 == 0:
            print(f"iter={iterations} TN={TN}, FN={FN}, TP={TP}, FP={FP}")

        if iterations == 1000000:
            break
