from .game import *


# Keeps track of the explored tree using a hash based graph
# The self.vertices dictionary maps hash(x) -> x
# while the self.edges dictionary maps hash(x) -> hash(x_parent)
class Expanded:
    def __init__(self):
        self.vertices = {}
        self.edges = {}

    def __len__(self):
        return len(self.vertices)

    def add(self, xvals):
        new_vertices = {xv["hash"]: xv["prev_action"] for xv in xvals}
        self.vertices.update(new_vertices)

        is_root = xvals["parent_hash"][0] == 0
        if not is_root:
            new_edges = {xv["hash"]: xv["parent_hash"] for xv in xvals}
            self.edges.update(new_edges)

    def contains(self, xvals):
        return np.array([xv["hash"] in self.vertices for xv in xvals])

    def path_to_root(self, X, x, xvals, max_depth=10000000):
        out = [x]

        hsh = xvals[0]["parent_hash"]
        a = xvals[0]["prev_action"]

        depth = 0
        while True:
            depth += 1
            if depth >= max_depth:
                raise TimeoutError(f"Reached max depth of {max_depth}. Probably a cycle in the graph.")

            x, xvals = apply_action(X, x.copy(), xvals, -a)
            out.append(x)

            if hsh not in self.edges:
                break

            a = self.vertices[hsh]
            hsh = self.edges[hsh]

        out.reverse()
        return np.concatenate(out)