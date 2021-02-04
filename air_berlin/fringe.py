import numpy as np
import heapq


# Keeps track of the Fringe in a priority queue
# Priority value based on cost from root plus the hueristic value
# Tie breakers are dealt with using self.counter
class Fringe:
    def __init__(self):
        self.fringe = []
        self.counter = 0

    def __len__(self):
        return len(self.fringe)

    def push(self, x, xvals):
        N = x.shape[0]
        for i in range(N):
            xval = xvals[i]
            heapq.heappush(self.fringe, (xval['g'] + xval['h'], (xval['h'], self.counter, x[i], xval)))
            self.counter += 1

    def pop(self, num):
        xs = []
        xvals = []
        for i in range(num):
            if self.fringe:
                _, _, x, xval = heapq.heappop(self.fringe)[1]
                xs.append(x)
                xvals.append(xval)
        xs = np.stack(xs)
        xvals = np.stack(xvals)
        return xs, xvals


class BeamFringe:
    def __init__(self):
        self.x = None
        self.xvals = None

    def __len__(self):
        if self.x is None:
            return 0
        return self.x.shape[0]

    def push(self, x, xvals):
        self.x = x
        self.xvals = xvals

    def pop(self, num):
        arg_best = self.xvals["h"].argsort()
        arg_best = arg_best[:min(num, self.x.shape[0])]
        out = self.x[arg_best], self.xvals[arg_best]

        self.x = None
        self.xvals = None

        return out
