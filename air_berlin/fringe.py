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
            heapq.heappush(self.fringe, (xval['g'] + xval['h'], (self.counter, x[i], xval)))
            self.counter += 1

    def pop(self, num):
        xs = []
        xvals = []
        for i in range(num):
            if self.fringe:
                _, x, xval = heapq.heappop(self.fringe)[1]
                xs.append(x)
                xvals.append(xval)
        xs = np.stack(xs)
        xvals = np.stack(xvals)
        return xs, xvals