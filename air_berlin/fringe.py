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
            heapq.heappush(self.fringe, (xvals['g'][i] + xvals['h'][i], (xvals['g'][i], xvals['h'][i], self.counter, x[i], xvals[i])))
            self.counter += 1

    def pop(self, num):
        xs = []
        xvals = []
        for i in range(num):
            if self.fringe:
                _, _, _, x, xval = heapq.heappop(self.fringe)[1]
                print(xval)
                xs.append(x)
                xvals.append(xval)
        xs = np.stack(xs)
        xvals = np.stack(xvals)
        return xs, xvals