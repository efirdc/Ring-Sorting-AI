import matplotlib.pyplot as plt
import numpy as np


class RingPlot:
    def __init__(self, n, cmap):
        self.cmap = cmap

        theta = np.linspace(0, 2 * np.pi, n*n + 2)[:-1]

        self.xvals = np.cos(theta)
        self.yvals = np.sin(theta)

    def show(self, x):
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        plt.scatter(self.xvals, self.yvals, c=x, cmap=self.cmap)
        plt.show()
