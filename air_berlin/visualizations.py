import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
import os
import gc

from .utils import *


class RingPlot:
    def __init__(self, n, cmap):
        self.cmap = cmap

        self.theta = np.linspace(0, 2 * np.pi, n*n + 2)[:-1]

        self.xvals = np.cos(self.theta)
        self.yvals = np.sin(self.theta)

        self.codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
        ]

    def get_bezier(self, i, jump, m):
        jump_to = (i + jump) % m
        start = np.array([self.xvals[i], self.yvals[i]])
        end = np.array([self.xvals[jump_to], self.yvals[jump_to]])

        scale = {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7}[abs(jump)]
        verts = [start, start * scale, end * scale, end]
        return verts

    def get_fig(self, X, x, action, hval, cost, t):
        self.xvals = np.cos(self.theta)
        self.yvals = np.sin(self.theta)

        fig, ax = plt.subplots(figsize=(10, 10))

        zero_pos = np.where(x == 0)[0]

        m = X.shape[0]
        for i in range(m):
            big_jump = X[i]
            jumps = (-1, 1)

            if big_jump != 1:
                jumps += (-big_jump, big_jump)

            for jump in jumps:
                verts = self.get_bezier(i, jump, m)
                path = Path(verts, self.codes)

                if action != 0 and i == zero_pos:
                    if jump == action:
                        ax.add_patch(PathPatch(path, facecolor='none', lw=4, linestyle="-", edgecolor="red", zorder=2))
                    else:
                        ax.add_patch(PathPatch(path, facecolor='none', lw=2, linestyle="-", edgecolor="white", zorder=1))
                else:
                    ax.add_patch(PathPatch(path, facecolor='none', lw=1, linestyle="--", edgecolor="gray", zorder=0))

        if t != 0:
            jump_bez = self.get_bezier(zero_pos, action, m)
            zero_newpos = bezier(*jump_bez, t)
            swap_newpos = bezier(*jump_bez, 1 - t)
            self.xvals[zero_pos] = zero_newpos[0]
            self.yvals[zero_pos] = zero_newpos[1]
            self.xvals[(zero_pos + action) % m] = swap_newpos[0]
            self.yvals[(zero_pos + action) % m] = swap_newpos[1]

        non_zero = np.arange(m) != zero_pos
        plt.tick_params(which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.scatter(self.xvals[non_zero], self.yvals[non_zero], c=x[non_zero], cmap=self.cmap, s=160, zorder=3)
        ax.scatter(self.xvals[zero_pos], self.yvals[zero_pos], c="white", s=240, zorder=12)
        ax.set_facecolor("black")

        text_params = dict(transform=ax.transAxes, color="white", fontweight="bold", fontfamily='monospace')
        fig.text(0.01, 0.97, f"Cost:      {cost}", **text_params)
        fig.text(0.01, 0.94, f"Heuristic: {hval:.2f}", **text_params)

        return fig, ax

    def save_frames(self, X, x, hvals, interp_steps, save_path, start_from=0):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        actions = get_actions(x)

        N = x.shape[0]
        steps = start_from * (interp_steps + 1)
        for i in range(start_from, N):
            if i == (N - 1):
                action = 0
                interp_steps = 0
            else:
                action = actions[i]

            tvals = np.linspace(0, 1, 2 + interp_steps)[:-1]
            tvals = 3*tvals*tvals - 2*tvals*tvals*tvals
            for t in tvals:
                fig, ax = self.get_fig(X, x[i], action, hvals[i], i, t)
                plt.savefig(save_path + f"/frame{steps:06}.png", bbox_inches='tight', pad_inches=0)
                fig.clear()
                plt.close(fig)
                steps += 1

            if i % 50 == 0:
                plt.close("all")
                gc.collect()

