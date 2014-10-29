# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['Optimizer', 'Log']

import numpy as np
import warnings

class Optimizer(object):

    def __init__(self):
        self.log = Log()

    def plotSamples(self, x, y):
        import matplotlib.pyplot as plt

        assert x.ndim == 2, x.ndim
        assert y.ndim == 2, y.ndim
        assert y.shape[1] == 1, y.shape

        if x.shape[1]==1:
            # 1D case
            fig = plt.figure(figsize=(16.0, 10.0))
            ax = fig.add_subplot(111)

            label = "samples"
            ax.plot(x[:,0], y, ".", label=label)
            
            # PLOT BEST SAMPLE
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.plot(x_min, y_min, ".r")

            # TITLE AND LABELS
            ax.set_title("Samples", fontsize=20)
            ax.set_xlabel(r"$x$", fontsize=32)
            ax.set_ylabel(r"$f(x)$", fontsize=32)

            # LEGEND
            ax.legend(loc='lower right', fontsize=20)

            # PLOT
            plt.show()
        elif x.shape[1]==2:
            from mpl_toolkits.mplot3d import axes3d

            # 2D case
            fig = plt.figure()
            ax = axes3d.Axes3D(fig)
            ax.scatter(x[:,0], x[:,1], y, color='b')
            
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.scatter(x_min[0], x_min[1],  y_min, color='r')

            plt.show()
        else:
            warnings.warn("Cannot plot samples: too many dimensions.")

    def plotCosts(self, y):
        import matplotlib.pyplot as plt

        assert y.shape[1]==1

        label = "value"

        fig = plt.figure(figsize=(16.0, 10.0))
        ax = fig.add_subplot(111)
        ax.plot(y, "-", label=label)

        # TITLE AND LABELS
        ax.set_title("Value over iterations", fontsize=20)
        ax.set_xlabel(r"iteration $i$", fontsize=32)
        ax.set_ylabel(r"$f(x)$", fontsize=32)

        # LEGEND
        ax.legend(loc='lower right', fontsize=20)

        # PLOT
        plt.show()


class Log:
    def __init__(self):
        self.data = {}
