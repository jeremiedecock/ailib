# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['Optimizer', 'Log']

import numpy as np
import warnings

class Optimizer(object):

    def __init__(self):
        self.log = Log()

    def plotSamples(self, x, y, objective_function=None):
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
            # 2D case

            from mpl_toolkits.mplot3d import axes3d
            if objective_function is not None:
                from mpl_toolkits.mplot3d import axes3d
                from matplotlib import cm

            fig = plt.figure()

            if objective_function is not None:
                ax = fig.gca(projection='3d')
            else:
                ax = axes3d.Axes3D(fig)

            # PLOT THE OBJECTIVE FUNCTION 

            if objective_function is not None:
                # BUILD DATA

                xmin = -1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                xmax =  1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                ymin = -1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                ymax =  1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                xstep = 0.05 # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS

                range_x = np.arange(xmin, xmax, xstep)
                range_y = np.arange(xmin, xmax, xstep)

                mesh_x,mesh_y = np.meshgrid(range_x, range_y)

                # TODO: take advantage of meshgrid, for now, it's not optimized at
                #       all and not very well written
                z = np.zeros(mesh_x.shape)         
                for xi in range(z.shape[0]):
                    for yi in range(z.shape[1]):
                        point = np.array([mesh_x[xi, yi], mesh_y[xi, yi]])
                        z[xi, yi] = objective_function(point)

                # PLOT
                ax.plot_surface(mesh_x, mesh_y, z, rstride=5, cstride=5, alpha=0.3)
                cset = ax.contourf(mesh_x, mesh_y, z, zdir='z', offset=0, cmap=cm.coolwarm)

            # PLOT VISITED POINTS

            ax.scatter(x[:,0], x[:,1], y, color='b')
            
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.scatter(x_min[0], x_min[1],  y_min, color='r')

            # TITLE AND LABELS
            ax.set_title('Visited points', fontsize=20)
            ax.set_xlabel(r'$x_1$', fontsize=32)
            ax.set_ylabel(r'$x_2$', fontsize=32)
            ax.set_zlabel(r'$f(x)$', fontsize=32)

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
