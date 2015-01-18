# -*- coding: utf-8 -*-

# Copyright (c) 2013,2014,2015 Jérémie DECOCK (http://www.jdhp.org)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import warnings

class Optimizer(object):

    def __init__(self):
        self.log = Log()

    def plotSamples(self, x, y, nabla=None, objective_function=None, save_filename=None):
        import matplotlib.pyplot as plt

        assert x.ndim == 2, x.ndim
        assert y.ndim == 2, y.ndim
        assert y.shape[1] == 1, y.shape

        if x.shape[1]==1:
            # 1D case

            fig = plt.figure(figsize=(16.0, 10.0))
            ax = fig.add_subplot(111)

            # PLOT THE OBJECTIVE FUNCTION 

            if objective_function is not None:
                # BUILD DATA

                xmin = -1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                xmax =  1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                ymin = -1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                ymax =  1    # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS
                xstep = 0.05 # TODO: AUTOMATICALY COMPUTE THIS WITH VISITED POINTS

                x_vec = np.arange(xmin, xmax, xstep)
                y_vec = objective_function(x_vec.reshape([-1, 1]))
                ax.plot(x_vec, y_vec, "-", label="objective function")

            # PLOT VISITED POINTS
            ax.plot(x[:,0], y, ".", label="visited points")
            
            # PLOT THE BEST VISITED POINTS
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.plot(x_min, y_min, "xr")

            # PLOT GRADIENT OF VISITED POINTS
            if nabla is not None:
                pass

            # TITLE AND LABELS
            ax.set_title('Visited points', fontsize=20)
            ax.set_xlabel(r"$x$", fontsize=32)
            ax.set_ylabel(r"$f(x)$", fontsize=32)

            # LEGEND
            ax.legend(loc='lower right', fontsize=20)

            # SAVE FILES ######################
            if save_filename is not None:
                filename = save_filename + ".pdf"
                plt.savefig(filename)

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

                range_x1 = np.arange(xmin, xmax, xstep)
                range_x2 = np.arange(xmin, xmax, xstep)

                mesh_x1,mesh_x2 = np.meshgrid(range_x1, range_x2)

                # TODO: take advantage of meshgrid, for now, it's not optimized at
                #       all and not very well written
                z = np.zeros(mesh_x1.shape)         
                for x1i in range(z.shape[0]):
                    for x2i in range(z.shape[1]):
                        point = np.array([mesh_x1[x1i, x2i], mesh_x2[x1i, x2i]])
                        z[x1i, x2i] = objective_function(point)

                # PLOT
                ax.plot_surface(mesh_x1, mesh_x2, z, rstride=5, cstride=5, alpha=0.3)
                cset = ax.contourf(mesh_x1, mesh_x2, z, zdir='z', offset=0, cmap=cm.coolwarm)

            # PLOT VISITED POINTS
            ax.scatter(x[:,0], x[:,1], y, color='b')
            
            # PLOT THE BEST VISITED POINT
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.scatter(x_min[0], x_min[1],  y_min, color='r')

            # PLOT GRADIENT OF VISITED POINTS
            if nabla is not None:
                pass

            # TITLE AND LABELS
            ax.set_title('Visited points', fontsize=20)
            ax.set_xlabel(r'$x_1$', fontsize=32)
            ax.set_ylabel(r'$x_2$', fontsize=32)
            ax.set_zlabel(r'$f(x)$', fontsize=32)

            # SAVE FILES ######################
            if save_filename is not None:
                filename = save_filename + ".pdf"
                plt.savefig(filename)

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
