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
    """
    Optimizer class.
    By default, all optimizers works in minimization.
    """

    optimizer_name = "unknown"

    def __init__(self):
        self.log = Log()

    def plotSamples(self, x, y, nabla=None, objective_function=None, save_filename=None):
        """
        Plot the objective function for x in the range (xmin, xmax, xstep) and
        the evaluated points.
        This only works for 1D and 2D functions.
        """
        import matplotlib.pyplot as plt

        #print("DEBUG plotSamples(): x =", x)
        #print("DEBUG plotSamples(): type(x) =", type(x))
        #print("DEBUG plotSamples(): y =", y)
        #print("DEBUG plotSamples(): type(y) =", type(y))

        assert x.ndim == 2, x.ndim
        assert y.ndim == 1, y.ndim
        assert y.shape[0] == x.shape[0], y.shape

        if x.shape[1]==1:
            # 1D case

            fig = plt.figure(figsize=(16.0, 10.0))
            ax = fig.add_subplot(111)

            # PLOT THE OBJECTIVE FUNCTION 

            if objective_function is not None:
                # BUILD DATA

                assert objective_function.domain_min.ndim == 1
                assert objective_function.domain_max.ndim == 1
                assert objective_function.domain_min.shape[0] == 1
                assert objective_function.domain_max.shape[0] == 1

                xmin = objective_function.domain_min[0]
                xmax = objective_function.domain_max[0]
                assert xmin < xmax

                xstep = (xmax - xmin) / 1000.

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

                assert objective_function.domain_min.ndim == 1
                assert objective_function.domain_max.ndim == 1
                assert objective_function.domain_min.shape[0] == 2
                assert objective_function.domain_max.shape[0] == 2

                x1min = objective_function.domain_min[0]
                x1max = objective_function.domain_max[0]
                assert x1min < x1max

                x2min = objective_function.domain_min[1]
                x2max = objective_function.domain_max[1]
                assert x2min < x2max

                x1step = (x1max - x1min) / 200.
                x2step = (x2max - x2min) / 200.

                range_x1 = np.arange(x1min, x1max, x1step)
                range_x2 = np.arange(x2min, x2max, x2step)

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
        """
        Plot the evolution of point's cost evaluated during iterations.
        """
        import matplotlib.pyplot as plt

        assert y.ndim == 1, "y.ndim = " + str(y.ndim)

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
    # TODO: this class is not used yet ?
    def __init__(self):
        self.data = {}
