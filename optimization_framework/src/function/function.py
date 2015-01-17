# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['ObjectiveFunction']

import numpy as np

class ObjectiveFunction(object):

    # EVAL ####################################################################

    def __call__(self, *pargs, **kargs):
        x = pargs[0]

        if x.ndim == 1:
            # Only one point
            assert x.shape[0] == self.ndim, x
            y = self._eval_one_sample(x)
            #assert type(y) == type(float), type(y)
        else:
            # Multiple points
            assert x.shape[1] == self.ndim, x
            y = self._eval_multiple_samples(x)
            assert y.ndim == 2, y
            assert y.shape == (x.shape[0], 1), y.shape

        return y


    def _eval_one_sample(self, x):
        """
        Return the value of the function at the point x.
        """
        raise NotImplementedError


    def _eval_multiple_samples(self, x):
        """
        This function can be redefined to speedup computations
        """
        y = []
        for xi in x:
            y.append(self._eval_one_sample(xi))
        return np.array(y).reshape([-1,1])


    # GRADIENT ################################################################

    def gradient(self, x):
        """
        Return the gradient of the function at one or multiple points.
        
        x can be a vector (a point) or a matrix (a tuple of points).
        """
        if x.ndim == 1:
            # Only one point
            assert x.shape[0] == self.ndim, x
            y = self._eval_one_gradient(x)
            #assert type(y) == type(float), type(y)
        else:
            # Multiple points
            assert x.shape[1] == self.ndim, x
            y = self._eval_multiple_gradients(x)
            assert y.ndim == 2, y
            assert y.shape == (x.shape[0], 1), y.shape

        return y


    def _eval_one_gradient(self, point):
        """
        Return the gradient of the function at the point x.
        By default, it does a numerical approximation of the gradient.

        This function can be redefined to speedup computations and get more
        accurate gradients.
        """
        return self._eval_one_num_gradient(point)


    def _eval_one_num_gradient(self, point):
        """
        Return the gradient of the function at the point x.
        It implements a numerical approximation of the gradient.
        """
        if not hasattr(self, "delta"):
            self.delta = 0.001

        x = point
        nabla = np.zeros(self.ndim)

        for dim_index in range(self.ndim):
            delta_vec = np.zeros(self.ndim)
            delta_vec[dim_index] = self.delta

            y1 = self._eval_one_sample(x - delta_vec)
            y2 = self._eval_one_sample(x + delta_vec)

            nabla[dim_index] = y2 - y1

        return nabla


    def _eval_multiple_gradients(self, points):
        """
        Return the gradient of the function at multiple points.

        The argument "points" is a np.array.
        For instance,
           points = [[1, 2, 3],
                     [4, 5, 6]]
        contains 2 points (1, 2, 3) and (4, 5, 6).

        This function can be redefined to speedup computations.
        """
        nabla_list = []
        for xi in points:
            # xi is a point in points
            nabla_list.append(self._eval_one_gradient(xi))
        return np.array(nabla_list).reshape([-1,1])


    # STR #####################################################################

    def __str__(self):
        return "%s" % (self.__class__)
        #return "Objective function"


    # PLOT ####################################################################

    def plot(self, xmin=-1., xmax=1., xstep=0.01):
        if self.ndim == 1:

            # 1D FUNCTIONS

            import matplotlib.pyplot as plt

            x_vec = np.arange(xmin, xmax, xstep)
            y_vec = self(x_vec.reshape([-1, 1]))

            try:
                label = self.label
            except:
                label = "f(x)"

            fig = plt.figure(figsize=(16.0, 10.0))
            ax = fig.add_subplot(111)
            ax.plot(x_vec, y_vec, "-", label=label)

            # TITLE AND LABELS
            ax.set_title('Objective function\n$' + str(self) + '$', fontsize=20)
            ax.set_xlabel(r"$x$", fontsize=32)
            ax.set_ylabel(r"$f(x)$", fontsize=32)

            # LEGEND
            ax.legend(loc='lower right', fontsize=20)

            # SAVE FILES ######################
            #filename = label + ".pdf"
            #plt.savefig(filename)

            # PLOT ############################
            plt.show()

        elif self.ndim == 2:

            # 2D FUNCTIONS

            import matplotlib.pyplot as plt

            from mpl_toolkits.mplot3d import axes3d
            from matplotlib import cm

            # BUILD DATA ################

            x = np.arange(xmin, xmax, xstep)
            y = np.arange(xmin, xmax, xstep)

            xx,yy = np.meshgrid(x, y)

            # TODO: take advantage of meshgrid, for now, it's not optimized at
            #       all and not very well written
            z = np.zeros(xx.shape)         
            for xi in range(z.shape[0]):
                for yi in range(z.shape[1]):
                    point = np.array([xx[xi, yi], yy[xi, yi]])
                    z[xi, yi] = self(point)

            # PLOT DATA #################

            fig = plt.figure()

            #ax = axes3d.Axes3D(fig)
            #ax.plot_wireframe(xx, yy, z)

            ax = fig.gca(projection='3d')

            ax.plot_surface(xx, yy, z, rstride=5, cstride=5, alpha=0.3)
            cset = ax.contourf(xx, yy, z, zdir='z', offset=0, cmap=cm.coolwarm)

            # TITLE AND LABELS
            ax.set_title('Objective function\n$' + str(self) + '$', fontsize=20)
            ax.set_xlabel(r'$x_1$', fontsize=32)
            ax.set_ylabel(r'$x_2$', fontsize=32)
            ax.set_zlabel(r'$f(x)$', fontsize=32)

            # SHOW ############################
            plt.show()
        else:
            pass


# TODO: doit être un objet qui permet de connaître:
# - le dommaine de définition de x
#   - continu / discret ?
#   - contraint ou non
# - le nombre de dimensions de x
# - minimisation ou maximisation
