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
import numbers

class ObjectiveFunction(object):
    """
    Objective function class.

    It's important to remember that by default, all optimizers of this
    framework works in minimization.
    """

    function_name = "Unknown"

    # EVAL ####################################################################

    def __call__(self, *pargs, **kargs):
        """
        Evaluate the point(s) x.

        If x is a vector (x.ndim=1), then return the value f(x) of the point x.
        This value y=f(x) is then a scalar number (not a vector ie. no
        multi-objective functions yet).

        If x is a matrix of dimension 2 (x.ndim=2), then return the value
        yi=f(xi) of each point xi in x.
        The x matrix is considered as following:
           number_of_points := x.shape[0]
           dimension_of_each_point := x.shape[1]
        with:
           x = [[x1],
                [x2],
                [x3],
                ...]
        For instance, the following matrix x means 3 points defined in R
        (1 dimension) have to be evaluated:
           x = [[ 2.],
                [ 3.],
                [ 4.]]
        For instance, the following matrix x means 3 points defined in RxR
        (2 dimensions) have to be evaluated:
           x = [[ 2., 2.],
                [ 3., 3.],
                [ 4., 4.]]
        Values yi=f(xi) are scalar numbers (not vectors ie. no multi-objective
        functions yet).
        """
        x = pargs[0]

        if x.ndim == 1:
            # Only one point ##########

            # Assert the number of elements of the vector x (ie. the dimension
            # of the point x) is equals to the dimension of the function (self).
            assert x.shape[0] == self.ndim, "x = " + str(x) + "; x.shape[0] = " + str(x.shape[0]) + "; self.ndim = " + str(self.ndim)

            # Get the value of the point x.
            y = self._eval_one_sample(x)

            # Assert y is a (scalar) number.
            assert isinstance(y, numbers.Number), "y = " + str(y)

        elif x.ndim == 2:
            # Multiple points #########

            number_of_points = x.shape[0]
            dimension_of_each_point = x.shape[1]

            # Assert the number of elements of the vector x (ie. the dimension
            # of the point x) is equals to the dimension of the function (self).
            # For instance, the following matrix x means 3 points defined in R
            # (1 dimension) have to be evaluated:
            #    x = [[ 2.],
            #         [ 3.],
            #         [ 4.]]
            # For instance, the following matrix x means 3 points defined in RxR
            # (2 dimensions) have to be evaluated:
            #    x = [[ 2., 2.],
            #         [ 3., 3.],
            #         [ 4., 4.]]
            assert dimension_of_each_point == self.ndim, "x.shape[1] = " + str(x) + "; self.ndim =" + str(self.ndim)

            y = self._eval_multiple_samples(x)

            # Assert there is one value yi=f(xi) for each point xi in x
            # and assert each yi is a scalar (not a vector).
            assert y.ndim == 2, "y.ndim = " + str(y)
            assert y.shape == (x.shape[0], 1), "y.shape = " + str(y.shape) + "x.shape = " + str(x.shape)

        else:
            raise Exception("Wrong value for x.")

        return y


    def _eval_one_sample(self, x):
        """
        Return the value y=f(x) of the function at the point x.
        This function must be redefined.

        The argument x must be a numpy vector (x.ndim=1), not a matrix.
        The returned value y=f(x) is a scalar number (not a vector ie. no
        multi-objective functions yet).

        This function should never be called by other functions than __call__()
        because all tests (assert) on arguments are made in __call__()
        (ie. this function assume arguments are well defined and doesn't test
        them). The main reason of this choice is to avoid to rewrite all
        tests (assert) in sub classes; all tests are written once for all
        in __call__().
        """
        raise NotImplementedError


    def _eval_multiple_samples(self, x):
        """
        Return the value yi=f(xi) of the function at the point xi in x.
        This function can be redefined to speedup computations.

        The argument x must a numpy matrix of dimension 2 (x.ndim=2).
        The returned the value yi=f(xi) of each point xi in x are scalar
        numbers (not vectors ie. no multi-objective functions yet).

        The x matrix given as argument is considered as following:
           number_of_points := x.shape[0]
           dimension_of_each_point := x.shape[1]
        with:
           x = [[x1],
                [x2],
                [x3],
                ...]
        For instance, the following matrix x means 3 points defined in R
        (1 dimension) have to be evaluated:
           x = [[ 2.],
                [ 3.],
                [ 4.]]
        For instance, the following matrix x means 3 points defined in RxR
        (2 dimensions) have to be evaluated:
           x = [[ 2., 2.],
                [ 3., 3.],
                [ 4., 4.]]

        This function should never be called by other functions than __call__()
        because all tests (assert) on arguments are made in __call__()
        (ie. this function assume arguments are well defined and doesn't test
        them). The main reason of this choice is to avoid to rewrite all
        tests (assert) in sub classes; all tests are written once for all
        in __call__().
        """
        y = []
        for xi in x:
            y.append(self._eval_one_sample(xi))
        return np.array(y).reshape([-1,1])


    # GRADIENT ################################################################

    def gradient(self, x):
        """
        Return the gradient of the function at one or multiple points.

        TODO: doc
        
        x can be a vector (a point) or a matrix (a tuple of points).
        """
        if x.ndim == 1:
            # Only one point ##########

            # Assert the number of elements of the vector x (ie. the dimension
            # of the point x) is equals to the dimension of the function (self).
            assert x.shape[0] == self.ndim, "x = " + str(x) + "; x.shape[0] = " + str(x.shape[0]) + "; self.ndim = " + str(self.ndim)

            # Get the gradient of the function at the point x.
            nabla = self._eval_one_gradient(x)

            # Assert nabla is a numpy vector with the same dimension than
            # point x.
            assert nabla.ndim == 1, "nabla.ndim = " + str(nabla)
            assert nabla.shape == x.shape, "nabla.shape = " + str(nabla.shape) + "x.shape = " + str(x.shape)

        elif x.ndim == 2:
            # Multiple points #########

            number_of_points = x.shape[0]
            dimension_of_each_point = x.shape[1]

            # Assert the number of elements of the vector x (ie. the dimension
            # of the point x) is equals to the dimension of the function (self).
            # For instance, the following matrix x means 3 points defined in R
            # (1 dimension) have to be evaluated:
            #    x = [[ 2.],
            #         [ 3.],
            #         [ 4.]]
            # For instance, the following matrix x means 3 points defined in RxR
            # (2 dimensions) have to be evaluated:
            #    x = [[ 2., 2.],
            #         [ 3., 3.],
            #         [ 4., 4.]]
            assert dimension_of_each_point == self.ndim, "x.shape[1] = " + str(x) + "; self.ndim =" + str(self.ndim)

            nabla = self._eval_multiple_gradients(x)

            # Assert there is one value nabla_i for each point x_i in x
            # and assert each nabla_i is a numpy vector with the same
            # dimension than points x_i.
            assert nabla.ndim == 2, "nabla.ndim = " + str(nabla)
            assert nabla.shape == x.shape, "nabla.shape = " + str(nabla.shape) + "x.shape = " + str(x.shape)

        else:
            raise Exception("Wrong value for x.")

        return nabla


    def _eval_one_gradient(self, point):
        """
        Return the gradient of the function at the point x.

        The argument x must be a numpy vector (x.ndim=1), not a matrix.
        The returned value nabla is a vector.

        This function should never be called by other functions than gradient()
        because all tests (assert) on arguments are made in gradient()
        (ie. this function assume arguments are well defined and doesn't test
        them). The main reason of this choice is to avoid to rewrite all
        tests (assert) in sub classes; all tests are written once for all
        in gradient().

        By default, this function does a numerical approximation of the
        gradient.

        This function can be redefined to speedup computations and get more
        accurate gradients (e.g. analytically computed gradient instead of the
        default numerically computed gradient).
        """
        return self._eval_one_num_gradient(point)


    def _eval_one_num_gradient(self, point):
        """
        Return the gradient of the function at the point x.
        It implements a numerical approximation of the gradient.

        The argument x must be a numpy vector (x.ndim=1), not a matrix.
        The returned value nabla is a vector.
        """
        if not hasattr(self, "delta"):
            self.delta = 0.001

        x = point
        nabla = np.zeros(self.ndim)

        for dim_index in range(self.ndim):
            # For each dimension of f
            delta_vec = np.zeros(self.ndim)
            delta_vec[dim_index] = self.delta

            y1 = self(x - delta_vec)
            y2 = self(x + delta_vec)

            nabla[dim_index] = (y2 - y1) / (2. * self.delta)

        return nabla


    def _eval_multiple_gradients(self, points):
        """
        Return the gradient of the function at multiple points.

        TODO: doc

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

        nabla = np.array(nabla_list)

        return nabla


    # STR #####################################################################

    def __str__(self):
        return "%s" % (self.function_name)


    # PLOT ####################################################################

    def plot(self, xmin=-1., xmax=1., xstep=0.02):
        """
        Plot the function for x in the range (xmin, xmax, xstep).
        This only works for 1D and 2D functions.
        """
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

            x1 = np.arange(xmin, xmax, xstep)
            x2 = np.arange(xmin, xmax, xstep)

            mesh_x1,mesh_x2 = np.meshgrid(x1, x2)

            # TODO: take advantage of meshgrid, for now, it's not optimized at
            #       all and it's not very well written
            z = np.zeros(mesh_x1.shape)         
            for x1i in range(z.shape[0]):
                for x2i in range(z.shape[1]):
                    point = np.array([mesh_x1[x1i, x2i], mesh_x2[x1i, x2i]])
                    z[x1i, x2i] = self(point)

            # PLOT DATA #################

            fig = plt.figure()

            #ax = axes3d.Axes3D(fig)
            #ax.plot_wireframe(mesh_x1, mesh_x2, z)

            ax = fig.gca(projection='3d')

            ax.plot_surface(mesh_x1, mesh_x2, z, rstride=5, cstride=5, alpha=0.3)
            cset = ax.contourf(mesh_x1, mesh_x2, z, zdir='z', offset=0, cmap=cm.coolwarm)

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
