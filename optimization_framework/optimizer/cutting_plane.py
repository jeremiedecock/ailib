#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2015 Jérémie DECOCK (http://www.jdhp.org)

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

import cvxopt
import cvxopt.solvers

# Or:
# from cvxopt import matrix, solvers

import numpy as np
import numbers

# TODO: improve this ?
if __name__ == '__main__':
    import optimizer
else:
    from . import optimizer

class Optimizer(optimizer.Optimizer):
    """
    Cutting plane method.

    See (list various ways to solve LP with Python): http://en.wikibooks.org/wiki/GLPK/Python

    A bit off-topic but interesting: http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
    """

    optimizer_name = "cutting plane"

    def optimize(self, objective_function, num_iterations=10):

        dmin = objective_function.domain_min
        dmax = objective_function.domain_max
        
        # Get the first point
        x = np.random.uniform(dmin, dmax, objective_function.ndim)

        # Init history lists
        x_history_list = np.zeros([num_iterations, objective_function.ndim])
        nabla_history_list = np.zeros([num_iterations, objective_function.ndim])

        # Main loop: for each iteration do...
        for sample_index in range(num_iterations):

            # Compute the gradient of objective_function at x
            nabla = objective_function.gradient(x)
            coef = .1  # TODO!!! : http://fr.wikipedia.org/wiki/Algorithme_du_gradient + http://fr.wikipedia.org/wiki/Recherche_lin%C3%A9aire  +  http://fr.wikipedia.org/wiki/Algorithme_%C3%A0_r%C3%A9gions_de_confiance

            x = x - coef * nabla

            # Keep an history of x and nabla to plot things...
            x_history_list[sample_index, :] = x
            nabla_history_list[sample_index, :] = nabla

            #print("DEBUG optimize(): xi =", x)
            #print("DEBUG optimize(): type(xi) =", type(x))

        y_history_list = objective_function(x_history_list)
        self.plotSamples(x_history_list, y_history_list, objective_function=objective_function)
        self.plotCosts(y_history_list)

        #print("DEBUG optimize(): x_history_list =", x_history_list)
        #print("DEBUG optimize(): type(x_history_list) =", type(x_history_list))
        #print("DEBUG optimize(): y_history_list =", y_history_list)
        #print("DEBUG optimize(): type(y_history_list) =", type(y_history_list))
        #print("DEBUG optimize(): x =", x)

        return x


    def getCutsFunctionList(self, x_array, y_array, nabla_array):
        """
        TODO...

        The argument x_array must a numpy array of dimension 2 (x_array.ndim=2).
        The returned value yi=f(xi) of each point xi in x_array are scalar
        numbers (not vectors i.e. no multi-objective functions yet).
        Therefore, the returned value y_array must have y_array.ndim=1 and
        y_array.shape[0]=x_array.shape[0].

        The x_array array given as argument is considered as following:
           number_of_points := x_array.shape[0]
           dimension_of_each_point := x_array.shape[1]
        with:
           x_array = [[x1],
                      [x2],
                      [x3],
                      ...]
        For instance, the following array x_array means 3 points defined in R
        (1 dimension) have to be evaluated:
           x_array = [[ 2.],
                      [ 3.],
                      [ 4.]]
        For instance, the following array x_array means 3 points defined in RxR
        (2 dimensions) have to be evaluated:
           x_array = [[ 2., 2.],
                      [ 3., 3.],
                      [ 4., 4.]]

        The nabla_array array given as argument is considered as following:
           number_of_gradients := nabla_array.shape[0]
           dimension_of_each_gradient := nabla_array.shape[1]
        with:
           nabla_array = [[nabla_1],
                          [nabla_2],
                          [nabla_3],
                          ...]

        TODO:...

        """

        # Assert the number of elements of the vector x_array (i.e. the dimension
        # of the point x_array) is equals to the dimension of the function (self).
        # For instance, the following numpy array x_array means 3 points defined in R
        # (1 dimension) have to be evaluated:
        #    x_array = [[ 2.],
        #               [ 3.],
        #               [ 4.]]
        # For instance, the following numpy array x_array means 3 points defined in RxR
        # (2 dimensions) have to be evaluated:
        #    x_array = [[ 2., 2.],
        #               [ 3., 3.],
        #               [ 4., 4.]]
        assert x_array.ndim == 2, "x_array.ndim = " + str(x_array.ndim)      # There are multiple points in x_array
        number_of_points = x_array.shape[0]
        dimension_of_each_point = x_array.shape[1]

        # Assert there is one value yi=f(xi) for each point xi in x_array
        # and assert each yi is a scalar number (not a numpy array).
        assert y_array.ndim == 1, "y_array.ndim = " + str(y_array.ndim)
        assert y_array.shape[0] == number_of_points, "y_array.shape = " + str(y_array.shape) + "x_array.shape = " + str(x_array.shape)

        # Assert nabla_array is a numpy array of dimension 2 (i.e. a matrix) with
        # the same number of elements (dimension) than point x_array.
        assert nabla_array.ndim == 2, "nabla_array.ndim = " + str(nabla_array.ndim)
        assert nabla_array.shape[0] == number_of_points, "nabla_array.shape = " + str(nabla_array.shape) + "x_array.shape = " + str(x_array.shape)
        assert nabla_array.shape[1] == dimension_of_each_point, "nabla_array.shape = " + str(nabla_array.shape) + "x_array.shape = " + str(x_array.shape)

        cut_list = []

        # BUG: closures/muttables/definition scope
        #    for (x_i, y_i, nabla_i) in zip(x_array, y_array, nabla_array):
        #        def cut_function(x):
        #            y = np.dot(nabla_i, (x-x_i)) + y_i
        #            return y
        #        cut_list.append(cut_function)
        # SEE: very interresting comments in http://math.andrej.com/2009/04/09/pythons-lambda-is-broken/comment-page-1/
        # BUG: a is a reference (=2 eventually)
        #  l = [lambda : a for a in range(3)]
        #  print([f() for f in l])
        # A SOLUTION:
        #  l = [lambda b=a : b for a in range(3)]
        #  print([f() for f in l])
        # OR
        #  l = [lambda x, _a=a : _a*x for a in range(3)]
        #  print([f(2) for f in l])
        # EXPLANATION:
        # "default values are calculated at definition time, not run time. So
        # “_a=a” effectively curries the value of “a” at definition time."
        for (x_i, y_i, nabla_i) in zip(x_array, y_array, nabla_array):
            # Check inputs
            assert x_i.ndim == 1
            assert x_i.shape[0] == dimension_of_each_point
            assert isinstance(y_i, numbers.Number)
            assert nabla_i.ndim == 1
            assert nabla_i.shape[0] == dimension_of_each_point

            def cut_function(x, _x=x_i, _y=y_i, _nabla=nabla_i):
                # Assert x is a numpy vector
                assert x.ndim == 1
                assert x.shape[0] == _x.shape[0]

                y = np.dot(_nabla, (x-_x)) + _y
            
                # Assert y is a (scalar) number.
                assert isinstance(y, numbers.Number), "y = " + str(y)

                return y

            cut_list.append(cut_function)

        return cut_list


    def getMinimumOfCuts(self, x_array, y_array, nabla_array, cut_list):
        """
        TODO

        The nabla_array array given as argument is considered as following:
           number_of_gradients := nabla_array.shape[0]
           dimension_of_each_gradient := nabla_array.shape[1]
        with:
           nabla_array = [[nabla_1],
                          [nabla_2],
                          [nabla_3],
                          ...]
        """

        assert x_array.ndim == 2, "x_array.ndim = " + str(x_array.ndim)      # There are multiple points in x_array
        number_of_points = x_array.shape[0]
        dimension_of_each_point = x_array.shape[1]

        # Assert there is one value yi=f(xi) for each point xi in x_array
        # and assert each yi is a scalar number (not a numpy array).
        assert y_array.ndim == 1, "y_array.ndim = " + str(y_array.ndim)
        assert y_array.shape[0] == number_of_points, "y_array.shape = " + str(y_array.shape) + "x_array.shape = " + str(x_array.shape)

        # Assert nabla_array is a numpy array of dimension 2 (i.e. a matrix) with
        # the same number of elements (dimension) than point x_array.
        assert nabla_array.ndim == 2, "nabla_array.ndim = " + str(nabla_array.ndim)
        assert nabla_array.shape[0] == number_of_points, "nabla_array.shape = " + str(nabla_array.shape) + "x_array.shape = " + str(x_array.shape)
        assert nabla_array.shape[1] == dimension_of_each_point, "nabla_array.shape = " + str(nabla_array.shape) + "x_array.shape = " + str(x_array.shape)

        # Assert there is one cut function per point.
        assert len(cut_list) == number_of_points, "len(cut_list) = " + str(len(cut_list)) + "x_array.shape = " + str(x_array.shape)

        # LP matrices #################

        number_of_variables = dimension_of_each_point + 1     # A dimension is added
        
        # Set the column array containing the objective function coefficients.
        # c = (0 0 0 ... 0 0 1)
        np_c = np.zeros(number_of_variables)
        np_c[-1] = 1.                          # the last element of c is equal to 1.

        # Set the matrix containing the constraints coefficients.
        # A = [[nabla_11, nabla_12, ..., nabla_1n, -1],
        #      [nabla_21, nabla_22, ..., nabla_2n, -1],
        #      ...
        #      [nabla_m1, nabla_m2, ..., nabla_mn, -1]]
        ones_col = (-1. * np.ones(number_of_points)).reshape([-1, 1])
        np_A = np.concatenate((nabla_array, ones_col), 1)

        # Set the column array containing the right-hand side value for each
        # constraint in the constraint matrix.
        # b = [-cut_1(0), -cut_2(0), ..., -cut_m(0)]
        orig_array = np.zeros(dimension_of_each_point)
        np_b = np.array([-cut(orig_array) for cut in cut_list])

        print("np_c =", np_c)
        print("np_A =", np_A)
        print("np_b =", np_b)

        # Convert numpy arrays to cvxopt matrices
        c = cvxopt.matrix(np_c)
        A = cvxopt.matrix(np_A)
        b = cvxopt.matrix(np_b)

        print("c =", c)
        print("A =", A)
        print("b =", b)

        # Optimize...
        sol = cvxopt.solvers.lp(c, A, b)

        # Get and return the solution
        xstar = sol['x']
        np_xstar = np.array(xstar)
        #np_xstar = np.array(xstar)[:-1]

        return np_xstar


    def plotSamples(self, x, y, nabla=None, cut_list=None, objective_function=None, save_filename=None, minimum_of_cuts=None):
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

                xstep = (xmax - xmin) / 100.

                x_vec = np.arange(xmin, xmax, xstep)
                y_vec = objective_function(x_vec.reshape([-1, 1]))

                ax.plot(x_vec, y_vec, "-", label="objective function")

            # PLOT GRADIENTS
            #if nabla is not None:
            #    # TODO
            #    x_delta = 0.5
            #    ax.plot(x[:,0] + x_delta, y + nabla[:,0] * x_delta, ".") # , label="gradients")
            #    pass

            # PLOT CUTS
            if cut_list is not None:
                xmin = objective_function.domain_min[0]
                xmax = objective_function.domain_max[0]
                assert xmin < xmax

                x_to_plot = np.array([xmin, xmax]) # TODO: 1D, require objective_function

                for cut in cut_list:
                    y_to_plot = np.array([cut(np.array([xmin])), cut(np.array([xmax]))])
                    ax.plot(x_to_plot, y_to_plot, "-g") # , label="gradients")

                ax.set_ylim(min(y), max(y))

            # PLOT MAX CUTS (the heuristic function)
            if cut_list is not None:
                xmin = objective_function.domain_min[0]
                xmax = objective_function.domain_max[0]
                assert xmin < xmax

                xstep = (xmax - xmin) / 1000.

                x_to_plot = np.arange(xmin, xmax, xstep)
                y_to_plot = np.array([ max([ cut(np.array([x_i])) for cut in cut_list ]) for x_i in x_to_plot ])

                ax.plot(x_to_plot, y_to_plot, "-r", linewidth=2, label="heuristic")


            # PLOT VISITED POINTS
            ax.plot(x[:,0], y, ".", label="visited points")
            
            # PLOT THE BEST VISITED POINTS
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.plot(x_min, y_min, "xr")

            # PLOT THE MINIMUM OF CUTS
            if minimum_of_cuts is not None:
                x_min = minimum_of_cuts[:-1]
                y_min = minimum_of_cuts[-1]
                ax.plot(x_min, y_min, "xr", markersize=10., markeredgewidth=3.)

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
                ax.plot_surface(mesh_x1, mesh_x2, z, rstride=5, cstride=5, linewidth=0.2, alpha=0.2)
                #cset = ax.contourf(mesh_x1, mesh_x2, z, zdir='z', offset=0, cmap=cm.coolwarm)

            # PLOT MAX CUTS (the heuristic function)
            if cut_list is not None:
                x1min = objective_function.domain_min[0]
                x1max = objective_function.domain_max[0]
                assert x1min < x1max

                x2min = objective_function.domain_min[1]
                x2max = objective_function.domain_max[1]
                assert x2min < x2max

                x1step = (x1max - x1min) / 100.
                x2step = (x2max - x2min) / 100.

                range_x1 = np.arange(x1min, x1max, x1step)
                range_x2 = np.arange(x2min, x2max, x2step)

                mesh_x1,mesh_x2 = np.meshgrid(range_x1, range_x2)

                # TODO: take advantage of meshgrid, for now, it's not optimized at
                #       all and not very well written
                z = np.zeros(mesh_x1.shape)         
                for x1i in range(z.shape[0]):
                    for x2i in range(z.shape[1]):
                        point = np.array([mesh_x1[x1i, x2i], mesh_x2[x1i, x2i]])
                        z[x1i, x2i] = max([ cut(point) for cut in cut_list ])

                # PLOT
                ax.plot_surface(mesh_x1, mesh_x2, z, rstride=5, cstride=5, alpha=0.5, color="r")
                #cset = ax.contourf(mesh_x1, mesh_x2, z, zdir='z', offset=0, cmap=cm.coolwarm)


            # PLOT VISITED POINTS
            ax.scatter(x[:,0], x[:,1], y, color='b')
            
            # PLOT THE BEST VISITED POINT
            x_min = x[y.argmin(), :]
            y_min = y.min()
            ax.scatter(x_min[0], x_min[1],  y_min, color='r')

            # PLOT THE MINIMUM OF CUTS
            if minimum_of_cuts is not None:
                x_min = minimum_of_cuts[:-1]
                y_min = minimum_of_cuts[-1]
                ax.scatter(x_min[0], x_min[1],  y_min, color='g')

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


# TEST ########################################################################

def test():

    from function.sphere import Function
    f1 = Function(1)
    f2 = Function(2)

    optimizer = Optimizer()

    # 1D PLOT ###################################

    x_hist_array = np.array([[-1.],
                             [0.5]])
    #x_hist_array = np.array([[-1.],
    #                         [0.],
    #                         [0.5]])
    #x_hist_array = np.array([[-1.],
    #                         [-0.5],
    #                         [0.],
    #                         [0.5],
    #                         [1.]])
    y_hist_array = f1(x_hist_array)
    nabla_hist_array = f1.gradient(x_hist_array)

    print("x_hist_array =", x_hist_array)
    print("y_hist_array =", y_hist_array)
    print("nabla_hist_array =", nabla_hist_array)

    cut_list = optimizer.getCutsFunctionList(x_hist_array, y_hist_array, nabla_hist_array)

    xstar = optimizer.getMinimumOfCuts(x_hist_array, y_hist_array, nabla_hist_array, cut_list)
    print("xstar =", xstar)

    optimizer.plotSamples(x_hist_array, y_hist_array, nabla=nabla_hist_array, cut_list=cut_list, objective_function=f1, minimum_of_cuts=xstar)

    print(80 * "*")

    # 2D PLOT ###################################

    # BUG!
    #x_hist_array = np.array([[-0.5,0.5],
    #                         [0.5,0.5]])
    x_hist_array = np.array([[-0.8,0.2],
                             [-0.4,-0.6],
                             [0.3,0.9],
                             [0.7,-0.6]])
    #x_hist_array = np.array([[-0.5,0.5],
    #                         [-0.5,-0.5],
    #                         [0.5,0.5],
    #                         [0.5,-0.5]])
    y_hist_array = f2(x_hist_array)
    nabla_hist_array = f2.gradient(x_hist_array)

    print("x_hist_array =", x_hist_array)
    print("y_hist_array =", y_hist_array)
    print("nabla_hist_array =", nabla_hist_array)

    cut_list = optimizer.getCutsFunctionList(x_hist_array, y_hist_array, nabla_hist_array)

    xstar = optimizer.getMinimumOfCuts(x_hist_array, y_hist_array, nabla_hist_array, cut_list)
    print("xstar =", xstar)

    optimizer.plotSamples(x_hist_array, y_hist_array, nabla=nabla_hist_array, cut_list=cut_list, objective_function=f2, minimum_of_cuts=xstar)

    # 1D PCP ####################################

    #best_x = optimizer.optimize(f1, num_iterations=10)

    # 2D PCP ####################################

    #best_x = optimizer.optimize(f2, num_iterations=10)

if __name__ == '__main__':
    test()

