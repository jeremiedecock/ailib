#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2014,2015,2016,2017 Jeremie DECOCK (http://www.jdhp.org)

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

# TODO: improve this ?
if __name__ == '__main__':
    import function
else:
    from . import function

class Function(function.ObjectiveFunction):

    function_name = "Degree 2 polynomial"

    def __init__(self, coef_deg2, coef_deg1, coef_deg0, ndim=1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)

        # Check coef are vectors (not matrices)
        assert coef_deg2.ndim == 1, "coef_deg2 = " + str(coef_deg2)
        assert coef_deg1.ndim == 1, "coef_deg1 = " + str(coef_deg1)

        # Check coef vector dimension == the polynomial dimension
        assert coef_deg2.shape[0] == ndim, "coef_deg2 = " + str(coef_deg2)
        assert coef_deg1.shape[0] == ndim, "coef_deg1 = " + str(coef_deg1)

        # Check coef_deg0 is an number (scalar not a vector or a matrix)
        assert isinstance(coef_deg0, numbers.Number), "coef_deg0 = " + str(coef_deg0)

        self.coef_deg2 = coef_deg2
        self.coef_deg1 = coef_deg1
        self.coef_deg0 = coef_deg0

        # Set self.function_formula #######################
        # TODO: considere negative coef (avoid "+ -n")

        terms_str_list = []

        for d_index in range(self.ndim):
            # Deg. 2
            if self.coef_deg2[d_index] != 0:
                if self.coef_deg2[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index) + "^2")
                else:
                    terms_str_list.append(str(self.coef_deg2[d_index]) + " x_" + str(d_index) + "^2")

        for d_index in range(self.ndim):
            # Deg. 1
            if self.coef_deg1[d_index] != 0:
                if self.coef_deg1[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index))
                else:
                    terms_str_list.append(str(self.coef_deg1[d_index]) + " x_" + str(d_index))

        if self.coef_deg0 != 0:
            # Deg. 0
            terms_str_list.append(str(self.coef_deg0))

        self.function_formula = "f(x) = " + " + ".join(terms_str_list)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        """
        Return the value y=f(x) of the function at the point x.

        The argument x must be a numpy array of dimension 1 (x.ndim=1 i.e. a
        vector not a matrix).
        The returned value y=f(x) is a scalar number (not a numpy array i.e. no
        multi-objective functions yet).
        """

        assert x.ndim == 1                   # There is only one point in x
        assert x.shape[0] == self.ndim       # This function is defined in self.ndim dim

        y = np.dot(self.coef_deg2 * x, x) + np.dot(self.coef_deg1, x) + self.coef_deg0

        # Assert y is a (scalar) number.
        assert isinstance(y, numbers.Number), "y = " + str(y)

        return y


    # GRADIENT ################################################################

    def _eval_one_gradient(self, x):
        """
        Return the gradient of the function at the point x.

        The argument x must be a numpy array of dimension 1 (x.ndim=1 i.e. a
        vector not a matrix).
        The returned value nabla is a numpy array of dimension 1 (i.e. a vector
        not a matrix).
        """

        assert x.ndim == 1                   # There is only one point in x
        assert x.shape[0] == self.ndim       # This function is defined in self.ndim dim

        nabla = 2. * self.coef_deg2 * x + self.coef_deg1

        # Assert nabla is a numpy array of dimension 1 (i.e. a vector) with
        # the same number of elements (dimension) than point x.
        assert nabla.ndim == 1, "nabla.ndim = " + str(nabla)    # there is only one point x
        assert nabla.shape == x.shape, "nabla.shape = " + str(nabla.shape) + "x.shape = " + str(x.shape)

        return nabla



# TEST ########################################################################

def test():
    f1 = Function(np.array([6.,2.]), np.array([1.,2.]), 1., 2)

    f1.plot()

    # One point
    for xi in [-1., 0., 1., 2.]:
        print()
        print("*** One point 2D ***")
        x = np.array([xi, xi])
        print("x =", x)
        print("x.ndim =", x.ndim)
        print("x.shape =", x.shape)
        y = f1(x)
        nabla = f1.gradient(x)
        nabla_num = f1._eval_one_num_gradient(x)
        print("f(x) =", y)
        print("nabla =", nabla)
        print("nabla_num =", nabla_num)

    # Multiple points (2D)
    print()
    print("*** 3 points 2D ***")
    x = np.array([[2., 2.], [3., 3.], [4., 4.]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f1(x)
    nabla = f1.gradient(x)
    print("f(x) =", y)
    print("nabla =", nabla)

if __name__ == '__main__':
    test()

