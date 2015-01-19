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


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        y = np.dot(self.coef_deg2 * x, x) + np.dot(self.coef_deg1, x) + self.coef_deg0
        return y


    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y.reshape([-1,1])
        return y


    # GRADIENT ################################################################

#    def _eval_one_gradient(self, point):
#        x = point
#        y = self.coef_deg2 * 2. * x + self.coef_deg1
#        return y


    # STR #####################################################################

    def __str__(self):

        # TODO: considere negative coef (avoid "+ -n")

        terms_str_list = []

        # Deg. 2
        for d_index in range(self.ndim):
            if self.coef_deg2[d_index] != 0:
                if self.coef_deg2[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index) + "^2")
                else:
                    terms_str_list.append(str(self.coef_deg2[d_index]) + " x_" + str(d_index) + "^2")

        # Deg. 1
        for d_index in range(self.ndim):
            if self.coef_deg1[d_index] != 0:
                if self.coef_deg1[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index))
                else:
                    terms_str_list.append(str(self.coef_deg1[d_index]) + " x_" + str(d_index))

        # Deg. 0
        if self.coef_deg0 != 0:
            terms_str_list.append(str(self.coef_deg0))

        function_str = "f(x) = " + " + ".join(terms_str_list)

        return function_str


# TEST ########################################################################

def test():
    f1 = Function(np.array([6.,2.]), np.array([1.,2.]), 1., 2)

    #f1.plot()

    # One point
    print("*** One point 2D ***")
    x = np.array([2., 2.])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f1(x)
    nabla = f1.gradient(x)
    print("f(x) =", y)
    print("nabla =", nabla)

    # Multiple points (2D)
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

