#!/usr/bin/env python3
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

# TODO: improve this ?
if __name__ == '__main__':
    import function
else:
    from . import function

class Function(function.ObjectiveFunction):

    function_name = "Sphere"

    def __init__(self, ndim=1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)

        # Set self.function_formula
        if self.ndim == 1:
            self.function_formula = r"f(x) = x^2"
        else:
            self.function_formula = r"f(x) = \sum_{i=1}^" + str(self.ndim) + r" x_i^2"


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        y = np.dot(x,x)
        return y


    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y.reshape([-1,1])
        return y


    # GRADIENT ################################################################

    def _eval_one_gradient(self, point):
        x = point
        nabla = 2. * x
        return nabla


# TEST ########################################################################

def test():
    f1 = Function(1)
    f2 = Function(2)

    f1.plot()
    f2.plot()

    # One point (1D)
    for xi in [-1., 0., 1., 2.]:
        print()
        print("*** One point 1D ***")
        x = np.array([xi])
        print("x =", x)
        print("x.ndim =", x.ndim)
        print("x.shape =", x.shape)
        y = f1(x)
        nabla = f1.gradient(x)
        nabla_num = f1._eval_one_num_gradient(x)
        print("f(x) =", y)
        print("nabla =", nabla)
        print("nabla_num =", nabla_num)

    # One point (2D)
    print()
    print("*** One point 2D ***")
    x = np.array([2., 2.])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f2(x)
    nabla = f2.gradient(x)
    nabla_num = f2._eval_one_num_gradient(x)
    print("f(x) =", y)
    print("nabla =", nabla)
    print("nabla_num =", nabla_num)

    # Multiple points (1D)
    print()
    print("*** 3 points 1D ***")
    x = np.array([[2.], [3.], [4.]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f1(x)
    nabla = f1.gradient(x)
    print("f(x) =", y)
    print("nabla =", nabla)

    # Multiple points (2D)
    print()
    print("*** 3 points 2D ***")
    x = np.array([[2., 2.], [3., 3.], [4., 4.]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f2(x)
    nabla = f2.gradient(x)
    print("f(x) =", y)
    print("nabla =", nabla)

if __name__ == '__main__':
    test()

