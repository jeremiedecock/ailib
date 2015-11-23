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
import numbers
import math

# TODO: improve this ?
if __name__ == '__main__':
    import function
else:
    from . import function

class Function(function.ObjectiveFunction):

    function_name = "Sinusoid 1"

    def __init__(self):
        self.ndim = 1
        self.domain_min = -10. * np.ones(self.ndim)
        self.domain_max =  10. * np.ones(self.ndim)

        # Set self.function_formula
        self.function_formula = r"f(x) = \sin(4\pi x) \frac{1}{\sqrt{2\pi}} \exp^{-x^2/2}"


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
        assert x.shape[0] == self.ndim == 1  # This function is defined in 1D

        x = x[0]
        y = math.sin(2. * 2. * math.pi * x) * 1./math.sqrt(2.*math.pi) * math.exp(-(x**2)/2.)

        # Assert y is a (scalar) number.
        assert isinstance(y, numbers.Number), "y = " + str(y)

        return y


# TEST ########################################################################

def test():
    f1 = Function()

    f1.plot()

    # One point (1D)
    for xi in [0., 1., 2.]:
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

if __name__ == '__main__':
    test()

