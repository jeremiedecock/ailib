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
import math

# TODO: improve this ?
if __name__ == '__main__':
    import signal_function
else:
    from . import signal_function

__all__ = ['PeriodicContinuousSignal']  # TODO

class PeriodicContinuousSignal(signal_function.SignalFunction):

    signal_name = "Periodic Continuous Signal"   # TODO (class member ?)

    def __init__(self, a, b, T=2*math.pi): # TODO: here, supports are cos and sin => periode is always 2*pi...
        # TODO: check a (a should be a numpy array of dim 1 or 2)
        # TODO: check b (b should be a numpy array of dim 1 or 2)
        # TODO: check T (T should be a scalar)

        self.ndim = 1
        self.a = a           # TODO
        self.b = b           # TODO
        self.periode = T     # TODO

        self.domain_min = -10. * np.ones(self.ndim)     # TODO: remove this (in self.plot too)
        self.domain_max =  10. * np.ones(self.ndim)     # TODO: remove this (in self.plot too)

        # Set self.function_formula
        self.function_formula = r"f(x) ="
        # TODO: for ...:  # TODO: the formula depends on a and b
        # TODO:     self.function_formula += r"a_0 + a_1 \sin(t) + b_1 \cos{t} + ..." # TODO: the formula depends on a and b


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
        assert x.shape[0] == self.ndim == 1  # This function is defined in 1D TODO...

        x = x[0]  # TODO

        y = 1. + math.sin(x) + 2. * math.cos(x) + math.cos(2. * x)     # TODO

        # Assert y is a (scalar) number.
        assert isinstance(y, numbers.Number), "y = " + str(y)

        return y


# TEST ########################################################################

def test():
    f1 = PeriodicContinuousSignal(np.array[1], np.array[1])

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
        print("f(x) =", y)

    # Multiple points (1D)
    print()
    print("*** 3 points 1D ***")
    x = np.array([[2.], [3.], [4.]])
    print("x =", x)
    print("x.ndim =", x.ndim)
    print("x.shape =", x.shape)
    y = f1(x)
    print("f(x) =", y)

if __name__ == '__main__':
    test()

