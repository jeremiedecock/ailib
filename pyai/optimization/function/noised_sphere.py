#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2013,2014,2015,2016,2017 Jeremie DECOCK (http://www.jdhp.org)

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

# STOCHASTIC OBJECTIVE FUNCTIONS ##############################################

# TODO: this class should be generic: take an other class as argument
#       of the constructor and simply add noise on the evaluations of
#       it.

class Function(function.ObjectiveFunction):

    function_name = "Noised sphere"

    def __init__(self, ndim=1, sigma=0.1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)

        self.sigma = sigma


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

        y = np.dot(x,x)
        y = y + np.random.normal(0., self.sigma)

        # Assert y is a (scalar) number.
        assert isinstance(y, numbers.Number), "y = " + str(y)

        return y

    def _eval_multiple_samples(self, x):
        """
        Return the value yi=f(xi) of the function at the point xi in x.
        This function can be redefined to speedup computations.

        The argument x must a numpy array of dimension 2 (x.ndim=2).
        The returned value yi=f(xi) of each point xi in x are scalar
        numbers (not vectors i.e. no multi-objective functions yet).
        Therefore, the returned value y must have y.ndim=1 and
        y.shape[0]=x.shape[0].

        The x array given as argument is considered as following:
           number_of_points := x.shape[0]
           dimension_of_each_point := x.shape[1]
        with:
           x = [[x1],
                [x2],
                [x3],
                ...]
        For instance, the following array x means 3 points defined in R
        (1 dimension) have to be evaluated:
           x = [[ 2.],
                [ 3.],
                [ 4.]]
        For instance, the following array x means 3 points defined in RxR
        (2 dimensions) have to be evaluated:
           x = [[ 2., 2.],
                [ 3., 3.],
                [ 4., 4.]]
        """

        assert x.ndim == 2                   # There are multiple points in x
        number_of_points = x.shape[0]
        dimension_of_each_point = x.shape[1]
        assert dimension_of_each_point == self.ndim, "x.shape[1] = " + str(x) + "; self.ndim =" + str(self.ndim)

        y = np.sum(np.power(x, 2.), 1)
        y = y + np.random.normal(0., self.sigma, y.shape[0])

        # Assert there is one value yi=f(xi) for each point xi in x
        # and assert each yi is a scalar number (not a numpy array).
        assert y.ndim == 1, "y.ndim = " + str(y.ndim)
        assert y.shape[0] == number_of_points, "y.shape = " + str(y.shape) + "x.shape = " + str(x.shape)

        return y


# TEST ########################################################################

def test():
    f = Function(1, sigma=0.1)
    f.plot()

    f = Function(2, sigma=0.1)
    f.plot()

if __name__ == '__main__':
    test()

