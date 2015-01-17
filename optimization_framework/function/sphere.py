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

from . import function

class Function(function.ObjectiveFunction):

    function_name = "Sphere"

    def __init__(self, ndim=1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        y = np.dot(x,x)
        return y


    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y.reshape([-1,1])
        return y


    # GRADIENT ################################################################

#    def _eval_one_gradient(self, point):
#        x = point
#        nabla = 2. * x
#        return nabla


    # STR #####################################################################

    def __str__(self):
        if self.ndim == 1:
            func_str = r"f(x) = x^2"
        else:
            func_str = r"f(x) = \sum_{i=1}^" + str(self.ndim) + r" x_i^2"

        return func_str 


# TEST ########################################################################

def test():
    f = Function(2)
    print(f(np.array([1,1])))
    print(f.gradient(np.array([1,1])))

if __name__ == '__main__':
    test()

