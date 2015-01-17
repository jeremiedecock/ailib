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
        y = np.dot(x,x)
        y = y + np.random.normal(0., self.sigma)
        return y

    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y + np.random.normal(0., self.sigma, y.shape[0])
        y = y.reshape([-1,1])
        return y


# TEST ########################################################################

def test():
    f = Function(1, sigma=0.1)
    f.plot(xstep=0.05)

    f = Function(2, sigma=0.1)
    f.plot(xstep=0.05)

if __name__ == '__main__':
    test()

