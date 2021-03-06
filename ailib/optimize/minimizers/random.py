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

__all__ = ['Random']

import numpy as np

from .optimizer import Optimizer

class Random(Optimizer):
    
    def minimize(self, objective_function, num_samples=1000, ndim=None, dmin=None, dmax=None):

        if dmin is None:
            dmin = objective_function.domain_min

        if dmax is None:
            dmax = objective_function.domain_max

        if ndim is None:
            ndim = objective_function.ndim
        
        x_samples = np.random.uniform(dmin, dmax, [ndim, num_samples])
        y_samples = objective_function(x_samples)
        x_min = x_samples[:, y_samples.argmin()]

        return x_min
