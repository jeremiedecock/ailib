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

    function_name = "Sinusoid 3"

    def __init__(self):
        self.ndim = 2
        self.domain_min = -10. * np.ones(self.ndim)
        self.domain_max =  10. * np.ones(self.ndim)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        r = np.sqrt(np.power(x[0], 2) + np.power(x[1], 2))
        y = np.sin(r)
        return y

    def _eval_multiple_samples(self, x):
        r = np.sqrt(np.power(x[:,0], 2) + np.power(x[:,1], 2))
        y = np.sin(r)
        y = y.reshape([-1,1])
        return y

