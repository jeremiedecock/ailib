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
import math

# TODO: improve this ?
if __name__ == '__main__':
    import function
else:
    from . import function

class Function(function.ObjectiveFunction):

    function_name = "Sinusoid 2"

    def __init__(self):
        self.ndim = 1
        self.domain_min = 0. * np.ones(self.ndim)
        self.domain_max = 10. * np.ones(self.ndim)

        # Set self.function_formula
        self.function_formula = r"f(x) = \sin(4\pi |x|) \exp^{-5|x|}"


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
        x = np.absolute(x)

        y = math.sin(2. * 2. * math.pi * x) * math.exp(-5. * x)

        # Assert y is a (scalar) number.
        assert isinstance(y, numbers.Number), "y = " + str(y)

        return y


# TEST ########################################################################

def test():
    f = Function()
    f.plot()

if __name__ == '__main__':
    test()

