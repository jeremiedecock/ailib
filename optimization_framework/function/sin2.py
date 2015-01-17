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

    function_name = "Sinusoid 2"

    def __init__(self):
        self.ndim = 1
        self.domain_min = 0. * np.ones(self.ndim)
        self.domain_max = 10. * np.ones(self.ndim)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        x = np.absolute(x)
        y = np.sin(2 * 2 * np.pi * x) * np.exp(-5 * x)
        return y


# TEST ########################################################################

def test():
    f = Function()
    f.plot()

if __name__ == '__main__':
    test()

