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
from matplotlib.finance import quotes_historical_yahoo
import datetime

# TODO: improve this ?
if __name__ == '__main__':
    import function
else:
    from . import function

class Function(function.ObjectiveFunction):

    function_name = "Yahoo"

    def __init__(self):
        self.ndim = 1

        date1 = datetime.date(1995, 1, 1) 
        date2 = datetime.date(2004, 4, 12)

        quotes = quotes_historical_yahoo('INTC', date1, date2)
        #self._quote = [q[1] for q in quotes]
        self._quote = [-q[1] for q in quotes]

        self.domain_min = 0. * np.ones(self.ndim)
        self.domain_max = len(self._quote) * np.ones(self.ndim)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        #y = float("-inf")
        y = float("inf")
        if self.domain_min <= x < self.domain_max:
            y = self._quote[int(x)]
        return y


# TEST ########################################################################

def test():
    f = Function()
    f.plot()

if __name__ == '__main__':
    test()

