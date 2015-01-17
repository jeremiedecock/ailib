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
import matplotlib.pyplot as plt
import warnings

# TODO: improve this ?
if __name__ == '__main__':
    import optimizer
else:
    from . import optimizer

class Optimizer(optimizer.Optimizer):

    def __init__(self):
        pass

    def optimize(self, objective_function, num_iterations=10, num_samples=10, num_selected=5):

        raise Exception("Work in progress...")

        dmin = objective_function.domain_min
        dmax = objective_function.domain_max
        
        x_vec = np.random.uniform(dmin, dmax, objective_function.ndim)

        for iteration_index in range(num_iterations):
            y_vec = objective_function(x_vec.reshape([-1, 1]))

            # pop = zip(x_vec, y_vec)
            pop = np.array([x_vec, y_vec]).T

            # TODO
            pop.sort(key=lambda indiv: indiv.y, reverse=False)
            pop_selected = pop[:num_selected]

            # Make the new probability distribution
            # TODO
            for dim_index in range(objective_function.ndim):
                y1 = objective_function(x)

            # Keep an history of x to plot things...
            # TODO
            x_samples[sample_index, :] = x

            # TODO
            x_vec = np.random.uniform(dmin, dmax, objective_function.ndim)

        y_samples = objective_function(x_samples)
        self.plotSamples(x_samples, y_samples)
        self.plotCosts(y_samples)

        return x


# TEST ########################################################################

def test():
    pass

if __name__ == '__main__':
    test()

