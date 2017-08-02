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

# TODO: improve this ?
if __name__ == '__main__':
    import optimizer
else:
    from . import optimizer

class Optimizer(optimizer.Optimizer):

    optimizer_name = "gradient descent"

    def optimize(self, objective_function, num_iterations=1000):

        dmin = objective_function.domain_min
        dmax = objective_function.domain_max
        
        # Get the first point
        x = np.random.uniform(dmin, dmax, objective_function.ndim)

        # Init history lists
        x_history_array = np.zeros([num_iterations, objective_function.ndim])
        nabla_history_array = np.zeros([num_iterations, objective_function.ndim])

        # Main loop: for each iteration do...
        for sample_index in range(num_iterations):

            # Compute the gradient of objective_function at x
            nabla = objective_function.gradient(x)
            coef = .1  # TODO!!! : http://fr.wikipedia.org/wiki/Algorithme_du_gradient + http://fr.wikipedia.org/wiki/Recherche_lin%C3%A9aire  +  http://fr.wikipedia.org/wiki/Algorithme_%C3%A0_r%C3%A9gions_de_confiance

            x = x - coef * nabla

            # Keep an history of x and nabla to plot things...
            x_history_array[sample_index, :] = x
            nabla_history_array[sample_index, :] = nabla

        y_history_array = objective_function(x_history_array)
        self.plotSamples(x_history_array, y_history_array, objective_function=objective_function)
        self.plotCosts(y_history_array)

        return x


# TEST ########################################################################

def test():
    pass

if __name__ == '__main__':
    test()

