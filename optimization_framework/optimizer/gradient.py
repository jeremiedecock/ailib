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
    import optimizer
else:
    from . import optimizer

class Optimizer(optimizer.Optimizer):

    def optimize(self, objective_function, num_iterations=1000):

        dmin = objective_function.domain_min
        dmax = objective_function.domain_max
        
        x = np.random.uniform(dmin, dmax, objective_function.ndim)

        x_history_list = np.zeros([num_iterations, objective_function.ndim])
        nabla_history_list = np.zeros([num_iterations, objective_function.ndim])

        # Compute the gradient of objective_function at x
        for sample_index in range(num_iterations):
            nabla = objective_function.gradient(x)
            coef = .1  # TODO!!! : http://fr.wikipedia.org/wiki/Algorithme_du_gradient + http://fr.wikipedia.org/wiki/Recherche_lin%C3%A9aire  +  http://fr.wikipedia.org/wiki/Algorithme_%C3%A0_r%C3%A9gions_de_confiance

            x = x - coef * nabla

            # Keep an history of x and nabla to plot things...
            x_history_list[sample_index, :] = x
            nabla_history_list[sample_index, :] = nabla

            #print("DEBUG optimize(): xi =", x)
            #print("DEBUG optimize(): type(xi) =", type(x))

        y_history_list = objective_function(x_history_list)
        self.plotSamples(x_history_list, y_history_list, objective_function=objective_function)
        self.plotCosts(y_history_list)

        #print("DEBUG optimize(): x_history_list =", x_history_list)
        #print("DEBUG optimize(): type(x_history_list) =", type(x_history_list))
        #print("DEBUG optimize(): y_history_list =", y_history_list)
        #print("DEBUG optimize(): type(y_history_list) =", type(y_history_list))
        #print("DEBUG optimize(): x =", x)

        return x


# TEST ########################################################################

def test():
    pass

if __name__ == '__main__':
    test()

