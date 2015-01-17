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
import math

#from optimizer.eda import Optimizer
from optimizer.gradient import Optimizer
#from optimizer.naive import Optimizer
#from optimizer.saes_hgb import Optimizer

# MAIN ########################################################################

def main():

    # SETUP OBJECTIVE FUNCTION ############################

    # Sphere ##########################
    from function.sphere import Function
    #f = Function(1)
    f = Function(2)

    # Noised sphere ###################
    #from function.noised_sphere import Function
    #f = Function(1)
    #f = Function(2)

    # Sinusoid functions ##############
    #from function.sin1 import Function
    #from function.sin2 import Function
    #from function.sin3 import Function
    #f = Function()

    # Yahoo function ##################
    #from function.yahoo import Function
    #f = Function()

    # Degree 2 polynomial function ####
    #from function.degree_2_polynomial import Function
    #f = Function(np.array([6.,2.]), np.array([1.,2.]), 1., 2)

    # Plot
    f.plot()


    # SETUP OPTIMIZER #####################################

    # Naive Minimizer #################
    #optimizer = Optimizer()

    # Gradient descent ################
    optimizer = Optimizer()
    f.delta = 0.01

    # SAES ############################
    #optimizer = Optimizer(x_init=np.ones(f.ndim), num_evals_func=lambda gen_index: math.floor(10. * pow(gen_index, 0.5)))
    #optimizer = Optimizer(x_init=np.ones(f.ndim))


    # OPTIMIZE ############################################

    #best_x = optimizer.optimize(f, num_gen=500)
    best_x = optimizer.optimize(f, num_iterations=3000)

    print("Best sample: f(", best_x, ") = ", f(best_x))

if __name__ == '__main__':
    main()

