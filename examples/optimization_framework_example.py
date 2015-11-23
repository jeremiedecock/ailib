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

"""
A PyAI (optimization framework) demo.
"""
import math
import numpy as np

# CONFIG ######################################################################

objective_func = "sphere"
optimizer_choice = "saes"

# MAIN ########################################################################

def main():
    """
    A PyAI (optimization framework) demo.
    """

    # SETUP OBJECTIVE FUNCTION ############################

    if objective_func == "sphere":
        # Sphere ##########################
        from pyai.optimization.function.sphere import Function
        #f = Function(1)
        f = Function(2)
        #f = Function(10)

    elif objective_func == "noised_sphere":
        # Noised sphere ###################
        from pyai.optimization.function.noised_sphere import Function
        #f = Function(1)
        f = Function(2)

    elif objective_func == "sin1":
        # Sinusoid functions ##############
        from pyai.optimization.function.sin1 import Function
        f = Function()

    elif objective_func == "sin2":
        # Sinusoid functions ##############
        from pyai.optimization.function.sin2 import Function
        f = Function()

    elif objective_func == "sin3":
        # Sinusoid functions ##############
        from pyai.optimization.function.sin3 import Function
        f = Function()

    elif objective_func == "yahoo":
        # Yahoo function ##################
        from pyai.optimization.function.yahoo import Function
        f = Function()

    elif objective_func == "deg_2_poly":
        # Degree 2 polynomial function ####
        from pyai.optimization.function.degree_2_polynomial import Function
        f = Function(np.array([6.,2.]), np.array([1.,2.]), 1., 2)

    else:
        raise Exception("Wrong objective_func value.")

    # Plot ########
    #f.plot()


    # OPTIMIZER ###########################################

    if optimizer_choice == "naive":
        # Naive Minimizer #################
        from pyai.optimization.optimizer.naive import Optimizer
        optimizer = Optimizer()
        best_x = optimizer.optimize(f, num_samples=300)

    elif optimizer_choice == "gradient":
        # Gradient descent ################
        from pyai.optimization.optimizer.gradient import Optimizer
        optimizer = Optimizer()
        f.delta = 0.01
        best_x = optimizer.optimize(f, num_iterations=30)

    elif optimizer_choice == "saes":
        # SAES ############################
        from pyai.optimization.optimizer.saes_hgb import Optimizer
        optimizer = Optimizer(x_init=np.ones(f.ndim), num_evals_func=lambda gen_index: math.floor(10. * pow(gen_index, 0.5)))
        optimizer = Optimizer(x_init=np.ones(f.ndim))
        best_x = optimizer.optimize(f, num_gen=50)

    elif optimizer_choice == "cutting_plane":
        # Cutting plane ###################
        from pyai.optimization.optimizer.cutting_plane import Optimizer
        optimizer = Optimizer()

        #best_x = optimizer.optimize(f, num_iterations=7)   # sphere with 1 dimension
        #best_x = optimizer.optimize(f, num_iterations=15) # sphere with 2 dimensions
        #best_x = optimizer.optimize(f, num_iterations=100) # sphere with 10 dimensions

        #best_x = optimizer.optimize(f, parallel="linear", num_iterations=7)   # sphere with 1 dimension
        #best_x = optimizer.optimize(f, parallel="linear", num_iterations=100)   # sphere with 10 dimension

        #best_x = optimizer.optimize(f, parallel="gaussian", num_iterations=7)   # sphere with 1 dimension
        #best_x = optimizer.optimize(f, parallel="gaussian", num_iterations=100)   # sphere with 10 dimension

        best_x = optimizer.optimize(f, num_iterations=15) # sphere with 2 dimensions

    elif optimizer_choice == "eda":
        # EDA #############################
        #from pyai.optimization.optimizer.eda import Optimizer
        pass

    else:
        raise Exception("Wrong optimizer_choice value.")

    print("Best sample: f(", best_x, ") = ", f(best_x))

if __name__ == '__main__':
    main()

