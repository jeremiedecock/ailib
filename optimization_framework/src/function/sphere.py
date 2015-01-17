# -*- coding: utf-8 -*-

# Copyright (c) 2013,2015 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['SphereFunction']

import numpy as np
from . import function

class SphereFunction(function.ObjectiveFunction):

    def __init__(self, ndim=1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)


    # EVAL ####################################################################

    def _eval_one_sample(self, x):
        y = np.dot(x,x)
        return y


    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y.reshape([-1,1])
        return y


    # GRADIENT ################################################################

#    def _eval_one_gradient(self, point):
#        x = point
#        nabla = 2. * x
#        return nabla


    # STR #####################################################################

    def __str__(self):
        if self.ndim == 1:
            func_str = r"f(x) = x^2"
        else:
            func_str = r"f(x) = \sum_{i=1}^" + str(self.ndim) + r" x_i^2"

        return func_str 


# TEST ########################################################################

def test():
    f = function.SphereFunction(2)
    print(f(np.array([1,1])))
    print(f.gradient(np.array([1,1])))

if __name__ == '__main__':
    test()

