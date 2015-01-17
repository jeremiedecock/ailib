# -*- coding: utf-8 -*-

# Copyright (c) 2015 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['Degree2PolynomialFunction']

import numpy as np
import numbers
from . import function

class Degree2PolynomialFunction(function.ObjectiveFunction):

    def __init__(self, coef_deg2, coef_deg1, coef_deg0, ndim=1):
        self.ndim = ndim
        self.domain_min = -1. * np.ones(ndim)
        self.domain_max =  1. * np.ones(ndim)

        # Check coef are vectors (not matrices)
        assert coef_deg2.ndim == 1, "coef_deg2 = " + str(coef_deg2)
        assert coef_deg1.ndim == 1, "coef_deg1 = " + str(coef_deg1)

        # Check coef vector dimension == the polynomial dimension
        assert coef_deg2.shape[0] == ndim, "coef_deg2 = " + str(coef_deg2)
        assert coef_deg1.shape[0] == ndim, "coef_deg1 = " + str(coef_deg1)

        # Check coef_deg0 is an number (scalar not a vector or a matrix)
        assert isinstance(coef_deg0, numbers.Number), "coef_deg0 = " + str(coef_deg0)

        self.coef_deg2 = coef_deg2
        self.coef_deg1 = coef_deg1
        self.coef_deg0 = coef_deg0

        print(self)

    def _eval_one_sample(self, x):
        y = np.dot(self.coef_deg2 * x, x) + np.dot(self.coef_deg1, x) + self.coef_deg0
        return y


    def _eval_multiple_samples(self, x):
        y = np.sum(np.power(x, 2.), 1)
        y = y.reshape([-1,1])
        return y


#    def _eval_one_gradient(self, point):
#        x = point
#        y = self.coef_deg2 * 2. * x + self.coef_deg1
#        return y


    def __str__(self):

        # TODO: considere negative coef (avoid "+ -n")

        terms_str_list = []

        # Deg. 2
        for d_index in range(self.ndim):
            if self.coef_deg2[d_index] != 0:
                if self.coef_deg2[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index) + "^2")
                else:
                    terms_str_list.append(str(self.coef_deg2[d_index]) + " x_" + str(d_index) + "^2")

        # Deg. 1
        for d_index in range(self.ndim):
            if self.coef_deg1[d_index] != 0:
                if self.coef_deg1[d_index] == 1:
                    terms_str_list.append("x_" + str(d_index))
                else:
                    terms_str_list.append(str(self.coef_deg1[d_index]) + " x_" + str(d_index))

        # Deg. 0
        if self.coef_deg0 != 0:
            terms_str_list.append(str(self.coef_deg0))

        function_str = "f(x) = " + " + ".join(terms_str_list)

        return function_str

