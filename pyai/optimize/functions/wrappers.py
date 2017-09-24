#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Jeremie DECOCK (http://www.jdhp.org)

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
This module contains some classical test functions for unconstrained continuous
single-objective optimization.
"""

__all__ = ['StochasticWrapper']

from scipy import stats

# STOCHASTIC WRAPPERS #########################################################

class StochasticWrapper(_ObjectiveFunction):
    """
    Stochastic wrapper for objective functions.

    TODO
    """
    def __init__(self, objective_function):
        super().__init__()

        self._objective_function = objective_function
        #self._rv = stats.uniform()
        self._rv = stats.norm()

        self.reset_eval_counters()
        self.reset_eval_logs()
        self.do_eval_logs = False

        self.stochastic = True

        self.function_name = " stochastic" + self.function_name


    def __call__(self, x):
        """
        TODO
        """
        #return self._objective_function(x) * self._rv.rvs()
        return self._objective_function(x) + self._rv.rvs()


    def gradient(self, x):
        """
        TODO
        """
        raise Exception("Not implemented")


    def hessian(self, x):
        """
        TODO
        """
        raise Exception("Not implemented")
