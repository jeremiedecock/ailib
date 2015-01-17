# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['EDA']

import numpy as np
import matplotlib.pyplot as plt
import warnings

from . import optimizer

class GradientDescent(optimizer.Optimizer):

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

