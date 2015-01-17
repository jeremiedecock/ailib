# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['GradientDescent']

import numpy as np

from . import optimizer

class GradientDescent(optimizer.Optimizer):

    def optimize(self, objective_function, num_iterations=1000):

        dmin = objective_function.domain_min
        dmax = objective_function.domain_max
        
        x = np.random.uniform(dmin, dmax, objective_function.ndim)

        x_history_list = np.zeros([num_iterations, objective_function.ndim])
        nabla_history_list = np.zeros([num_iterations, objective_function.ndim])

        # Compute the gradient of objective_function at x
        for sample_index in range(num_iterations):
            nabla = objective_function.gradient(x)
            coef = 1.  # TODO!!! : http://fr.wikipedia.org/wiki/Algorithme_du_gradient + http://fr.wikipedia.org/wiki/Recherche_lin%C3%A9aire  +  http://fr.wikipedia.org/wiki/Algorithme_%C3%A0_r%C3%A9gions_de_confiance

            x = x - coef * nabla

            # Keep an history of x and nabla to plot things...
            x_history_list[sample_index, :] = x
            nabla_history_list[sample_index, :] = nabla

        y_history_list = objective_function(x_history_list)
        self.plotSamples(x_history_list, y_history_list, objective_function)
        self.plotCosts(y_history_list)

        return x

