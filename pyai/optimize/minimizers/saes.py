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

__all__ = ['SAES']

import math
import numpy as np
import random

from .optimizer import Optimizer


class Individual:
    """The individual class.

    Parameters
    ----------
    x : ndarray
        The individual's value (1D numpy array).
    sigma : float
        The individual's sigma (TODO).
    y : float
        The individual's cost.
    """

    def __init__(self, x, sigma, y):
        self.x = x
        self.sigma = sigma
        self.y = y

    def __str__(self):
        return "{0} {1} {2}".format(self.x, self.sigma, self.y)


class SAES(Optimizer):
    """SAES optimizer.

    See:
    * http://www.scholarpedia.org/article/Evolution_strategies
    * https://homepages.fhv.at/hgb/downloads/mu_mu_I_lambda-ES.oct

    Parameters
    ----------
    mu : int
        The number of parents.
    lambda_ : int
        The number of offspring.
    sigma_init : float
        The initial global mutation strength sigma. 
    sigma_min : float
        The stop criterion: the optimization is stopped when `sigma` is smaller
        than `sigma_min`.
    sigma_init : int
        The number of times the (noisy) objective functions should be called
        at each evaluation (taking the average value of these calls).
    """

    def __init__(self,
                 mu=3,                  # TODO: move that to `minimize` ?
                 lambda_=12,            # TODO: move that to `minimize` ?
                 sigma_init=1.,         # TODO: move that to `minimize` ?
                 sigma_min=1e-5,        # TODO: move that to `minimize` ?
                 num_evals_func=None):  # TODO: move that to `minimize` ?
        super().__init__()

        self.mu = mu
        self.lambda_ = lambda_
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.num_evals_func = num_evals_func

    def select_individuals(self, pop):
        """This sorts the population according to the individuals' fitnesses.

        Parameters
        ----------
        pop : list of Individual
            The list of Individual objects to sort and select.

        Returns
        -------
        list of Individual
            The list of selected Individuals.
        """
        pop.sort(key=lambda indiv: indiv.y, reverse=False)
        return pop[:self.mu]

    def recombine_individuals(self, parents):
        """This performs intermediate (multi-)recombination.

        Parameters
        ----------
        parents : list of Individual
            The list of Individual objects to recombine.

        Returns
        -------
        list of Individual
            The list of recombined Individuals.
        """
        parents_y = np.array([indiv.x for indiv in parents])
        parents_sigma = np.array([indiv.sigma for indiv in parents])
        recombinant = Individual(parents_y.mean(axis=0), parents_sigma.mean(), 0) # TODO
        return recombinant

    def minimize(self,
                 objective_function,
                 num_gen=50,
                 x_init=None):
        """TODO

        Parameters
        ----------
        x_init : ndarray
            The initial parent vector (a 1D numpy array).

        Returns
        -------
        ndarray
            The optimal point found (a 1D numpy array).
        """

        self.log.data["x"] = []
        self.log.data["sigma"] = []
        self.log.data["y"] = []

        if x_init is None:
            x_init = np.random.random(objective_function.ndim)   # draw samples in [0.0, 1.0)

            min_bounds = objective_function.bounds[0]
            max_bounds = objective_function.bounds[1]

            x_init *= (max_bounds - min_bounds)
            x_init += min_bounds

        assert x_init.ndim == 1
        assert x_init.shape[0] == objective_function.ndim

        # Initialization
        n = x_init.shape[0]              # determine search space dimensionality n   
        tau = 1. / math.sqrt(2.*n)       # self-adaptation learning rate

        # Initializing individual population
        y = objective_function(x_init)
        parent_pop = [Individual(x_init, self.sigma_init, y) for i in range(self.mu)]

        gen_index = 0

        # Evolution loop of the (mu/mu_I, lambda)-sigma-SA-ES
        while parent_pop[0].sigma > self.sigma_min and gen_index < num_gen:
            offspring_pop = []
            recombinant = self.recombine_individuals(parent_pop) # TODO: BUG ? this statement may be in the next line
            for offspring_index in range(1, self.lambda_):
                offspring_sigma = recombinant.sigma * math.exp(tau * random.normalvariate(0,1))
                offspring_x = recombinant.x + offspring_sigma * np.random.normal(size=n)

                if self.num_evals_func is None:
                    # If the objective function is deterministic
                    offspring_y = objective_function(offspring_x)
                else:
                    # If the objective function is stochastic
                    # TODO: move this in function or in optimizer (?) class so that it is available for all optimiser implementations...
                    num_evals = self.num_evals_func(gen_index)
                    offspring_y_list = np.zeros(num_evals)
                    for eval_index in range(num_evals):
                        offspring_y_list[eval_index] = float(objective_function(offspring_x))
                    offspring_y = np.mean(offspring_y_list)
                    # TODO: generate the confidence bounds of offspring_y and plot it

                offspring = Individual(offspring_x, offspring_sigma, offspring_y)
                offspring_pop.append(offspring)
            parent_pop = self.select_individuals(offspring_pop)
            #parent_pop = self.select_individuals(parent_pop + offspring_pop)

            gen_index += 1

            self.log.data["x"].append(parent_pop[0].x)  # TODO use a "log" object instead
            self.log.data["y"].append(parent_pop[0].y)  # TODO
            print(parent_pop[0])

        #self.plotSamples(np.array(self.log.data["x"]), np.array(self.log.data["y"]), objective_function=objective_function)
        #self.plotCosts(np.array(self.log.data["y"]))

        return parent_pop[0].x

# Remark: Final approximation of the optimizer is in "parent_pop[0].x"
#         corresponding fitness is in "parent_pop[0].y" and the final 
#         mutation strength is in "parent_pop[0].sigma"
