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

from .optimizer import Optimizer


class SAES(Optimizer):
    """SAES optimizer.

    See:
    * http://www.scholarpedia.org/article/Evolution_strategies
    * Notebook "ai_optimization_saes_en.ipynb" on jdhp.org

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

    def minimize(self,
                 objective_function,
                 init_pop_mu,
                 init_pop_sigma,
                 num_gen=50,
                 mu=3,
                 lmb=6,
                 rho=1,
                 selection_operator='+'):
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

        d = objective_function.ndim
        tau = 1./math.sqrt(2.*d)         # self-adaptation learning rate

        # Init the population ##########################

        # "pop" array layout:
        # - the first mu lines contain parents
        # - the next lambda lines contain children
        # - the first column contains the individual's strategy (sigma)
        # - the last column contains the individual's assess (f(x))
        # - the other columns contain the individual value (x)

        pop = np.full([mu+lmb, d+2], np.nan)
        pop[:mu, 0] = 1.                                       # init the parents strategy to 1.0
        pop[:mu, 1:-1] = np.random.normal(init_pop_mu,
                                          init_pop_sigma,
                                          size=[mu,d])         # init the parents value
        pop[:mu, -1] = objective_function(pop[:mu, 1:-1].T)                  # evaluate parents
        #print("Initial population:\n", pop)

        ## Sort parents
        #pop = pop[pop[:,-1].argsort()]
        #print(pop)

        for gen in range(num_gen):
            # Make children ################################
            if rho == 1:
                # Each child is made from one randomly selected parent
                pop[mu:,:] = pop[np.random.randint(mu, size=lmb)]
            elif rho == mu:
                # Recombine all parents for each child
                raise NotImplemented() # TODO
            elif 1 < rho < mu:
                # Recombine rho randomly selected parents for each child
                raise NotImplemented() # TODO
            else:
                raise ValueError()

            pop[mu:,-1] = np.nan
            #print("Children:\n", pop)

            # Mutate children's sigma ######################
            pop[mu:,0] = pop[mu:,0] * np.exp(tau * np.random.normal(size=lmb))
            #print("Mutated children (sigma):\n", pop)

            # Mutate children's value ######################
            pop[mu:,1:-1] = pop[mu:,1:-1] + pop[mu:,1:-1] * np.random.normal(size=[lmb,d])
            #print("Mutated children (value):\n", pop)

            # Evaluate children ############################
            pop[mu:, -1] = objective_function(pop[mu:, 1:-1].T)
            #print("Evaluated children:\n", pop)

            # Select the best individuals ##################
            if selection_operator == '+':
                # *plus-selection* operator
                pop = pop[pop[:,-1].argsort()]
            elif selection_operator == ',':
                # *comma-selection* operator
                pop[:lmb,:] = pop[pop[mu:,-1].argsort()]   # TODO: check this...
            else:
                raise ValueError()

            pop[mu:, :] = np.nan

            #print("Selected individuals for the next generation:\n", pop)

        return pop[0, 1:-1]
