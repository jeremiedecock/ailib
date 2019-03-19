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
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors
except Exception as e:
    print(e)

from .optimizer import Optimizer


class SAES(Optimizer):
    """SAES optimizer.

    ($\mu$/1+$\lambda$)-$\sigma$-Self-Adaptation-ES


    Init pop

    $\forall$ gen

    $\quad$ $\forall$ child

    $\quad\quad$ 1. select $\rho$ parents

    $\quad\quad$ 2. recombination of selected parents (if $\rho > 1$)

    $\quad\quad$ 3. mutation of $\sigma$ (individual strategy) : $\sigma \leftarrow \sigma ~ e^{\tau \mathcal{N}(0,1)}$

    $\quad\quad$ 4. mutation of $\boldsymbol{x}$ (objective param) : $\boldsymbol{x} \leftarrow \boldsymbol{x} + \sigma ~ \mathcal{N}(0,1)$

    $\quad\quad$ 5. eval $f(\boldsymbol{x})$
    
    $\quad$ Select next gen individuals


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
                 init_pop_mean,
                 init_pop_std,
                 num_gen=50,
                 mu=3,
                 lmb=6,
                 rho=1,
                 tau=None,
                 selection_operator='+',
                 isotropic_mutation=True,
                 plot=False):
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

        assert selection_operator in (',', '+')

        # Number of dimension of the solution space
        d = objective_function.ndim

        # Self-adaptation learning rate
        if tau is None:
            tau = 1./math.sqrt(2.*d)

        # Set indices alias ############################

        all_indices = slice(None, None)
        parent_indices = slice(0, mu)
        children_indices = slice(mu, None)

        sigma_col = 0
        x_cols = slice(1, -1)
        y_col = -1

        sigma_col_label = "sigma"
        y_col_label = "y"

        # Init the population ##########################

        # "pop" array layout:
        # - the first mu lines contain parents
        # - the next lambda lines contain children
        # - the first column contains the individual's strategy (sigma)
        # - the last column contains the individual's assess (f(x))
        # - the other columns contain the individual value (x)

        pop = pd.DataFrame(np.full([mu+lmb, d+2], np.nan),
                           columns=[sigma_col_label] + ["x" + str(d) for d in range(d)] + [y_col_label])

        pop.iloc[parent_indices, sigma_col] = 1.                                    # init the parents strategy to 1.0
        #pop.iloc[parent_indices, x_cols] = np.random.normal(init_pop_mean,
        #                                                    init_pop_std,
        #                                                    size=[mu, d])          # init the parents value
        pop.iloc[parent_indices, x_cols] = np.random.uniform(low=-10., high=10., size=[mu, d])    # init the parents value
        pop.iloc[parent_indices, y_col] = objective_function(pop.iloc[parent_indices, x_cols].values.T)  # evaluate parents
        #print("Initial population:\n", pop, "\n")

        # Plot #############################################

        if plot:
            cmap = cm.gnuplot2 # magma

            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 6))
            ax3.set_xlabel('gen')
            ax3.set_ylabel('y')

        for gen in range(num_gen):

            # Parent selection #############################
    
            if rho == 1:
                
                # Each child is made from one randomly selected parent
                selected_parent_indices = np.random.randint(mu, size=lmb)
            
            elif rho == mu:
            
                # Recombine all parents for each child
                raise NotImplemented() # TODO
            
            elif 1 < rho < mu:
            
                # Recombine rho randomly selected parents for each child
                raise NotImplemented() # TODO
            
            else:
            
                raise ValueError()
                
            #print("Parent selection")
            #display(selected_parent_indices)
            
            # Recombination ################################

            if rho == 1:
                
                # Each child is made from one randomly selected parent
                pop.iloc[children_indices] = pop.iloc[selected_parent_indices].values
                
            elif rho == mu:
                
                # Recombine all parents for each child
                raise NotImplemented() # TODO
                
            elif 1 < rho < mu:
                
                # Recombine rho randomly selected parents for each child
                raise NotImplemented() # TODO
                
            else:
                
                raise ValueError()
            
            pop.iloc[children_indices, y_col] = np.nan
            
            #print("Recombination")
            #display(pop)

            # Mutate children's sigma ######################
    
            pop.iloc[children_indices, sigma_col] *= np.exp(tau * np.random.normal(size=lmb))
            
            #print("Mutated children's sigma")
            #display(pop)

            # Mutate children's value ######################
    
            sigma_array = np.tile(pop.iloc[children_indices, sigma_col], [d,1]).T    # TODO: <- CHECK THIS !!!
            random_array = np.random.normal(size=[lmb,d])
            pop.iloc[children_indices, x_cols] += sigma_array * random_array

            #print("Mutated children's value")
            #display(pop)

            # Evaluate children ############################

            pop.iloc[children_indices, y_col] = objective_function(pop.iloc[children_indices, x_cols].values.T)

            #print("Evaluate children")
            #display(pop)

            if plot:
                color_str = matplotlib.colors.rgb2hex(cmap(float(gen) / num_gen))
                pop.plot.scatter(x="x0", y="x1", c=color_str, ax=ax1)
                pop.plot.scatter(x="sigma", y="y", c=color_str, loglog=True, ax=ax2)
                ax3.semilogy(np.full(shape=pop.y.shape, fill_value=gen), pop.y, '.', color=color_str)

            # Select the best individuals ##################

            if selection_operator == ',':
                pop.iloc[parent_indices] = np.nan
            
            pop = pop.sort_values(by=[y_col_label], na_position='last').reset_index(drop=True)
            
            pop.iloc[children_indices] = np.nan
            
            #print("Selected individuals for the next generation")
            #display(pop)

        if plot:
            plt.show()

        return pop.iloc[0, x_cols].values
