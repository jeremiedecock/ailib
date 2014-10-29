# -*- coding: utf-8 -*-

# Copyright (c) 2013 Jérémie DECOCK (http://www.jdhp.org)

__all__ = ['ObjectiveFunction']

import numpy as np

class ObjectiveFunction(object):

    def __call__(self, *pargs, **kargs):
        x = pargs[0]

        if x.ndim == 1:
            # Only one sample
            assert x.shape[0] == self.ndim, x
            y = self._eval_one_sample(x)
            #assert type(y) == type(float), type(y)
        else:
            # Multiple samples
            assert x.shape[1] == self.ndim, x
            y = self._eval_multiple_samples(x)
            assert y.ndim == 2, y
            assert y.shape == (x.shape[0], 1), y.shape

        return y

    def _eval_one_sample(self, x):
        raise NotImplementedError

    # This function can be redefined to speedup computations
    def _eval_multiple_samples(self, x):
        y = []
        for xi in x:
            y.append(self._eval_one_sample(xi))
        return np.array(y).reshape([-1,1])

    def plot(self, xmin=-10, xmax=10, xstep=0.01):
        if self.ndim == 1:
            import matplotlib.pyplot as plt

            x_vec = np.arange(xmin, xmax, xstep)
            y_vec = self(x_vec.reshape([-1, 1]))

            try:
                label = self.label
            except:
                label = "f(x)"

            fig = plt.figure(figsize=(16.0, 10.0))
            ax = fig.add_subplot(111)
            ax.plot(x_vec, y_vec, "-", label=label)

            # TITLE AND LABELS
            ax.set_title("Objective function", fontsize=20)
            ax.set_xlabel(r"$x$", fontsize=32)
            ax.set_ylabel(r"$f(x)$", fontsize=32)

            # LEGEND
            ax.legend(loc='lower right', fontsize=20)

            # SAVE FILES ######################
            #filename = label + ".pdf"
            #plt.savefig(filename)

            # PLOT ############################
            plt.show()
        elif self.ndim == 2:
            pass # TODO
        else:
            pass


# TODO: doit être un objet qui permet de connaître:
# - le dommaine de définition de x
#   - continu / discret ?
#   - contraint ou non
# - le nombre de dimensions de x
# - minimisation ou maximisation
