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

__all__ = ['sphere', 'Sphere', 'sphere1d', 'sphere2d',     # TODO
           'rosen',
           'himmelblau',
           'rastrigin',
           'easom',
           'crossintray',
           'holder']

import numpy as np

# GENERIC OBJECTIVE FUNCTION ##################################################

class _ObjectiveFunction(object):
    """Generic *objective function*.

    TODO
    """
    def __init__(self):
        self.reset_eval_counters()
        self.reset_eval_logs()
        self.do_eval_logs = False

        self.noise = None

        self.ndim = None
        self.bounds = None

        self.unimodal = None
        self.continuous = None

        self.function_name = None
        self.function_formula = None

        self.arg_min = None


    @property
    def stochastic(self):
        return self.noise is not None


    def reset_eval_counters(self):
        # TODO: make an external Log (or Counter) class
        self.num_eval = 0
        self.num_gradient_eval = 0
        self.num_hessian_eval = 0


    def reset_eval_logs(self):
        # TODO: make an external Log class
        self.eval_logs_dict = {'x': [], 'fx': []}  # TODO


    def _eval(self, func, x):
        """Evaluate one or several points.

        This function is a wrapper that does several boring task aside the
        evaluation of `func`: check arguments, log results, ...

        Parameters
        ----------
        func : callable object
            The function used to evaluate `x`.
        y : ndarray
            The 1D or 2D numpy array containing the points to evaluate.
            If `x` is a 2D array, the coordinates of each points are
            distributed along *the first dimension*.
            For instance, to evaluate the three 2D points (0,0), (1,1) and
            (2,2), `x` have to be coded as the following:
            `x = np.array([[0, 1, 2], [0, 1, 2]])`
            so that the first point is given by `x[:,0]`, the second point by
            `x[:,1]`, ... (this makes functions definition much simpler).

        Returns
        -------
        float or ndarray
            The results of the evaluation: a scalar if only one point has been
            evaluated or a 1D numpy array if several points have been
            evaluated.
        """
        # Check x shape #########################
        if x.ndim > 0:
            if x.shape[0] != self.ndim:
                raise Exception('Wrong number of dimension: x has {} rows instead of {}.'.format(x.shape[0], self.ndim))

        # Update the evaluations counter ########
        # TODO: make an external Log (or Counter) class
        if (x.ndim == 0) or (x.ndim == 1):
            self.num_eval += 1
        elif x.ndim == 2:
            self.num_eval += x.shape[1]
        else:
            raise Exception('Wrong number of dimension: x is a {} dimensions numpy array ; 1 or 2 dimensions are expected.'.format(x.ndim))

        # Eval x ################################
        y = func(x)

        # Apply noise ###########################
        if self.noise is not None:
            y = self.noise(x, y)

        # Update the evals log ##################
        # TODO: make an external Log class
        if self.do_eval_logs:
            if y.ndim == 0:
                self.eval_logs_dict['x'].append(x)       # TODO
            elif y.ndim == 1:
                self.eval_logs_dict['x'].extend(x.T)     # TODO
            else:
                raise Exception("Wrong output dimension.")

            if y.ndim == 0:
                self.eval_logs_dict['fx'].append(y)    # TODO
            elif y.ndim == 1:
                self.eval_logs_dict['fx'].extend(y)    # TODO
            else:
                raise Exception("Wrong output dimension.")

        return y


    def _eval_gradient(self, gradient_func, x):
        """
        TODO
        """
        # Check x shape #########################
        if x.shape[0] != self.ndim:
            raise Exception('Wrong number of dimension: x has {} rows instead of {}.'.format(x.shape[0], self.ndim))

        # Update the evaluations counter ########
        # TODO: make an external Log (or Counter) class
        if x.ndim == 1:
            self.num_gradient_eval += 1
        elif x.ndim == 2:
            self.num_gradient_eval += x.shape[1]
        else:
            raise Exception('Wrong number of dimension: x is a {} dimensions numpy array ; 1 or 2 dimensions are expected.'.format(x.ndim))

        # Eval x ################################
        grad = gradient_func(x)

        return grad


    def _eval_hessian(self, hessian_func, x):
        """
        TODO
        """
        # Check x shape #########################
        if x.shape[0] != self.ndim:
            raise Exception('Wrong number of dimension: x has {} rows instead of {}.'.format(x.shape[0], self.ndim))

        # Update the evaluations counter ########
        # TODO: make an external Log (or Counter) class
        if x.ndim == 1:
            self.num_hessian_eval += 1
        elif x.ndim == 2:
            self.num_hessian_eval += x.shape[1]
        else:
            raise Exception('Wrong number of dimension: x is a {} dimensions numpy array ; 1 or 2 dimensions are expected.'.format(x.ndim))

        # Eval x ################################
        hess = hessian_func(x)

        return hess


    def __str__(self):
        name = r""

        if self.stochastic is not None:
            name += "stochastic "

        if self.function_name is not None:
            name += self.function_name
        else:
            name += self.__class__.__name__

        if self.function_formula is not None:
            name += ": " + self.function_formula

        return name

# SPHERE FUNCTION #############################################################

def sphere(x):
    r"""The Sphere function.

    The Sphere function is a famous **convex** function used to test the performance of optimization algorithms.
    This function is very easy to optimize and can be used as a first test to check an optimization algorithm.

    $$
    f(\boldsymbol{x}) = \sum_{i=1}^{n} x_{i}^2
    $$

    Global minimum:
    $$
    f(\boldsymbol{0}) = 0
    $$

    Search domain:
    $$
    \boldsymbol{x} \in \mathbb{R}^n
    $$

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> sphere( np.array([0, 0]) )
    0.0

    The result should be $f(x) = 0$.

    Example: single 3D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$:

    >>> sphere( np.array([1, 1, 1]) )
    3.0

    The result should be $f(x) = 3.0$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$ at once:

    >>> sphere( np.array([[0, 1, 2], [0, 1, 2]]) )
    array([   0.,    2.,  8.])

    The result should be $f(x_1) = 0$, $f(x_2) = 1$ and $f(x_3) = 8$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Sphere function is to be computed
        or a two dimension Numpy array of points at which the Sphere function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Sphere function for the given point(s) `x`.
    """
    # Remark: `sum(x**2.0)` is equivalent to `np.sum(x**2.0, axis=0)` but only the latter works if x is a scallar (e.g. x = np.float(3)).
    return np.sum(x**2.0, axis=0)


def sphere_gradient(x):
    """
    TODO
    """
    return 2.0 * x


def sphere_hessian(x):
    """
    TODO
    """
    return 2.0 * np.ones(x.shape)


class Sphere(_ObjectiveFunction):
    """
    TODO
    """
    def __init__(self, ndim):
        super().__init__()

        self.ndim = ndim

        self.bounds = np.ones((2, self.ndim))    # TODO: take this or the transpose of this ?
        self.bounds[0,:] = -10.
        self.bounds[1,:] =  10.

        self.unimodal = True
        self.continuous = True

        self.arg_min = np.zeros(self.ndim)

    def __call__(self, x):
        """
        TODO
        """
        return super()._eval(sphere, x)

    def gradient(self, x):
        """
        TODO
        """
        return super()._eval_gradient(sphere_gradient, x)

    def hessian(self, x):
        """
        TODO
        """
        return super()._eval_hessian(sphere_hessian, x)


sphere1d = Sphere(ndim=1)

sphere2d = Sphere(ndim=2)

# ROSENBROCK FUNCTION #########################################################

def rosen(x):
    r"""The (extended) Rosenbrock function.

    The Rosenbrock function is a famous **non-convex** function used to test
    the performance of optimization algorithms. The classical two-dimensional
    version of this function is **unimodal** but its *extended* $n$-dimensional
    version (with $n \geq 4$) is **multimodal**
    [[ref.](http://www.mitpressjournals.org/doi/abs/10.1162/evco.2006.14.1.119)].

    $$
    f(\boldsymbol{x}) = \sum_{i=1}^{n-1} \left[100 \left( x_{i+1} - x_{i}^{2} \right)^{2} + \left( x_{i} - 1 \right)^2 \right]
    $$

    Global minimum:
    $$
    \min =
    \begin{cases}
        n = 2 & \rightarrow \quad f(1,1) = 0, \\
        n = 3 & \rightarrow \quad f(1,1,1) = 0, \\
        n > 3 & \rightarrow \quad f(\underbrace{1,\dots,1}_{n{\text{ times}}}) = 0 \\
    \end{cases}
    $$

    Search domain:
    $$
    \boldsymbol{x} \in \mathbb{R}^n
    $$

    The Rosenbrock has exactly one (global) minimum $(\underbrace{1, \dots,
    1}_{n{\text{ times}}})^\top$ for $n \leq 3$ and an additional *local*
    minimum for $n \geq 4$ near $(-1, 1, 1, \dots, 1)^\top$.

    See http://www.mitpressjournals.org/doi/abs/10.1162/evco.2006.14.1.119
    (freely available at http://dl.acm.org/citation.cfm?id=1118014) and
    https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalisations
    for more information.

    See https://en.wikipedia.org/wiki/Rosenbrock_function and
    http://mathworld.wolfram.com/RosenbrockFunction.html for more information.

    The Rosenbrock function, its derivative (i.e. gradient) and its hessian matrix are also implemented in Scipy
    ([scipy.optimize.rosen](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen.html#scipy.optimize.rosen),
    [scipy.optimize.rosen_der](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_der.html#scipy.optimize.rosen_der),
    [scipy.optimize.rosen_hess](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess.html#scipy.optimize.rosen_hess) and
    [scipy.optimize.rosen_hess_prod](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.rosen_hess_prod.html#scipy.optimize.rosen_hess_prod)).
    See [Scipy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.html#rosenbrock-function) for more information.

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> rosen( np.array([0, 0]) )
    1.0

    The result should be $f(x) = 1$.

    Example: single 3D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$:

    >>> rosen( np.array([1, 1, 1]) )
    0.0

    The result should be $f(x) = 0$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$ at once:

    >>> rosen( np.array([[0, 1, 2], [0, 1, 2]]) )
    array([   1.,    0.,  401.])

    The result should be $f(x_1) = 1$, $f(x_2) = 0$ and $f(x_3) = 401$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Rosenbrock function is to be computed
        or a two dimension Numpy array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Rosenbrock function for the given point(s) `x`.
    """
    return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)

# HIMMELBLAU'S FUNCTION #######################################################

def himmelblau(x):
    r"""The Himmelblau's function.

    The Himmelblau's function is a two-dimensional **multimodal** function.

    $$
    f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2
    $$

    The function has four global minima:
    $$
    \begin{eqnarray}
        f(3, 2) = 0 \\
        f(-2.805118, 3.131312) = 0 \\
        f(-3.779310, -3.283186) = 0 \\
        f(3.584428, -1.848126) = 0
    \end{eqnarray}
    $$

    Search domain:
    $$
    \boldsymbol{x} \in \mathbb{R}^2
    $$

    It also has one local maximum at $f(-0.270845, -0.923039) = 181.617$.

    The locations of all the minima can be found analytically (roots of cubic
    polynomials) but expressions are somewhat complicated.

    The function is named after David Mautner Himmelblau, who introduced it in
    *Applied Nonlinear Programming* (1972), McGraw-Hill, ISBN 0-07-028921-2.

    See https://en.wikipedia.org/wiki/Himmelblau%27s_function for more information.

    Example: single point
    ---------------------

    To evaluate $x = \begin{pmatrix} 3 \\ 2 \end{pmatrix}$:

    >>> himmelblau( np.array([3, 2]) )
    0.0

    The result should be $f(x) = 1$.

    Example: multiple points
    ------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$ at once:

    >>> himmelblau( np.array([[0, 1, 2], [0, 1, 2]]) )
    array([ 170.,  106.,   26.])

    The result should be $f(x_1) = 170$, $f(x_2) = 106$ and $f(x_3) = 26$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Himmelblau's function is to be computed
        or a two dimension Numpy array of points at which the Himmelblau's function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Himmelblau's function for the given point(s) `x`.
    """
    assert x.shape[0] == 2, x.shape
    return (x[0]**2.0 + x[1] - 11.0)**2.0 + (x[0] + x[1]**2.0 - 7.0)**2.0

# RASTRIGIN FUNCTION ##########################################################

def rastrigin(x):
    r"""The Rastrigin function.

    The Rastrigin function is a famous **multimodal** function.
    Finding the minimum of this function is a fairly difficult problem due to
    its large search space and its large number of local minima.

    The classical two-dimensional version of this function has been introduced
    by L. A. Rastrigin in *Systems of extremal control* Mir, Moscow (1974).

    Its *generalized* $n$-dimensional version has been proposed by H.
    Mühlenbein, D. Schomisch and J. Born in *The Parallel Genetic Algorithm as
    Function Optimizer* Parallel Computing, 17, pages 619–632, 1991.

    On an n-dimensional domain it is defined by:

    $$
    f(\boldsymbol{x}) = An + \sum_{i=1}^{n} \left[ x_{i}^{2} - A \cos(2 \pi x_{i}) \right]
    $$
    where $A = 10$.

    Global minimum:
    $$
    f(\boldsymbol{0}) = 0
    $$

    Search domain:
    $$
    \boldsymbol{x} \in \mathbb{R}^n
    $$

    See https://en.wikipedia.org/wiki/Rastrigin_function for more information.

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> rastrigin( np.array([0, 0]) )
    0.0

    The result should be $f(x) = 0$.

    Example: single 3D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$:

    >>> rastrigin( np.array([0, 0, 0]) )
    0.0

    The result should be $f(x) = 0$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$ at once:

    >>> rastrigin( np.array([[0, 1, 2], [0, 1, 2]]) )
    array([   1.,    0.,  401.])

    The result should be $f(x_1) = 1$, $f(x_2) = 0$ and $f(x_3) = 401$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Rastrigin function is to be computed
        or a two dimension Numpy array of points at which the Rastrigin function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Rastrigin function for the given point(s) `x`.
    """
    A = 10.
    n = x.shape[0]
    return A * n + np.sum(x**2.0 - A * np.cos(2.0 * np.pi * x), axis=0)

# EASOM FUNCTION ##############################################################

def easom(x):
    r"""The Easom function.

    The Easom function is a 2 dimensions **unimodal** function.

    $$
    f(x_1, x_2) = -\cos(x_1) \cos(x_2) \exp \left( -\left[ (x_1-\pi)^2 + (x_2-\pi)^2 \right] \right)
    $$

    Global minimum:
    $$
    f(\pi, \pi) = -1
    $$

    Search domain:
    $$
    \boldsymbol{x} \in \mathbb{R}^2
    $$

    See https://www.sfu.ca/~ssurjano/easom.html for more information.

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> easom( np.array([np.pi, np.pi]) )
    -1.0

    The result should be $f(x) = -1$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} \pi \\ \pi \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ at once:

    >>> easom( np.array([[np.pi, 0, 1], [np.pi, 0, 1]]) )
    array([   -1.,    -2.67528799e-09,  -3.03082341e-05])

    The result should be $f(x_1) = -1$, $f(x_2) \approx 0$ and $f(x_3) \approx 0$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Easom function is to be computed
        or a two dimension Numpy array of points at which the Easom function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Easom function for the given point(s) `x`.
    """
    assert x.shape[0] == 2, x.shape
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0]-np.pi)**2.0 + (x[1]-np.pi)**2.0))

# CROSS-IN-TRAY FUNCTION ######################################################

def crossintray(x):
    r"""The Cross-in-tray function.

    The Cross-in-tray function is a 2 dimensions **multimodal** function, with
    four global minima.

    $$
    f(x_1, x_2) = -0.0001 \left( \left| \sin(x_1) \sin(x_2) \exp \left( \left| 100 - \frac{\sqrt{x_1^2 + x_2^2}}{\pi} \right| \right)\right| + 1 \right)^{0.1}
    $$

    Global minima:

    $$
    \text{Min} =
    \begin{cases}
        f(1.34941, -1.34941)  &= -2.06261 \\
        f(1.34941, 1.34941)   &= -2.06261 \\
        f(-1.34941, 1.34941)  &= -2.06261 \\
        f(-1.34941, -1.34941) &= -2.06261 \\
    \end{cases}
    $$

    Search domain:
    $$
    -10 \leq x_1, x_2 \leq 10
    $$

    **References**: *Test functions for optimization* (Wikipedia):
    https://en.wikipedia.org/wiki/Test_functions_for_optimization.

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> crossintray( np.array([0, 0]) )
    -0.0001

    The result should be $f(x) = -0.0001$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 1.34941 \\ 1.34941 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} -1.34941 \\ -1.34941 \end{pmatrix}$ at once:

    >>> crossintray( np.array([[0, 1.34941, -1.34941], [0, 1.34941, -1.34941]]) )
    array([ -0.0001,    -2.06261,  -2.06261])

    The result should be $f(x_1) = -0.0001$, $f(x_2) = -2.06261$ and $f(x_3) = -2.06261$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Cross-in-tray function is to be computed
        or a two dimension Numpy array of points at which the Cross-in-tray function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Cross-in-tray function for the given point(s) `x`.
    """
    assert x.shape[0] == 2, x.shape
    return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp( np.abs( 100.0 - np.sqrt(x[0]**2.0 + x[1]**2.0)/np.pi ))) + 1.0)**0.1

# HÖLDER TABLE FUNCTION #######################################################

def holder(x):
    r"""The Hölder table function.

    The Hölder table function is a 2 dimensions **multimodal** function, with
    four global minima.

    $$
    f(x_1, x_2) =
    -\left| \sin(x_1) \cos(x_2) \exp \left( \left| 1 - \frac{\sqrt{x_1^2 + x_2^2}}{\pi} \right| \right) \right|
    $$

    Global minima:
    $$
    \text{Min} =
    \begin{cases}
        f(8.05502, 9.66459)   &= -19.2085 \\
        f(-8.05502, 9.66459)  &= -19.2085 \\
        f(8.05502, -9.66459)  &= -19.2085 \\
        f(-8.05502, -9.66459) &= -19.2085
    \end{cases}
    $$

    Search domain:
    $$
    -10 \leq x_1, x_2 \leq 10
    $$

    **References**: *Test functions for optimization* (Wikipedia):
    https://en.wikipedia.org/wiki/Test_functions_for_optimization.

    Example: single 2D point
    ------------------------

    To evaluate $x = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$:

    >>> holder( np.array([0, 0]) )
    0.0

    The result should be $f(x) = 0$.

    Example: multiple 2D points
    ---------------------------

    To evaluate $x_1 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$,
    $x_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$ and
    $x_3 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$ at once:

    >>> holder( np.array([[0., 0., 1.], [0., 1., 0.]]) )
    array([-0. , -0. , -1.66377043])

    The result should be $f(x_1) = 0$, $f(x_2) = 0$ and $f(x_3) = -1.66377043$.

    Parameters
    ----------
    x : array_like
        One dimension Numpy array of the point at which the Hölder table function is to be computed
        or a two dimension Numpy array of points at which the Hölder table function is to be computed.

    Returns
    -------
    float or array_like
        The value(s) of the Hölder table function for the given point(s) `x`.
    """
    assert x.shape[0] == 2, x.shape
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1.0 - np.sqrt(x[0]**2.0 + x[1]**2.0)/np.pi )))

