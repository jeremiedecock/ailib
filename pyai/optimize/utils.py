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
TODO

**TODO**:

Plots:
- [x] visited solutions in the solution space (2D plots with pcolormesh and contour)
- [x] f(x*) w.t. iteration number (well... actually this one is pretty useless...)
- [x] error w.t. iteration number (=> + fit to have a clear view of the convergence rate ?)
- [x] error w.t. number of function evaluations + error w.t. *total* number of function evaluations (i.e. including the number of gradient and hessian evaluations)
- [x] error w.t. execution time
- [ ] error w.t. ... => add an option to choose between the current solution or the best current solution 
- [ ] (benchmark session ! distinguish the derivative-free to the non-derivative free case) average version of 3., 4., 5. over several runs with random initial state (+ error bar or box plot)
- [ ] (benchmark session) err w.t. algorithms parameters (plot the iteration or evaluation number or execution time to reach in average an error lower than N% with e.g. N=99%)
"""

__all__ = ['array_list_to_array',
           'plot_contour_2d_solution_space',
           'plot_2d_solution_space',
           'plot_fx_wt_iteration_number',
           'plot_err_wt_iteration_number',
           'plot_err_wt_execution_time',
           'plot_err_wt_num_feval']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d

def array_list_to_array(y_array_list):
    """Convert a sequence of 1D arrays (with possibly different sizes) to a 2D array.

    When given 1D arrays don't have the same size, "missing" items at the end
    of each 1D array are replaced by np.nan.

    Parameters
    ----------
    y_array_list : sequence of ndarray
        The sequence of 1D numpy arrays (with possibly different sizes) to
        join.

    Returns
    -------
    ndarray
        The 2D numpy array made from the concatenation `y_array_list` items.
    """
    max_size = max((y_array.shape[0] for y_array in y_array_list))
    y_array = np.full((len(y_array_list), max_size), np.nan)

    for index, y_array_i in enumerate(y_array_list):
        y_array[index,:y_array_i.shape[0]] = y_array_i

    return y_array


def plot_1d(y,
            x=None,
            ax=None,
            x_label=None,
            y_label=None,
            legend=None,
            title=None,
            x_log=False,
            y_log=False,
            plot_option=None,
            **kwargs):
    """Make a 1D plot.

    Parameters
    ----------
    y : ndarray
        The 1D or 2D numpy array containing the data's y-coordinate (ordinate)
        to plot. If `y` is a 2D array, it is aggregated *on the first dimension*
        (e.g. `y = np.nanmean(y, axis=0)`) which means the `y` array should be
        defined like the following:
        `y = np.array([sample_1, sample_2, ..., sample_n])`
        where `sample_i` is a 1D numpy array containing for instance the error
        of (the ith execution of) a minimizer w.t. the evaluation number
        (i.e. `sample_1` is the errors of a first execution of the algorithm,
        `sample_2` is the errors of a second execution of the algorithm,
        ...).
    x : ndarray
        The 1D numpy array containing the data's x-coordinate (abscissa) to
        plot.
    ax : AxesSubplot
        The Matplotlib axis object to plot on. If `ax` is `None`, a new axis is
        created and automatically shown. Otherwise, the `ax` object is updated
        with the plot but is not automatically shown.
    x_label : string
        The label for the x-axis.
    y_label : string
        The label for the y-axis.
    legend : string
        The legend label.
    title : string
        The plot title.
    x_log : bool
        If `True`, a log scale is used on the x-axis.
    y_log : bool
        If `True`, a log scale is used on the y-axis.
    plot_option : string
        Define the aggregation method if `y` is a 2D array. Possible values are
        "mean" or "median".
    **kwargs
        Any arbitrary keyword arguments accepted by `matplotlib.pyplot.plot()`.
    """
    if y.ndim == 2:
        # Aggregate data
        if plot_option == 'mean':
            y = np.nanmean(y, axis=0)
        elif plot_option == 'median':
            y = np.nanmedian(y, axis=0)
        else:
            raise ValueError("Two dimension arrays require a `plot_option` to define the aggregation method.")
    elif y.ndim != 1:
        raise ValueError("Wrong number of dimensions: {}.".format(y.ndim))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if x_log:
        ax.set_xscale("log", nonposx='clip')   # TODO

    if y_log:
        ax.set_yscale("log", nonposy='clip')   # TODO

    if x is None:
        ax.plot(y, **kwargs)
    else:
        ax.plot(x, y, **kwargs)

    if legend is not None:
        ax.legend(loc='best')

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if ax is None:
        plt.show()


def plot_fx_wt_iteration_number(fx_array, **kwargs):
    """Plot f(x*) w.t. iteration number.

    TODO
    """
    plot_1d(y=fx_array,
            x_label="Iteration number",
            y_label="f(x)",
            **kwargs)


def plot_err_wt_iteration_number(error_array, **kwargs):
    """Plot error w.t. iteration number.

    TODO
    """
    plot_1d(y=error_array,
            x_label="Iteration number",
            y_label="Error",
            **kwargs)


def plot_err_wt_execution_time(error_array, time_array, **kwargs):
    """Plot error w.t. execution time.

    TODO
    """
    plot_1d(y=error_array,
            x=time_array,
            x_label="Time (sec)",
            y_label="Error",
            **kwargs)


def plot_err_wt_num_feval(error_array, num_eval_array=None, **kwargs):
    """Plot error w.t. number of function evaluations.

    TODO
    """
    plot_1d(y=error_array,
            x=num_eval_array,
            x_label="Num fonction evaluations",
            y_label="Error",
            **kwargs)


def plot_contour_2d_solution_space(func,
                                   fig=None,
                                   ax=None,
                                   show=True,
                                   xmin=-np.ones(2),
                                   xmax=np.ones(2),
                                   xstar=None,
                                   xvisited=None,
                                   title=""):
    """Plot points visited during the execution of an optimization algorithm.

    TODO
    """
    if (fig is None) or (ax is None):                # TODO
        fig, ax = plt.subplots(figsize=(12, 8))

    if xvisited is not None:
        xmin = np.amin(np.hstack([xmin.reshape([-1, 1]), xvisited]), axis=1)
        xmax = np.amax(np.hstack([xmax.reshape([-1, 1]), xvisited]), axis=1)

    x1_space = np.linspace(xmin[0], xmax[0], 200)
    x2_space = np.linspace(xmin[1], xmax[1], 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_space, x2_space)

    zz = func(np.array([x1_mesh.ravel(), x2_mesh.ravel()])).reshape(x1_mesh.shape)

    ############################

    min_value = func(xstar)
    max_value = zz.max()

    levels = np.logspace(0.1, 3., 5)          # TODO

    im = ax.pcolormesh(x1_mesh, x2_mesh, zz,
                       vmin=0.1,              # TODO
                       vmax=max_value,
                       norm=colors.LogNorm(), # TODO
                       shading='gouraud',
                       cmap='gnuplot2') # 'jet' # 'gnuplot2'

    plt.colorbar(im, ax=ax)

    cs = plt.contour(x1_mesh, x2_mesh, zz, levels,
                     linewidths=(2, 2, 2, 2, 3),
                     linestyles=('dotted', '-.', 'dashed', 'solid', 'solid'),
                     alpha=0.5,
                     colors='white')
    ax.clabel(cs, inline=False, fontsize=12)

    ############################

    if xvisited is not None:
        ax.plot(xvisited[0],
                xvisited[1],
                '-og',
                alpha=0.5,
                label="$visited$")

    ############################

    if xstar is not None:
        sc = ax.scatter(xstar[0],
                   xstar[1],
                   c='red',
                   label="$x^*$")
        sc.set_zorder(10)        # put this point above every thing else

    ############################

    ax.set_title(title)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.legend(fontsize=12)

    if show:
        plt.show()

    return fig, ax


def plot_2d_solution_space(func,
                           fig=None,
                           ax=None,
                           show=True,
                           xmin=-np.ones(2),
                           xmax=np.ones(2),
                           xstar=None,
                           xvisited=None,
                           angle_view=None,
                           title=""):
    """Plot points visited during the execution of an optimization algorithm.

    TODO
    """
    if fig is None or ax is None:                # TODO
        fig = plt.figure(figsize=(12, 8))
        ax = axes3d.Axes3D(fig)

    if angle_view is not None:
        ax.view_init(angle_view[0], angle_view[1])

    x1_space = np.linspace(xmin[0], xmax[0], 100)
    x2_space = np.linspace(xmin[1], xmax[1], 100)

    x1_mesh, x2_mesh = np.meshgrid(x1_space, x2_space)

    zz = func(np.array([x1_mesh.ravel(), x2_mesh.ravel()])).reshape(x1_mesh.shape)   # TODO

    ############################

    surf = ax.plot_surface(x1_mesh,
                           x2_mesh,
                           zz,
                           cmap='gnuplot2', # 'jet' # 'gnuplot2'
                           norm=colors.LogNorm(),   # TODO
                           rstride=1,
                           cstride=1,
                           #color='b',
                           shade=False)

    ax.set_zlabel(r"$f(x_1, x_2)$")

    fig.colorbar(surf, shrink=0.5, aspect=5)

    ############################

    if xstar is not None:
        ax.scatter(xstar[0],
                   xstar[1],
                   func(xstar),
                   #s=50,          # TODO
                   c='red',
                   alpha=1,
                   label="$x^*$")

    ax.set_title(title)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.legend(fontsize=12)

    if show:
        plt.show()

    return fig, ax
