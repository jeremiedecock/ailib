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

__all__ = ['plot_contour_2d_solution_space',
           'plot_2d_solution_space',
           'plot_fx_wt_iteration_number',
           'plot_err_wt_iteration_number',
           'plot_err_wt_execution_time',
           'plot_err_wt_num_feval']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d

def plot_fx_wt_iteration_number(fx_seq,
                                ax=None):
    """
    Plot f(x*) w.t. iteration number.

    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogy(fx_seq)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("f(x)")

    if ax is None:
        plt.show()


def plot_err_wt_iteration_number(error_seq,
                                 ax=None):
    """
    Plot error w.t. iteration number.

    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogy(error_seq)
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Error")

    if ax is None:
        plt.show()


def plot_err_wt_execution_time(error_seq,
                               time_seq,
                               ax=None):
    """
    Plot error w.t. execution time.

    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.semilogy(time_seq, error_seq)
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Error")

    if ax is None:
        plt.show()


def plot_err_wt_num_feval(error_seq,
                          num_eval_seq=None,
                          ax=None):
    """
    Plot error w.t. number of function evaluations.

    TODO
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if num_eval_seq is not None:
        ax.loglog(num_eval_seq, error_seq)
    else:
        ax.loglog(error_seq)

    ax.set_xlabel("Num fonction evaluations")
    ax.set_ylabel("Error")

    if ax is None:
        plt.show()


def plot_contour_2d_solution_space(func,
                                   fig=None,
                                   ax=None,
                                   show=True,
                                   xmin=-np.ones(2),
                                   xmax=np.ones(2),
                                   xstar=None,
                                   xvisited=None,
                                   title=""):
    """
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
    """
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
