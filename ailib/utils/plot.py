#!/usr/bin/env python3
# coding: utf-8

"""
Plot functions.

This module contains some useful plot functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d

###############################################################################

def plot_2d_contour_solution_space(func,
                                   xmin=-np.ones(2),
                                   xmax=np.ones(2),
                                   xstar=None,
                                   title="",
                                   vmin=None,
                                   vmax=None,
                                   zlog=True,
                                   output_file_name=None):
    """TODO
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    x1_space = np.linspace(xmin[0], xmax[0], 200)
    x2_space = np.linspace(xmin[1], xmax[1], 200)
    
    x1_mesh, x2_mesh = np.meshgrid(x1_space, x2_space)

    zz = func(np.array([x1_mesh.ravel(), x2_mesh.ravel()])).reshape(x1_mesh.shape)
    
    ############################
    
    if xstar.ndim == 1:
        min_value = func(xstar)
    else:
        min_value = min(func(xstar))
    max_value = zz.max()
    
    if vmin is None:
        if zlog:
            vmin = 0.1             # TODO
        else:
            vmin = min_value
        
    if vmax is None:
        vmax = max_value
        
    if zlog:
        norm = colors.LogNorm()
    else:
        norm = None
    
    levels = np.logspace(0.1, 3., 5)          # TODO

    im = ax.pcolormesh(x1_mesh, x2_mesh, zz,
                       vmin=vmin,
                       vmax=vmax,
                       norm=norm,
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

    if xstar is not None:
        ax.scatter(xstar[0],
                   xstar[1],
                   c='red',
                   label="$x^*$")

    ax.set_title(title)

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.legend(fontsize=12)
    
    if output_file_name is not None:
        plt.savefig(output_file_name, transparent=True)

    plt.show()

###############################################################################

def plot_2d_solution_space(func,
                           xmin=-np.ones(2),
                           xmax=np.ones(2),
                           xstar=None,
                           angle_view=None,
                           title="",
                           zlog=True,
                           output_file_name=None):
    """TODO
    """
    fig = plt.figure(figsize=(12, 8))
    ax = axes3d.Axes3D(fig)
    
    if angle_view is not None:
        ax.view_init(angle_view[0], angle_view[1])

    x1_space = np.linspace(xmin[0], xmax[0], 100)
    x2_space = np.linspace(xmin[1], xmax[1], 100)
    
    x1_mesh, x2_mesh = np.meshgrid(x1_space, x2_space)

    zz = func(np.array([x1_mesh.ravel(), x2_mesh.ravel()])).reshape(x1_mesh.shape)   # TODO

    ############################
    
    if zlog:
        norm = colors.LogNorm()
    else:
        norm = None
        
    surf = ax.plot_surface(x1_mesh,
                           x2_mesh,
                           zz,
                           cmap='gnuplot2', # 'jet' # 'gnuplot2'
                           norm=norm,
                           rstride=1,
                           cstride=1,
                           #color='b',
                           shade=False)

    ax.set_zlabel(r"$f(x_1, x_2)$")

    fig.colorbar(surf, shrink=0.5, aspect=5)

