#!/usr/bin/env python3
# coding: utf-8

"""
================================================
Optimization Benchmark: Plot the Sphere Function
================================================

This example show how to plot the *Sphere function*.
"""

###############################################################################
# Import required packages

import numpy as np
import matplotlib.pyplot as plt

from ailib.utils.plot import plot_2d_contour_solution_space, plot_2d_solution_space
from ailib.optimize.functions.unconstrained import sphere2d as sphere

###############################################################################
# Plot the sphere function

plot_2d_solution_space(sphere,
                       xmin=-2*np.ones(2),
                       xmax=2*np.ones(2),
                       xstar=np.zeros(2),
                       angle_view=(55, 83),
                       title="Sphere function")

plt.tight_layout()

plt.show()

###############################################################################
# Plot the contours

plot_2d_contour_solution_space(sphere,
                               xmin=-10*np.ones(2),
                               xmax=10*np.ones(2),
                               xstar=np.zeros(2),
                               title="Sphere function")

plt.tight_layout()

plt.show()

