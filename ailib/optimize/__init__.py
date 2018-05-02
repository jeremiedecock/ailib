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
=============================
PyAI - Optimization Framework
=============================

This package contains the PyAI Optimization Framework.

This package implements some optimization algorithms and some test case
functions.
The aim of optimization algorithms is to find the minimum (or maximum if
stated) of a given function.


.. currentmodule:: ailib.optimize

Optimization
============

Local Optimization
------------------

.. toctree::

    optimize.minimizers.gd


Global Optimization
-------------------

.. toctree::

    optimize.minimizers.saes


Test functions
==============


"""

#__all__ = ['functions',
#           'minimizers']

# The following lines are inspired by https://github.com/scipy/scipy/blob/master/scipy/optimize/__init__.py

from .minimizers import *

__all__ = [s for s in dir() if not s.startswith('_')]

#print("ailib.optimize.__init__.py:", __all__)
