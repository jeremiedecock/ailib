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
==============================
PyAI - Optimization algorithms
==============================

This package contains some optimization algorithms.

.. currentmodule:: ailib.optimize.minimizers

Algorithms
==========

.. toctree::

    saes

"""

# The following lines are inspired by https://github.com/scipy/scipy/blob/master/scipy/optimize/__init__.py

from .gd import *
from .random import *
from .saes import *

__all__ = [s for s in dir() if not s.startswith('_')]

#print("ailib.optimize.minimizers.__init__.py:", __all__)
