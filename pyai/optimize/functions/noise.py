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
"""

__all__ = ['GaussianNoise', 'additive_gaussian_noise', 'multiplicative_gaussian_noise',
           'UniformNoise', 'additive_uniform_noise', 'multiplicative_uniform_noise']

import numpy as np

class GaussianNoise:
    """
    Noise class for objective functions.

    TODO
    """
    def __init__(self, loc=0., scale=1., noise_type='additive'):
        self.noise_type = noise_type
        self.loc = loc
        self.scale = scale

    def __call__(self, x, y):
        """
        TODO
        """
        rvs = np.random.normal(loc=self.loc, scale=self.scale, size=y.shape)

        if self.noise_type == 'additive':
            y += rvs
        elif self.noise_type == 'multiplicative':
            y += y * rvs
        else:
            raise ValueError("Unknown value {}.".format(self.noise_type))

        return y

additive_gaussian_noise = GaussianNoise(loc=0., scale=1., noise_type='additive')
multiplicative_gaussian_noise = GaussianNoise(loc=0., scale=1., noise_type='multiplicative')


class UniformNoise:
    """
    Noise class for objective functions.

    TODO
    """
    def __init__(self, high=0., low=1., noise_type='additive'):
        self.noise_type = noise_type
        self.high = high
        self.low = low

    def __call__(self, x, y):
        """
        TODO
        """
        rvs = np.random.uniform(high=self.high, low=self.low, size=y.shape)

        if self.noise_type == 'additive':
            y += rvs
        elif self.noise_type == 'multiplicative':
            y += np.log(y * rvs)
        else:
            raise ValueError("Unknown value {}.".format(self.noise_type))

        return y

additive_uniform_noise = UniformNoise(high=0., low=1., noise_type='additive')
multiplicative_uniform_noise = UniformNoise(high=0., low=1., noise_type='multiplicative')
