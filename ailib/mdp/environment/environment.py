#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2014,2015,2016,2017 Jeremie DECOCK (http://www.jdhp.org)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Scipy discrete probability distribution (used in doTransition function)
from scipy.stats import rv_discrete

class Environment(object):

    def reward(self, current_state, action=None, next_state=None):
        raise NotImplementedError

    def transition(self, current_state, action, next_state):
        raise NotImplementedError

    # doTransition is common to all Environment classes
    def doTransition(self, current_state, action):
        assert current_state in self.stateSet
        assert action in self.actionSet

        next_state_proba = self.transition(current_state, action)

        # randomly generate the next state acording to the next_state_proba distribution
        state_proba_list = [item for item in next_state_proba.items()]
        state_list = [item[0] for item in state_proba_list]
        proba_list = [item[1] for item in state_proba_list]

        # A Scipy probability distribution
        distrib = rv_discrete(values=(range(len(state_list)), proba_list))

        # One sample
        next_state_index = distrib.rvs()

        next_state = state_list[next_state_index]

        reward = self.reward(current_state, action, next_state)

        return (next_state, reward)


    def display(self, state=None):
        raise NotImplementedError


    def simulate(self, agent, initial_state=None, max_it=float("inf")):
        """
        max_it (maximum number of iterations) can be used to avoid infinites simulations.
        """
        if initial_state is None:
            initial_state = self.initialState
        else:
            assert initial_state in self.stateSet

        state_list = [initial_state]
        action_list = []
        reward_list = []

        state = initial_state

        while(state not in self.finalStateSet and len(action_list) < max_it):
            action = agent.getAction(state)
            (state, reward) = self.doTransition(state, action)

            state_list.append(state)
            action_list.append(action)
            #reward_list.append(reward)  # TODO

        reward_list = [self.reward(state) for state in state_list]  # TODO

        return (state_list, action_list, reward_list)

