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

# TODO: improve this ?
if __name__ == '__main__':
    import environment
else:
    from . import environment

class Environment(environment.Environment):
    """
    ...
    """

    # self.stateSet
    # self.actionSet
    # self.initialState
    # self.finalStateSet

    def __init__(self, initial_state = 's1', discount_factor=1):

        self.discountFactor = discount_factor                    # TODO: in environment or in agent ???
        assert 0 <= self.discountFactor <= 1                     # TODO

        # set of final states
        self.finalStateSet = {'s3'}

        # set of states
        self.stateSet = {'s1', 's2', 's3'}

        assert self.finalStateSet <= self.stateSet               # is finalStateSet subset of stateSet ?

        # set of actions
        self.actionSet = {'c', 'q'}

        # initial state
        assert initial_state in self.stateSet
        self.initialState = initial_state


    def reward(self, current_state, action=None, next_state=None):
        assert current_state in self.stateSet

        if current_state == 's3':
            reward = 1.
        else:
            reward = -0.1

        return reward


    def transition(self, current_state, action):
        assert current_state in self.stateSet
        assert action in self.actionSet

        if current_state == 's1' and action == 'c': 
            next_state_proba = {'s1':0.1, 's2':0.9, 's3':0.}

        elif current_state == 's1' and action == 'q': 
            next_state_proba = {'s1':0.1, 's2':0., 's3':0.9}

        elif current_state == 's2' and action == 'c': 
            next_state_proba = {'s1':0.9, 's2':0.1, 's3':0.}

        elif current_state == 's2' and action == 'q': 
            next_state_proba = {'s1':0., 's2':0.1, 's3':0.9}

        else: 
            raise Exception("Internal error")

        return next_state_proba


    # DEBUG FUNCTIONS #########################################################


    def displayStateAction(self, current_state, current_action=None, iteration=None):
        assert current_state in self.stateSet

        # TODO: graphviz


    def displayReward(self):
        pass


    def displayValueFunction(self, value_utility_dict, iteration=None):
        print(value_utility_dict)


    def displayPolicy(self, agent, iteration=None):
        print(agent.policy)


    def displayTransitionProbabilityDistribution(self, current_state, current_action):
        """
        Display the probability mass function for a given (state, action).
        """
        print(current_state, current_action)


def display_maze_with_cairo(num_col, num_row, text_dict=None, color_dict=None, bold_set=set(), inner_square_dict=dict(), text_sub_dict=dict(), title=None):
    pass


# TEST ########################################################################


class Agent():

    def __init__(self):
        # The optimal policy...
        self.policy = {'s1':'q', 's2':'q'}

    def getAction(self, state):
        action = self.policy[state]
        return action


def test():
    pass


if __name__ == '__main__':
    test()

