#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Jérémie DECOCK (http://www.jdhp.org)

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

from . import agent

class Agent(agent.Agent):

    def __init__(self, environment, discount_factor= 0.5):
        # TODO: fix a bug: utility values are incorrect... cf. book p.692

        self.environment = environment
        self.discountFactor = discount_factor # TODO: in environment ???

        # Init value utility
        self.valueUtility = {state:0. for state in self.environment.stateSet}

        # Iteratively update self.valueUtility with the following function
        # U_{t+1}(s) := R(\s) + \gamma \max_{a} \sum_{s'} T(s, a, s') U_t(s')
        it = 0 # TODO
        while it < 30: # TODO
            value_utility = {}
            for state, value in self.valueUtility.items():
                (action, action_meu) = self.actionMaximumExpectedUtility(state)
                value_utility[state] = self.environment.reward(state) + self.discountFactor * action_meu

            #TODO : final state shoud keep their value (TODO: improve writing)
            for state in environment.finalStateSet:
                value_utility[state] = self.environment.reward(state)

            self.valueUtility = value_utility

            environment.displayValueFunction(value_utility, iteration=it)

            it += 1  # TODO

        # Build the policy
        self.policy = {}
        for state in self.environment.stateSet:
            (action, action_meu) = self.actionMaximumExpectedUtility(state)
            self.policy[state] = action


    def actionMaximumExpectedUtility(self, state):
        """
        Compute \pi^*(s) = \arg\max_{a} \sum_{s'} T(s, a, s') U_i(s')
        and         U(s) =     \max_{a} \sum_{s'} T(s, a, s') U_i(s')
        """

        best_action = None
        best_action_utility = float("-inf")

        for action in self.environment.actionSet:
            expected_action_utility = 0.
            for next_state in self.environment.stateSet:
                expected_action_utility += self.environment.transition(state, action)[next_state] * self.valueUtility[next_state]
            if expected_action_utility > best_action_utility:
                best_action = action
                best_action_utility = expected_action_utility

        return (best_action, best_action_utility)


    def getAction(self, state):
        """
        Returns the action to be performed by the agent for a given state.
        """
        (action, action_meu) = self.actionMaximumExpectedUtility(state)
        return action


if __name__ == '__main__':
    pass
