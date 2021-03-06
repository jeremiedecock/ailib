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

from . import agent

class Agent(agent.Agent):
    """
    Value Iteration.
    One of the two main DP algorithm to solve MDP.
    The algorithm terminates after a given number of iterations.

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 691-696.
    """

    def __init__(self, environment, maximum_iteration = 30):

        self.environment = environment

        # Init value utility to 0
        self.valueUtility = {state:0. for state in self.environment.stateSet}

        # Iteratively update self.valueUtility with the following function
        # V_{i+1}(s) := R(s) + discount \max_{a} \sum_{s'} T(s, a, s') U_i(s')
        for iteration in range(maximum_iteration):
            next_value_utility = {}      # init V'

            # For all s
            for state in self.environment.stateSet:
                if state in environment.finalStateSet:
                    # If s is a final state then V_i(s)=R(s) for all i      (TODO: this is not explicitely written in most references...)
                    next_value_utility[state] = self.environment.reward(state)
                else:
                    # Compute \pi^*(s) := \arg\max_{a} \sum_{s'} T(s, a, s') V_i(s')
                    # and         V(s) :=     \max_{a} \sum_{s'} T(s, a, s') V_i(s')
                    (action, action_meu) = self.actionMaximumExpectedUtility(state)

                    # Compute V_{i+1}(s) := R(s) + discount \max_{a} \sum_{s'} T(s, a, s') V_i(s')
                    next_value_utility[state] = self.environment.reward(state) + self.environment.discountFactor * action_meu

            self.valueUtility = next_value_utility

            environment.displayValueFunction(self.valueUtility, iteration=iteration)

        # Build the policy
        self.policy = {}
        for state in self.environment.stateSet:
            (action, action_meu) = self.actionMaximumExpectedUtility(state)
            self.policy[state] = action


    def actionMaximumExpectedUtility(self, state):
        """
        Compute \pi^*(s) = \arg\max_{a} \sum_{s'} T(s, a, s') V_i(s')
        and         V(s) =     \max_{a} \sum_{s'} T(s, a, s') V_i(s')
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
