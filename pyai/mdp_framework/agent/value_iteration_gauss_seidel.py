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
    """
    Value Iteration (VI).
    One of the two main DP algorithm to solve MDP.
    The algorithm terminates after a given number of iterations.

    Gauss-Seidel variant: use V_{i+1}(s) instead V_i(s) in Bellman updates if
    already computed. In standard Value Iteration, the same current cost
    estimate V is used to update all states at the next iterate. Instead, here
    we incorporate earlier in the computations the new values of the cost for
    the states that have already been treated.
    This version is a bit faster than the original VI algorithm.

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 691-696.
    See: Perny's courses for the Gauss-Seidel variant.
    """

    def __init__(self, environment, maximum_iteration = 30):

        self.environment = environment

        # Init value utility to 0
        self.valueUtility = {state:0. for state in self.environment.stateSet}

        # Iteratively update self.valueUtility with the following function
        # V_{i+1}(s) := R(s) + discount \max_{a} \sum_{s'} T(s, a, s') U_i(s')
        for iteration in range(maximum_iteration):
            # For all s
            for state in self.environment.stateSet:
                if state in environment.finalStateSet:
                    # If s is a final state then V_i(s)=R(s) for all i      (TODO: this is not explicitely written in most references...)
                    self.valueUtility[state] = self.environment.reward(state)
                else:
                    # Compute \pi^*(s) := \arg\max_{a} \sum_{s'} T(s, a, s') U_i(s')
                    # and         U(s) :=     \max_{a} \sum_{s'} T(s, a, s') U_i(s')
                    (action, action_meu) = self.actionMaximumExpectedUtility(state)

                    # Compute V_{i+1}(s) := R(s) + discount \max_{a} \sum_{s'} T(s, a, s') U_i(s')
                    self.valueUtility[state] = self.environment.reward(state) + self.environment.discountFactor * action_meu

            environment.displayValueFunction(self.valueUtility, iteration=iteration)

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
