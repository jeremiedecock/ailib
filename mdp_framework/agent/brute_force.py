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
import itertools
import numpy as np

class Agent(agent.Agent):

    def __init__(self, environment):

        assert 0 < environment.discountFactor < 1   # TODO: may get singular matrices if discountFactor == 1

        self.policy = {}
        initial_state = environment.initialState

        print("Computing the policy... please wait.")

        S = environment.stateSet
        A = environment.actionSet

        number_of_possible_policies = pow(len(A), len(S))
        print("Number of possible policies :", number_of_possible_policies)

        best_policy = None
        best_policy_values_dict = {state: float("-inf") for state in S}

        
        # The set of all possible (stationary) policies for an environment with
        # len(S) states and len(A) actions (where A is the set of all possible
        # actions and S the set of all possible states) is:
        #
        #    {itertools.product(*[A] * len(S))}
        #
        # There are len(A)^len(S) possible policies (|A|^|S|) where
        # len(A)^len(S) is the cartesian product A x A x A x ...
        #
        # itertools.product() can be used to enumerate all possible policies
        # an other solution is to use len(S) nested loops but this is not
        # really convenient...
        # 
        # itertools.product(*[A] * 2) := itertools.product(A, A)       := A x A cartesian product         := [(a1, a2) for a1 in A for a2 in A] for the set of all possible policies for an environment with 2 states and len(A) actions (A is the set of all actions)
        # itertools.product(*[A] * 3) := itertools.product(A, A, A)    := A x A x A cartesian product     := [(a1, a2, a3) for a1 in A for a2 in A for a3 in A] for the set of all possible policies for an environment with 3 states and len(A) actions (A is the set of all actions)
        # itertools.product(*[A] * 4) := itertools.product(A, A, A, A) := A x A x A x A cartesian product := [(a1, a2, a3, a4) for a1 in A for a2 in A for a3 in A for a4 in A] for the set of all possible policies for an environment with 4 states and len(A) actions (A is the set of all actions)
        # ...
        #
        # Example:
        #   A = {'←', '→', '↓', '↑'}
        #   S = {1, 2}
        #   P = itertools.product(*[A] * len(S))
        #   [p for p in P]
        #   >>> [('↓', '↓'),
        #        ('↓', '→'),
        #        ('↓', '↑'),
        #        ('↓', '←'),
        #        ('→', '↓'),
        #        ('→', '→'),
        #        ('→', '↑'),
        #        ('→', '←'),
        #        ('↑', '↓'),
        #        ('↑', '→'),
        #        ('↑', '↑'),
        #        ('↑', '←'),
        #        ('←', '↓'),
        #        ('←', '→'),
        #        ('←', '↑'),
        #        ('←', '←')]
        policy_it = 1
        for pi in itertools.product(*[A] * len(S)):
            # Print progress
            if policy_it%1000 == 0:
                print("{0}% ({1}/{2})".format(float(policy_it)/number_of_possible_policies * 100., policy_it, number_of_possible_policies))

            self.policy = {p[0]:p[1] for p in zip(S, pi)}

            # Evaluate the policy (stochastic environment)
            policy_values_dict = evaluatePolicy(self.policy, environment)

            # Update best_policy
            if policy_values_dict[initial_state] > best_policy_values_dict[initial_state]:
                best_policy = self.policy
                best_policy_values_dict = policy_values_dict
                environment.displayPolicy(self, iteration=policy_it)
                environment.displayValueFunction(policy_values_dict, iteration=policy_it)

            policy_it += 1

        self.policy = best_policy

        environment.displayPolicy(self)

        print("Done.")


    def getAction(self, state):
        action = self.policy[state]
        return action

def evaluatePolicy(policy, environment):
    """
    Evaluate the policy (ie. compute V^{\pi_i}(s) \forall s in S)
    """
    state_list = list(environment.stateSet)
    #print("STATES", state_list)

    # Make the transition matrix
    transition_list = []
    for state_from in state_list:
        if state_from in environment.finalStateSet:
            transition_list.append([0 for state_to in state_list])
        else:
            action = policy[state_from]
            transition_list.append([-environment.discountFactor * environment.transition(state_from, action)[state_to] for state_to in state_list])
    transition_matrix = np.array(transition_list)
    transition_matrix = transition_matrix + np.eye(len(transition_list))
    #print("TRANSITION\n", transition_matrix)

    # Make the reward vector
    reward_vector = np.array([environment.reward(state) for state in state_list])
    #print("REWARD", reward_vector)

    # Solve the system of simplified Bellman equations
    #value_vector = np.dot(np.linalg.inv(transition_matrix), reward_vector)
    value_vector = np.linalg.solve(transition_matrix, reward_vector)
    #print("VALUE", value_vector)

    value_of_the_current_policy_dict = {state: value_vector[state_index] for state_index, state in enumerate(state_list)}
    #print(value_of_the_current_policy_dict)

    return value_of_the_current_policy_dict


if __name__ == '__main__':
    pass
