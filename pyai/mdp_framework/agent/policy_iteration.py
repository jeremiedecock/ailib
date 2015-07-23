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

# TODO: improve this ?
if __name__ == '__main__':
    import agent
else:
    from . import agent

import numpy as np
import random

class Agent(agent.Agent):
    """
    Policy Iteration.
    One of the two main DP algorithm to solve MDP.

    Requires discount factor in [0;1[ (will bug if discout_factor==1).

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 697-698.
    """

    def __init__(self, environment, maximum_iteration = 30):
        self.environment = environment
        assert 0 < environment.discountFactor < 1   # TODO: may get singular matrices if discountFactor == 1

        # Randomly initialize the policy
        self.policy = {state: random.sample(self.environment.actionSet, 1)[0] for state in self.environment.stateSet}

        for iteration in range(maximum_iteration):
            print("* Iteration", iteration) # TODO

            # EVALUATE THE POLICY #########################
            # (ie. compute V^{\pi_i}(s) \forall s in S)

            value_of_the_current_policy_dict = evaluatePolicy(self.policy, self.environment)

            self.environment.displayPolicy(self, iteration=iteration)
            self.environment.displayValueFunction(value_of_the_current_policy_dict, iteration=iteration)

            # IMPROVE THE POLICY ##########################

            self.policy = {}
            for state in self.environment.stateSet:
                (action, action_meu) = actionMaximumExpectedUtility(state, value_of_the_current_policy_dict, self.environment)
                self.policy[state] = action


    def getAction(self, state):
        """
        Returns the action to be performed by the agent for a given state.
        """
        action = self.policy[state]
        return action


def actionMaximumExpectedUtility(state, value_utility_dict, environment):
    """
    Compute \pi^*(s) = \arg\max_{a} \sum_{s'} T(s, a, s') U_i(s')
    and         U(s) =     \max_{a} \sum_{s'} T(s, a, s') U_i(s')
    """

    best_action = None
    best_action_utility = float("-inf")

    for action in environment.actionSet:
        expected_action_utility = 0.
        for next_state in environment.stateSet:
            expected_action_utility += environment.transition(state, action)[next_state] * value_utility_dict[next_state]
        if expected_action_utility > best_action_utility:
            best_action = action
            best_action_utility = expected_action_utility

    return (best_action, best_action_utility)


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
    transition_matrix = transition_matrix + np.eye(len(transition_list)) # TODO: CONSIDERE FINAL STATES
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


# TEST ########################################################################

#class Agent():
#
#    def __init__(self, policy):
#        self.policy = policy
#
#    def getAction(self, state):
#        action = self.policy[state]
#        return action
#
#def test():
#    # DEBUG
#
#    from environment.maze import Environment
#
#    initial_state = (0,0)
#    environment = Environment(initial_state = initial_state)
#
#    # Optimal policy
#    policy_opt = { (0,2):'right', (1,2):'right', (2,2):'right',
#               (0,1):'up',                   (2,1):'up',
#               (0,0):'up',    (1,0):'left',  (2,0):'left', (3,0):'left' }
#
#    policy_ill = { (0,2):'right', (1,2):'right', (2,2):'right',
#               (0,1):'up',                   (2,1):'up',
#               (0,0):'up',    (1,0):'left',  (2,0):'left', (3,0):'left' }
#
#    agent_opt = Agent(policy_opt)
#    agent_ill = Agent(policy_ill)
#
#    value_of_the_current_policy_dict = evaluatePolicy(policy_opt, environment)
#
#    environment.displayPolicy(agent_opt, iteration=0)
#    environment.displayValueFunction(value_of_the_current_policy_dict, iteration=0)
#
#
#if __name__ == '__main__':
#    test()
