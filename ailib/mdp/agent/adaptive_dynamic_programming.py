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
    import agent
else:
    from . import agent

import numpy as np

class Agent(agent.Agent):
    """
    Adaptive Dynamic Programming.
    This is a passive RL algorithm (approximate V function for a given policy).

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 853-855.
    """

    def __init__(self, environment, policy, number_of_simulations = 100):

        self.environment = environment
        self.policy = policy

        initial_state_set = {None}                # perform all simulations from the default initial state (some states may not be explored...)
        #initial_state_set = environment.stateSet  # perform simulations from all states (all states are explored)

        # Init value utility to 0
        self.countStateActionDict = {}
        self.countStateActionNextStateDict = {}
        self.approximateRewardDict = {}
        self.approximateTransitionDict = {(state, action, next_state):0 for state in self.environment.stateSet - self.environment.finalStateSet for action in self.environment.actionSet for next_state in self.environment.stateSet}

        for initial_state in initial_state_set:
            for simulation_index in range(number_of_simulations):
                # Do the simulation
                (state_list, action_list, reward_list) = self.environment.simulate(self, initial_state=initial_state,  max_it=1000)

                #print(len(state_list), len(action_list), len(reward_list))

                # Update approximateRewardDict
                for index in range(len(state_list)):
                    reward = reward_list[index]
                    state = state_list[index]

                    print(index, state, reward)

                    if state not in self.approximateRewardDict:
                        self.approximateRewardDict[state] = reward

                # Update countStateActionDict and countStateActionNextStateDict
                for index in range(1, len(state_list)):
                    state = state_list[index-1]
                    action = action_list[index-1]
                    next_state = state_list[index]
                    
                    print(index, state, action, next_state)

                    # Update countStateActionDict
                    if (state, action) in self.countStateActionDict:
                        self.countStateActionDict[(state, action)] += 1
                    else:
                        self.countStateActionDict[(state, action)] = 1

                    # Update countStateActionNextStateDict
                    if (state, action, next_state) in self.countStateActionNextStateDict:
                        self.countStateActionNextStateDict[(state, action, next_state)] += 1
                    else:
                        self.countStateActionNextStateDict[(state, action, next_state)] = 1

        # Update approximateTransitionDict
        for (state, action, next_state) in self.countStateActionNextStateDict:
            self.approximateTransitionDict[(state, action, next_state)] = float(self.countStateActionNextStateDict[(state, action, next_state)]) / self.countStateActionDict[(state, action)]

        # Display
        print("Approximate reward function:\n", self.approximateRewardDict)
        print("Approximate transition function:")
        for state in self.environment.stateSet - self.environment.finalStateSet:
            for action in self.environment.actionSet:
                for next_state in self.environment.stateSet:
                    if self.approximateTransitionDict[(state, action, next_state)] != 0:
                        print(state, action, next_state, self.approximateTransitionDict[(state, action, next_state)])
        print("...all other transition probabilities are equals to 0")


    def getAction(self, state):
        """
        Returns the action to be performed by the agent for a given state.
        """
        action = self.policy[state]
        return action


if __name__ == '__main__':
    from environment.maze import Environment

    environment = Environment()
    policy = { (0,2):'right', (1,2):'right', (2,2):'right',
               (0,1):'up',                   (2,1):'up',
               (0,0):'up',    (1,0):'left',  (2,0):'left', (3,0):'left' }
    agent = Agent(environment, policy)
