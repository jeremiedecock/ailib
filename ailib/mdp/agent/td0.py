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
    Temporal Difference Learning "TD(0)".
    This is a passive RL algorithm (approximate V function for a given policy).

    Parameters:
    - self.learningRate : the learning rate (often noted alpha)
    - self.discountFactor : the discount factor (often noted gamma)

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 854-857.
    """

    def __init__(self, environment, policy, number_of_simulations = 10000):

        self.environment = environment
        self.policy = policy

        self.learningRate = lambda n : 1.0/n
        self.discountFactor = 0.999

        initial_state_set = {None}                # perform all simulations from the default initial state (some states may not be explored...)
        #initial_state_set = environment.stateSet  # perform simulations from all states (all states are explored)

        # Init value utility to 0
        self.valueUtility = {state:None for state in self.environment.stateSet}

        self.stateVisitDict = {state:0 for state in self.environment.stateSet}

        for initial_state in initial_state_set:
            for simulation_index in range(number_of_simulations):
                # Do the simulation
                (state_list, action_list, reward_list) = environment.simulate(self, initial_state=initial_state,  max_it=100)

                previous_state = None
                previous_reward = None

                for index in range(len(state_list)):
                    current_state = state_list[index]
                    current_reward = reward_list[index]

                    if self.valueUtility[current_state] is None:
                        self.valueUtility[current_state] = current_reward

                    if previous_state is not None:
                        #print(index, previous_state, previous_reward, current_state, current_reward, self.valueUtility[previous_state], self.valueUtility[current_state])
                        self.stateVisitDict[previous_state] += 1      # What about the last visited state of the simulation ? -> no problem as we won't call alpha(current_state) but only alpha(previous_state)

                        alpha = self.learningRate(self.stateVisitDict[previous_state])
                        self.valueUtility[previous_state] = self.valueUtility[previous_state] + alpha * (previous_reward + self.discountFactor * self.valueUtility[current_state] - self.valueUtility[previous_state])

                    previous_state = current_state
                    previous_reward = current_reward

                # Display
                #print(self.valueUtility)
                #environment.displayValueFunction(self.valueUtility, iteration=simulation_index)
        environment.displayValueFunction(self.valueUtility, iteration=simulation_index)


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
