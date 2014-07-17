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

class Agent(agent.Agent):
    """
    Direct Utility Estimation.
    This is a passive RL algorithm (approximate V function for a given policy).

    See: Stuart Russell, Peter Norvig, "Intelligence artificielle", 2e édition,
    Pearson, 2006, pp. 852-853.
    """

    def __init__(self, environment, policy, number_of_simulations = 50):

        self.environment = environment
        self.policy = policy

        initial_state_set = {None}                # perform all simulations from the default initial state (some states may not be explored...)
        #initial_state_set = environment.stateSet  # perform simulations from all states (all states are explored)

        # Init value utility to 0
        self.valueUtility = {state:None for state in self.environment.stateSet}

        state_total_reward_dict = {state:[] for state in self.environment.stateSet}

        for initial_state in initial_state_set:
            for simulation_index in range(number_of_simulations):
                # Do the simulation
                (state_list, action_list, reward_list) = environment.simulate(self, initial_state=initial_state,  max_it=1000)
                print(state_list, "\n", reward_list)

                for index, state in enumerate(state_list):
                    total_reward = sum(reward_list[index:])
                    print(index, state_list[index], reward_list[index], total_reward)
                    state_total_reward_dict[state].append(total_reward)

                self.valueUtility = {state: np.mean(total_reward_list) if len(total_reward_list) > 0 else None for state, total_reward_list in state_total_reward_dict.items()}

                print(self.valueUtility)
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
