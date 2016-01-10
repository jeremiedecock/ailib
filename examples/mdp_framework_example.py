#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2014,2015 Jérémie DECOCK (http://www.jdhp.org)

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

"""
A PyAI (Markov Decision Processes framework) demo.
"""

from pyai.mdp.environment.maze import Environment
#from pyai.mdp.environment.graph import Environment
#from pyai.mdp.environment.maze import Agent
#from pyai.mdp.agent.brute_force import Agent
#from pyai.mdp.agent.value_iteration import Agent
#from pyai.mdp.agent.value_iteration_gauss_seidel import Agent
#from pyai.mdp.agent.value_iteration_error_rate import Agent
from pyai.mdp.agent.policy_iteration import Agent
#from pyai.mdp.agent.direct_utility_estimation import Agent

def main():
    """Main function"""

    #initial_state = (0,0)
    #environment = Environment(initial_state = initial_state)

    #environment = Environment()
    environment = Environment(discount_factor=0.999)
    agent = Agent(environment)

    environment.displayReward()
    environment.displayPolicy(agent)

    # Do the simulation
    (state_list, action_list, reward_list) = environment.simulate(agent)

    # Display states
    print("States:", state_list)
    print("Actions:", action_list)
    print("Rewards:", reward_list)

    for iteration, (state, action) in enumerate(zip(state_list, action_list)):
        environment.displayStateAction(current_state=state, current_action=action, iteration=iteration)
        print("{0}: {1} {2} {3}".format(iteration, state, action, state_list[iteration+1]))

    environment.displayStateAction(current_state=state_list[-1], iteration=len(state_list)-1)

    print("Global reward =", sum(reward_list))

if __name__ == '__main__':
    main()
