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


from environment.maze import Environment
#from environment.maze import Agent
#from agent.brute_force import Agent
from agent.value_iteration import Agent

def main():
    """Main function"""

    initial_state = (0,0)

    environment = Environment(initial_state = initial_state)
    agent = Agent(environment)

    # Do the simulation
    (state_list, action_list, reward_list) = environment.simulate(agent)

    environment.display(initial_state)
    for (state, action) in zip(state_list[1:], action_list):
        print()
        print(action)
        print()
        environment.display(state)

    print("Global reward =", sum(reward_list))

    # display the policy
    print("Policy:")
    environment.displayPolicy(agent.policy)


if __name__ == '__main__':
    main()
