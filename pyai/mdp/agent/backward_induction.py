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

class Agent(agent.Agent):
    """
    Backward Induction.
    One of the main DP algorithm to solve MDP.
    This algorithm compute the optimal policy for a given time horizon.

    See: Perny's courses
    See: Dynamic Programming literature
    """

    def __init__(self, environment, time_horizon = 10):

        self.environment = environment

        self.valueUtilityList = [{} for time_step in range(time_horizon)]

        # Init value utility for the last time step (ie. time_step = time_horizon)
        value_utility = {state: self.environment.reward(state) for state in self.environment.stateSet}
        self.valueUtilityList[-1] = value_utility

        environment.displayValueFunction(value_utility, iteration=time_horizon-1)

        # Compute value_utility backward in time
        # rem: reversed(range(10)) -> [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        for time_step in reversed(range(time_horizon-1)):

            value_utility = {}      # init V_t

            # For all s
            for state in self.environment.stateSet:
                if state in environment.finalStateSet:
                    # If s is a final state then V_i(s)=R(s) for all i      (TODO: this is not explicitely written in most references...)
                    value_utility[state] = self.environment.reward(state)
                else:
                    # Compute \pi^*(s) := \arg\max_{a} \sum_{s'} T(s, a, s') V_{t+1}(s')
                    # and       V_t(s) :=     \max_{a} \sum_{s'} T(s, a, s') V_{t+1}(s')
                    (action, action_meu) = self.actionMaximumExpectedUtility(state, time_step)

                    # Compute V_t(s) := R(s) + discount \max_{a} \sum_{s'} T(s, a, s') V_{t+1}(s')
                    value_utility[state] = self.environment.reward(state) + action_meu

            self.valueUtilityList[time_step] = value_utility

            environment.displayValueFunction(value_utility, iteration=time_step)


    def actionMaximumExpectedUtility(self, state, time_step):
        """
        Compute \pi^*(s) = \arg\max_{a} \sum_{s'} T(s, a, s') V_{t+1}(s')
        and         V(s) =     \max_{a} \sum_{s'} T(s, a, s') V_{t+1}(s')
        """

        best_action = None
        best_action_utility = float("-inf")

        for action in self.environment.actionSet:
            expected_action_utility = 0.
            for next_state in self.environment.stateSet:
                expected_action_utility += self.environment.transition(state, action)[next_state] * self.valueUtilityList[time_step+1][next_state]
            if expected_action_utility > best_action_utility:
                best_action = action
                best_action_utility = expected_action_utility

        return (best_action, best_action_utility)


    def getAction(self, state, time_step):
        """
        Returns the action to be performed by the agent for a given state.
        """
        (action, action_meu) = self.actionMaximumExpectedUtility(state, time_step)
        return action


if __name__ == '__main__':
    from environment.maze import Environment

    environment = Environment()
    agent = Agent(environment)

    environment.displayReward()

    initial_state = None
    max_it = 10

    # Do the simulation
    #(state_list, action_list, reward_list) = environment.simulate(agent)
    if initial_state is None:
        initial_state = environment.initialState
    else:
        assert initial_state in environment.stateSet

    state_list = [initial_state]
    action_list = []
    reward_list = []

    state = initial_state

    while(state not in environment.finalStateSet and len(action_list) < max_it):
        action = agent.getAction(state, len(action_list))
        (state, reward) = environment.doTransition(state, action)

        state_list.append(state)
        action_list.append(action)
        #reward_list.append(reward)  # TODO

    reward_list = [environment.reward(state) for state in state_list]  # TODO

    # Display states
    print("States:", state_list)
    print("Actions:", action_list)
    print("Rewards:", reward_list)

    for iteration, (state, action) in enumerate(zip(state_list, action_list)):
        environment.displayStateAction(current_state=state, current_action=action, iteration=iteration)
        print("{0}: {1} {2} {3}".format(iteration, state, action, state_list[iteration+1]))

    environment.displayStateAction(current_state=state_list[-1], iteration=len(state_list)-1)

    print("Global reward =", sum(reward_list))
