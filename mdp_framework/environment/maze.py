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

import numpy as np
from scipy.stats import rv_discrete
import random

class Environment:

    # self.stateSet
    # self.actionSet
    # self.initialState
    # self.finalStateSet

    def __init__(self, initial_state = (0,0)):
        # set of states
        self.stateSet = {(x,y) for x in range(4) for y in range(3)} - {(1,1)}

        # set of actions
        self.actionSet = {'up', 'down', 'left', 'right'}

        # initial state
        assert initial_state in self.stateSet
        self.initialState = initial_state

        # set of final states
        self.finalStateSet = {(3,2), (3,1)}


    def reward(self, current_state, action, next_state):
        assert current_state in self.stateSet
        assert action in self.actionSet

        if next_state == (3,2):
            reward = 1
        elif next_state == (3,1):
            reward = -1
        else:
            reward = -0.04

        return reward


    def transition(self, current_state, action):
        assert current_state in self.stateSet
        assert action in self.actionSet

        next_state_proba = {state:0. for state in self.stateSet}

        x, y = current_state

        s_up = (x, y+1)
        s_down = (x, y-1)
        s_left = (x-1, y)
        s_right = (x+1, y)
        
        # TODO: make it shorter
        if action == 'up': 

            # up
            if s_up in self.stateSet:
                next_state_proba[s_up] = 0.8
            else:
                next_state_proba[current_state] += 0.8   # hit the wall, stay in current state

            # left
            if s_left in self.stateSet:
                next_state_proba[s_left] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

            # right
            if s_right in self.stateSet:
                next_state_proba[s_right] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

        elif action == 'down': 

            # down
            if s_down in self.stateSet:
                next_state_proba[s_down] = 0.8
            else:
                next_state_proba[current_state] += 0.8   # hit the wall, stay in current state

            # left
            if s_left in self.stateSet:
                next_state_proba[s_left] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

            # right
            if s_right in self.stateSet:
                next_state_proba[s_right] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

        elif action == 'left': 

            # left
            if s_left in self.stateSet:
                next_state_proba[s_left] = 0.8
            else:
                next_state_proba[current_state] += 0.8   # hit the wall, stay in current state

            # up
            if s_up in self.stateSet:
                next_state_proba[s_up] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

            # down
            if s_down in self.stateSet:
                next_state_proba[s_down] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

        elif action == 'right': 

            # right
            if s_right in self.stateSet:
                next_state_proba[s_right] = 0.8
            else:
                next_state_proba[current_state] += 0.8   # hit the wall, stay in current state

            # up
            if s_up in self.stateSet:
                next_state_proba[s_up] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

            # down
            if s_down in self.stateSet:
                next_state_proba[s_down] = 0.1
            else:
                next_state_proba[current_state] += 0.1   # hit the wall, stay in current state

        else: 
            raise Exception("Internal error")

        return next_state_proba


    # doTransition is common to all Environment classes
    def doTransition(self, current_state, action):
        assert current_state in self.stateSet
        assert action in self.actionSet

        next_state_proba = self.transition(current_state, action)


        # randomly generate the next state acording to the next_state_proba distribution
        state_proba_list = [item for item in next_state_proba.items()]
        state_list = [item[0] for item in state_proba_list]
        proba_list = [item[1] for item in state_proba_list]

        # A Scipy probability distribution
        distrib = rv_discrete(values=(range(len(state_list)), proba_list))

        # One sample
        next_state_index = distrib.rvs()

        next_state = state_list[next_state_index]

        reward = self.reward(current_state, action, next_state)

        return (next_state, reward)


    def display(self, state):
        assert state in self.stateSet

        # ASCII version (TODO: cairo version)
        display_list = ['x' if (x,y) == state else '.' for y in range(3) for x in range(4)]
        display_list = [display_list[i:i+4] for i in range(0, len(display_list), 4)]
        display_list[1][1] = '#'
        display_list[2][3] = '+'
        display_list[1][3] = '-'

        for y in reversed(range(3)):
            for x in range(4):
                print(display_list[y][x], end="")
            print()


    ### DEBUG FUNCTIONS ###


    def display_probas(self, next_state_proba):
        # ASCII version (TODO: cairo version)
        display_list = [next_state_proba[(x,y)] if (x,y) != (1,1) else 0. for y in range(3) for x in range(4)]
        display_list = [display_list[i:i+4] for i in range(0, len(display_list), 4)]

        for y in reversed(range(3)):
            for x in range(4):
                print(display_list[y][x], end=" ")
            print()


### TEST ###


class Agent():

    def __init__(self):
        self.policy = { (0,2):'right', (1,2):'right', (2,2):'right',
                        (0,1):'up',                   (2,1):'up',
                        (0,0):'up',    (1,0):'left',  (2,0):'left', (3,0):'left' }


    def getAction(self, state):
        action = self.policy[state]
        return action


def test():

    # test display
    state = (0,0)
    environment = Environment(initial_state = state)
    environment.display(state)

    # test reward
    print()
    for state in {(x,y) for x in range(4) for y in range(3)} - {(1,1)}:
        print(state, environment.reward((0,0), 'up', state))

    # test transition
    for state in {(0,0), (1,0), (0,1)}:
        print()
        environment.display(state)
        for action in environment.actionSet:
            print()
            print(state, action)
            next_state_proba = environment.transition(state, action)
            environment.display_probas(next_state_proba)
    

if __name__ == '__main__':
    test()

