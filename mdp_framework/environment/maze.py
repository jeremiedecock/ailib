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
    import environment
else:
    from . import environment

class Environment(environment.Environment):
    """
    This environment is taken from the following book:
    S. J. Russell and P. Norvig, "Intelligence artificielle",
    Pearson Education France, 2e édition, pp. 685-686, 2006.

    The optimal policy when R(s)=-0.04 (for non terminal states) and discard_factor=1 is:
    → → → +  
    ↑   ↑ - 
    ↑ ← ← ←
    (see p.688)
    """

    # self.stateSet
    # self.actionSet
    # self.initialState
    # self.finalStateSet

    def __init__(self, initial_state = (0,0)):
        # maze size
        self.numCol = 4
        self.numRow = 3

        # forbidden states
        self.forbiddenStateSet = {(1,1)}

        # set of final states
        self.finalStateSet = {(3,2), (3,1)}

        # set of states
        self.stateSet = {(col, row) for col in range(self.numCol) for row in range(self.numRow)} - self.forbiddenStateSet

        assert len(self.forbiddenStateSet & self.stateSet) == 0  # check whether forbiddenStateSet and stateSet are disjoint sets
        assert self.finalStateSet <= self.stateSet               # is finalStateSet subset of stateSet ?

        # set of actions
        # A = {'←', '→', '↓', '↑'}
        self.actionSet = {'up', 'down', 'left', 'right'}

        # initial state
        assert initial_state in self.stateSet
        self.initialState = initial_state


    def reward(self, current_state, action=None, next_state=None):
        assert current_state in self.stateSet

        #if action is not None:
        #    assert action in self.actionSet
        #if next_state is not None:
        #    assert next_state in self.stateSet
        #
        ## TODO: which version should I considere ? R(s) or R(s,a,s') ?
        #if next_state is not None:
        #    if next_state == (3,2):
        #        reward = 1.
        #    elif next_state == (3,1):
        #        reward = -1.
        #    else:
        #        reward = -0.04
        #else:
        if current_state == (3,2):
            reward = 1.
        elif current_state == (3,1):
            reward = -1.
        else:
            reward = -0.04

        return reward


    def transition(self, current_state, action):
        assert current_state in self.stateSet
        assert action in self.actionSet

        next_state_proba = {state:0. for state in self.stateSet}

        col, row = current_state

        s_up = (col, row+1)
        s_down = (col, row-1)
        s_left = (col-1, row)
        s_right = (col+1, row)
        
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


    # DEBUG FUNCTIONS #########################################################


    def displayStateAction(self, current_state, current_action=None, iteration=None):
        assert current_state in self.stateSet

        text_dict = {}

        if current_action is not None:
            char_dict = {'left':'←', 'right':'→', 'down':'↓', 'up':'↑', ' ':' '}
            text_dict[current_state] = char_dict[current_action]

        min_reward = min([self.reward(state) for state in self.stateSet])
        max_reward = max([self.reward(state) for state in self.stateSet])
        min_color = 0.15
        max_color = 0.85
        scale_color = lambda reward: (float(reward * -1. - min_reward)  / float(max_reward - min_reward)) * (max_color - min_color) + min_color

        color_dict = {state: (0, 0, 1, scale_color(self.reward(state))) for state in self.stateSet}

        inner_square_dict = {current_state: (1, 0, 0, 1)}

        title = "iteration_{0}".format(iteration) if iteration is not None else None
        display_maze_with_cairo(self.numCol, self.numRow, text_dict=text_dict, color_dict=color_dict, inner_square_dict=inner_square_dict, title=title)


    def displayReward(self):
        text_dict = {state:str(self.reward(state)) for state in self.stateSet}

        min_reward = min([self.reward(state) for state in self.stateSet])
        max_reward = max([self.reward(state) for state in self.stateSet])
        min_color = 0.15
        max_color = 0.85
        scale_color = lambda reward: (float(reward * -1. - min_reward)  / float(max_reward - min_reward)) * (max_color - min_color) + min_color

        color_dict = {state: (0, 0, 1, scale_color(self.reward(state))) for state in self.stateSet}

        bold_set = self.finalStateSet | {self.initialState}
        inner_square_dict = {state: (1, 0, 0, 1) for state in self.finalStateSet}
        inner_square_dict[self.initialState] = (0, 0, 1, 1)

        title = "rewards"
        display_maze_with_cairo(self.numCol, self.numRow, text_dict=text_dict, color_dict=color_dict, inner_square_dict=inner_square_dict, bold_set=bold_set, title=title)


    def displayValueFunction(self, value_utility_dict, iteration=None):
        text_dict = {state:"{0:0.3f}".format(value_utility_dict[state]) for state in self.stateSet}

        min_value = min([value_utility_dict[state] for state in self.stateSet])
        max_value = max([value_utility_dict[state] for state in self.stateSet])
        min_color = 0.15
        max_color = 0.85
        scale_color = lambda value: (float(value * -1. - min_value)  / float(max_value - min_value)) * (max_color - min_color) + min_color

        color_dict = {state: (0, 0, 1, scale_color(value_utility_dict[state])) for state in self.stateSet}

        bold_set = self.finalStateSet | {self.initialState}
        inner_square_dict = {state: (1, 0, 0, 1) for state in self.finalStateSet}
        inner_square_dict[self.initialState] = (0, 0, 1, 1)

        title = "value_function"
        if iteration != None:
            title += "_" + str(iteration)
        display_maze_with_cairo(self.numCol, self.numRow, text_dict=text_dict, color_dict=color_dict, inner_square_dict=inner_square_dict, bold_set=bold_set, title=title)


    def displayPolicy(self, agent, iteration=None):
        char_dict = {'left':'←', 'right':'→', 'down':'↓', 'up':'↑', ' ':' '}
        text_dict = {state:char_dict[agent.getAction(state)] for state in self.stateSet - self.finalStateSet}

        min_reward = min([self.reward(state) for state in self.stateSet])
        max_reward = max([self.reward(state) for state in self.stateSet])
        min_color = 0.15
        max_color = 0.85
        scale_color = lambda reward: (float(reward * -1. - min_reward)  / float(max_reward - min_reward)) * (max_color - min_color) + min_color

        color_dict = {state: (0, 0, 1, scale_color(self.reward(state))) for state in self.stateSet}

        bold_set = self.finalStateSet | {self.initialState}
        inner_square_dict = {state: (1, 0, 0, 1) for state in self.finalStateSet}
        inner_square_dict[self.initialState] = (0, 0, 1, 1)

        title = "policy"
        if iteration != None:
            title += "_" + str(iteration)
        display_maze_with_cairo(self.numCol, self.numRow, text_dict=text_dict, color_dict=color_dict, inner_square_dict=inner_square_dict, bold_set=bold_set, title=title)


    def displayTransitionProbabilityDistribution(self, current_state, current_action):
        """
        Display the probability mass function for a given (state, action).
        """
        char_dict = {'left':'←', 'right':'→', 'down':'↓', 'up':'↑', ' ':' '}
        next_state_proba_distribution = self.transition(current_state, current_action)

        text_dict = {next_state:str(next_state_proba_distribution[next_state]) for next_state in self.stateSet}
        text_sub_dict = {current_state: char_dict[current_action]}
        color_dict = {next_state: (0, 0, 1, next_state_proba_distribution[next_state]) for next_state in self.stateSet}
        bold_set = {current_state}
        inner_square_dict = {current_state: (1, 0, 0, 1)}

        title = "p.m.f._" + str(current_state[0]) + "_" + str(current_state[1]) + "_" + current_action
        display_maze_with_cairo(self.numCol, self.numRow, text_dict=text_dict, color_dict=color_dict, inner_square_dict=inner_square_dict, bold_set=bold_set, text_sub_dict=text_sub_dict, title=title)


def display_maze_with_cairo(num_col, num_row, text_dict=None, color_dict=None, bold_set=set(), inner_square_dict=dict(), text_sub_dict=dict(), title=None):
    assert num_col > 0
    assert num_row > 0
    
    SQUARE_SIZE = 96  # pixels
    BORDER_WIDTH = 3  # pixels

    suffixe = "_{0}".format(title) if title is not None else ""
    file_name = "maze" + suffixe + ".svg"

    res_width = num_col * SQUARE_SIZE   # pixels
    res_height = num_row * SQUARE_SIZE + (0 if title is None else 32) # pixels 

    import cairo

    # Image surfaces provide the ability to render to memory buffers either
    # allocated by cairo or by the calling code.
    # List of supported surfaces: http://www.cairographics.org/manual/cairo-surfaces.html
    surface = cairo.SVGSurface(file_name, res_width, res_height)

    # cairo.Context is the object that you send your drawing commands to.
    context = cairo.Context(surface)

    # background
    context.set_source_rgb(1, 1, 1)
    context.rectangle(0, 0, res_width, res_height)
    context.fill()

    for col in range(num_col):
        for row in range(num_row):

            cairo_col = col
            cairo_row = num_row - 1 - row   # cairo use an inverted system of coordinate : (0,0) point is at the top left

            # fill...
            if (col, row) in color_dict:
                context.set_source_rgba(*color_dict[(col, row)])
            else:
                context.set_source_rgba(0, 0, 0, 0.9)

            context.rectangle(SQUARE_SIZE * cairo_col, SQUARE_SIZE * cairo_row, SQUARE_SIZE, SQUARE_SIZE)
            context.fill_preserve()                # preserve path for stroke

            # ...and stroke
            context.set_source_rgb(0, 0, 0)
            context.set_line_width(BORDER_WIDTH)
            context.stroke()

            # inner square
            if (col, row) in inner_square_dict:
                context.set_source_rgba(*inner_square_dict[(col, row)])
                context.rectangle(SQUARE_SIZE * cairo_col + BORDER_WIDTH, SQUARE_SIZE * cairo_row + BORDER_WIDTH, SQUARE_SIZE - 2. * BORDER_WIDTH, SQUARE_SIZE - 2. * BORDER_WIDTH)
                context.stroke()

            # text
            if (col, row) in text_dict:
                context.set_source_rgb(0, 0, 0)

                if (col, row) in bold_set:
                    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                else:
                    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                context.set_font_size(16.0)

                (x, y, text_width, text_height, dx, dy) = context.text_extents(text_dict[(col, row)])

                square_center = (SQUARE_SIZE * cairo_col + (SQUARE_SIZE/2.), SQUARE_SIZE * cairo_row + (SQUARE_SIZE/2.))

                context.move_to(square_center[0] - text_width/2., square_center[1] + text_height/2.)
                context.show_text(text_dict[(col, row)])
                context.move_to(0, 0)

            if (col, row) in text_sub_dict:
                context.set_source_rgb(0, 0, 0)

                if (col, row) in bold_set:
                    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
                else:
                    context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                context.set_font_size(12.0)

                (x, y, text_width, text_height, dx, dy) = context.text_extents(text_sub_dict[(col, row)])

                square_center = (SQUARE_SIZE * cairo_col + (SQUARE_SIZE/2.), SQUARE_SIZE * cairo_row + (SQUARE_SIZE * 3./4.))

                context.move_to(square_center[0] - text_width/2., square_center[1] + text_height/2.)
                context.show_text(text_sub_dict[(col, row)])
                context.move_to(0, 0)

    # title
    if title is not None:
        context.set_source_rgb(0, 0, 0)
        context.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)

        (x, y, text_width, text_height, dx, dy) = context.text_extents(title)

        pos_center = (res_width / 2., res_height - 16)
        context.move_to(pos_center[0] - text_width/2., pos_center[1] + text_height/2.)
        context.show_text(title)


    # write the svg file
    surface.finish()


# TEST ########################################################################


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
    initial_state = (0,0)
    environment = Environment(initial_state = initial_state)
    agent = Agent()

    environment.displayReward()
    environment.displayPolicy(agent)

    # test reward
    print()
    for state in {(col,row) for col in range(environment.numCol) for row in range(environment.numRow)} - environment.forbiddenStateSet:
        print(state, environment.reward(state))

    # test transition
    for state in environment.stateSet - environment.finalStateSet:
        for action in environment.actionSet:
            environment.displayTransitionProbabilityDistribution(state, action)

            # Check the probability mass function
            # TODO: check the equality of 2 floats is ackward...
            probability_distribution = environment.transition(state, action)
            assert sum(probability_distribution.values()) == 1.
    

if __name__ == '__main__':
    test()

