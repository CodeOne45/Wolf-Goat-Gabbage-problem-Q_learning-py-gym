# Authors: Aman KUMAR & Théo DILLENSEGER


# Description: This file contains the implementation of the problem of the wolf,
# the goat and the cabbage.
# The problem is to move the wolf, the goat and the cabbage from the left bank
# to the right bank of a river.
# The boat can only carry one element at a time.
# Rules:
#       - If the wolf and the goat are alone on one bank, the wolf eats the goat.
#       - If the goat and the cabbage are alone on one bank, the goat eats the cabbage.
#       - If the wolf and the goat are alone on one bank, the wolf eats the goat.


import gym
from gym import spaces
import numpy as np
import random
import copy
from typing import Callable 


class WolfGoatCabbageEnv(gym.Env):

    # The state of the problem is represented by a tuple of 3 sets:
    # - the first set represents the elements on the left bank
    # - the second set represents the elements on the boat
    # - the third set represents the elements on the right bank
    # The elements are represented by the letters '🐺' for the wolf, '🐐' for the
    # goat, '🥦' for the cabbage and '🚣🏽' for the boat.
    def __init__(self, start_state=[['🥦', '🐐', '🚣🏽', '🐺'], [], []],
        goal_state=[[], [], ['🚣🏽', '🐐', '🥦', '🐺']]):

        self.start_state = tuple(set(x) for x in start_state)
        self.goal_state = tuple(set(x) for x in goal_state)

        self.action_space = spaces.Discrete(16)  # 16 possible actions
        self.observation_space = spaces.Tuple([spaces.MultiDiscrete([4, 4, 4])])

        self.state = self.start_state
        self.done = False

    # Moves an element from a set to another
    # @param
    # state: current state of the problem
    # what: element to move
    # where_from: index of the set where the element is
    # where_to: index of the set where the element will be
    def move(self, state, what, where_from, where_to):
        state[where_from].remove(what)
        state[where_to].add(what)

    # Returns the reward of a state
    # @param
    # state: state of the problem
    # @return
    # reward: reward of the state, if the state is the goal state,
    # the reward is 100, else it is 0
    def get_reward(self, state):
        if state == self.goal_state:
            return 100
        return 0

    # is_win_state: returns True if the state is the goal state, else returns False
    # @param
    # state: state of the problem
    def is_win_state(self, state):
        return state == self.goal_state

    # step: performs an action on the current state
    # @param
    # action: action to perform, it is an integer between 0 and 15 because
    # there are 16 possible actions (4 elements to move * 4 possible directions)
    def step(self, action):

        # List of all possible actions
        actions = [
            'MOVE_CABBAGE_AND_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_CABBAGE_AND_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_CABBAGE_AND_PLAYER_FROM_RIGHT_TO_BOAT',
            'MOVE_CABBAGE_AND_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_GOAT_AND_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_GOAT_AND_PLAYER_FROM_RIGHT_TO_BOAT',
            'MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_WOLF_AND_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_WOLF_AND_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_WOLF_AND_PLAYER_FROM_RIGHT_TO_BOAT',
            'MOVE_WOLF_AND_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_BOAT_TO_LEFT',
            'MOVE_PLAYER_FROM_BOAT_TO_RIGHT',
            'MOVE_PLAYER_FROM_LEFT_TO_BOAT',
            'MOVE_PLAYER_FROM_RIGHT_TO_BOAT',
        ]


        chosen_action = actions[action]
        next_state = self.get_next_state(self.state, chosen_action)
        reward = self.get_reward(next_state)
        self.state = next_state

        # If the state is not valid, which means that the wolf eats the goat or
        # the goat eats the cabbage,
        if not self.is_valid_state(self.state):
            self.done = True
            return self.state, reward, self.done, {}

        # If the state is the goal state
        if self.is_win_state(self.state):
            self.done = True
            return self.state, reward, self.done, {}

        return self.state, reward, False, {}


    # is valid state: returns True if the state is valid, else returns False
    # @param
    # state: state of the problem
    # Description: a state is valid if the wolf does not eat the goat and the
    # goat does not eat the cabbage
    def is_valid_state(self, state):
        if state == self.goal_state:
            return True
        if {'🐐', '🐺'} <= state[0] and '🚣🏽' not in state[0]:
            return False
        if {'🐐', '🥦'} <= state[0] and '🚣🏽' not in state[0]:
            return False
        if {'🐐', '🐺'} <= state[2] and '🚣🏽' not in state[2]:
            return False
        if {'🐐', '🥦'} <= state[2] and '🚣🏽' not in state[2]:
            return False
        return True

    def is_win_state(self, state):
        return state == self.goal_state

    # get next state: returns the next state of the problem
    # @param
    # starting_state: state of the problem
    # action: action to perform
    # Description: Moves an element in the tuple dicteonary
    def get_next_state(self, starting_state, action):

        next_state = copy.deepcopy(starting_state)

        if action == 'MOVE_PLAYER_FROM_LEFT_TO_BOAT':
            if '🚣🏽' in next_state[0]:
                self.move(next_state, '🚣🏽' ,0 , 1)

        if action == 'MOVE_PLAYER_FROM_RIGHT_TO_BOAT':
            if '🚣🏽' in next_state[2]:
                self.move(next_state, '🚣🏽', 2, 1)

        if action == 'MOVE_PLAYER_FROM_BOAT_TO_LEFT':
            if len(next_state[1]) == 1 and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 0)

        if action == 'MOVE_PLAYER_FROM_BOAT_TO_RIGHT':
            if len(next_state[1]) == 1 and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 2)
        if action == 'MOVE_GOAT_AND_PLAYER_FROM_LEFT_TO_BOAT':
            if '🐐' in next_state[0] and '🚣🏽' in next_state[0]:
                self.move(next_state, '🚣🏽', 0, 1)
                self.move(next_state, '🐐', 0, 1)

        if action == 'MOVE_WOLF_AND_PLAYER_FROM_LEFT_TO_BOAT':
            if '🐺' in next_state[0] and '🚣🏽' in next_state[0]:
                self.move(next_state, '🚣🏽', 0, 1)
                self.move(next_state, '🐺', 0, 1)

        if action == 'MOVE_CABBAGE_AND_PLAYER_FROM_LEFT_TO_BOAT':
            if '🥦' in next_state[0] and '🚣🏽' in next_state[0]:
                self.move(next_state, '🚣🏽', 0, 1)
                self.move(next_state, '🥦', 0, 1)

        if action == 'MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_LEFT':
            if '🐐' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1 ,0)
                self.move(next_state, '🐐', 1, 0)

        if action == 'MOVE_WOLF_AND_PLAYER_FROM_BOAT_TO_LEFT':
            if '🐺' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 0)
                self.move(next_state, '🐺', 1, 0)

        if action == 'MOVE_CABBAGE_AND_PLAYER_FROM_BOAT_TO_LEFT':
            if '🥦' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 0)
                self.move(next_state, '🥦', 1, 0)
        if action == 'MOVE_GOAT_AND_PLAYER_FROM_BOAT_TO_RIGHT':
            if '🐐' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 2)
                self.move(next_state, '🐐', 1, 2)

        if action == 'MOVE_WOLF_AND_PLAYER_FROM_BOAT_TO_RIGHT':
            if '🐺' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 2)
                self.move(next_state, '🐺', 1, 2)

        if action == 'MOVE_CABBAGE_AND_PLAYER_FROM_BOAT_TO_RIGHT':
            if '🥦' in next_state[1] and '🚣🏽' in next_state[1]:
                self.move(next_state, '🚣🏽', 1, 2)
                self.move(next_state, '🥦', 1, 2)

        if action == 'MOVE_GOAT_AND_PLAYER_FROM_RIGHT_TO_BOAT':
            if '🐐' in next_state[2] and '🚣🏽' in next_state[2]:
                self.move(next_state, '🚣🏽', 2, 1)
                self.move(next_state, '🐐', 2, 1)

        if action == 'MOVE_WOLF_AND_PLAYER_FROM_RIGHT_TO_BOAT':
            if '🐺' in next_state[2] and '🚣🏽' in next_state[2]:
                self.move(next_state, '🚣🏽', 2, 1)
                self.move(next_state, '🐺', 2, 1)

        if action == 'MOVE_CABBAGE_AND_PLAYER_FROM_RIGHT_TO_BOAT':
            if '🥦' in next_state[2] and '🚣🏽' in next_state[2]:
                self.move(next_state, '🚣🏽', 2, 1)
                self.move(next_state, '🥦', 2, 1)

        return next_state

    def reset(self):
        start_state=[['🥦', '🐐', '🚣🏽', '🐺'], [], []]
        self.state = tuple(set(x) for x in start_state)
        self.done = False
        return self.state

    def render(self):
        print("Current State: ", self.state)






