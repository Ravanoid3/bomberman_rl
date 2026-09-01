import os
import pickle
import random
from collections import defaultdict

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

ALPHA = 0.1
GAMMA = 0.9

EPSILON = 1.0
MIN_EPSILON = 0.05

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        print("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        #self.model = weights / weights.sum()
        self.model = defaultdict(lambda: np.zeros(len(ACTIONS)))
    else:
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.logger.info("Setting up the agent gero.")
    #self.model = defaultdict(lambda: np.zeros(len(ACTIONS)))
    self.epsilon = EPSILON
    print("Number of Q-states:", len(self.model))

    nonzero_states = 0
    max_abs_q = 0.0

    for state, q_values in self.model.items():
        if np.any(q_values != 0):
            nonzero_states += 1
            max_abs_q = max(max_abs_q, np.max(np.abs(q_values)))

    print("States with non-zero Q-values:", nonzero_states)
    print("Maximum absolute Q-value:", max_abs_q)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    state = state_to_features(game_state)
    if self.train and random.random() < self.epsilon:
        return random.choice(ACTIONS)

    q_values = self.model[state]

    max_q = np.max(q_values)

    best_actions = np.flatnonzero(q_values == max_q)

    action_index = np.random.choice(best_actions)

    action = ACTIONS[action_index]

    return action


def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]

    width, height = field.shape

    def blocked(nx, ny):
        if nx < 0 or nx >= width:
            return 1
        if ny < 0 or ny >= height:
            return 1

        return int(field[nx, ny] != 0)

    blocked_up = blocked(x, y - 1)
    blocked_right = blocked(x + 1, y)
    blocked_down = blocked(x, y + 1)
    blocked_left = blocked(x - 1, y)

    coins = game_state["coins"]

    if not coins:
        return (
            0, 0, 0,
            blocked_up,
            blocked_right,
            blocked_down,
            blocked_left
        )

    # Closest coin using Manhattan distance
    coin = min(
        coins,
        key=lambda c: abs(c[0] - x) + abs(c[1] - y)
    )

    cx, cy = coin

    dx = cx - x
    dy = cy - y

    # Direction
    dx_direction = int(np.sign(dx))
    dy_direction = int(np.sign(dy))


    # Distance bucket
    def bucket_distance(d):
        d = abs(d)

        if d == 0:
            return 0
        elif d == 1:
            return 1
        elif d <= 3:
            return 2
        else:
            return 3

    dx_distance = bucket_distance(dx)
    dy_distance = bucket_distance(dy)

    return (
        np.sign(dx),
        np.sign(dy),
        bucket_distance(abs(dx)),
        bucket_distance(abs(dy)),
        blocked_up,
        blocked_right,
        blocked_down,
        blocked_left
    )

