import os
import pickle
import random
from collections import defaultdict, deque

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

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
        # self.model = defaultdict(lambda: weights / weights.sum())
        self.model = defaultdict(lambda: np.zeros(len(ACTIONS)))
    else:
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

    self.logger.info("Setting up the agent gero.")
    self.epsilon = EPSILON


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

    q_values = self.model.get(state)
    if q_values is None:
        q_values = np.zeros(len(ACTIONS))

    max_q = np.max(q_values)
    best_actions = np.flatnonzero(q_values == max_q)
    action_index = np.random.choice(best_actions)
    action = ACTIONS[action_index]

    return action


def state_to_features(game_state: dict) -> tuple:
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bomb_possible, (x, y) = game_state["self"]

    width, height = field.shape
    coins = game_state["coins"]

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

    if coins:
        coin = min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        cx, cy = coin
        tx = cx - x
        ty = cy - y

        coin_visible = True
        coin_dx = int(np.sign(tx))
        coin_dy = int(np.sign(ty))
        coin_distance_x = bucket_distance(tx)
        coin_distance_y = bucket_distance(ty)

    else:
        coin_visible = False
        coin_dx = 0
        coin_dy = 0
        coin_distance_x = 3
        coin_distance_y = 3

    # check if a field is dangerous i.e. in explosion distance to a detonating bomb
    def is_dangerous(i, j):
        explosions = game_state["explosion_map"]
        if explosions[i, j] > 0:
            return True

        for (bx, by), countdown in game_state["bombs"]:
            if (i, j) == (bx, by):
                return True

        return False

    def get_tile_type(i, j):
        if i < 0 or i >= width or j < 0 or j >= height:
            return 2
        if field[i, j] == -1:
            return 2
        if field[i, j] == 1:
            return 1
        return 0

    def get_bomb_danger(i, j):
        explosions = game_state["explosion_map"]

        danger_here = 0
        danger_up = 0
        danger_right = 0
        danger_down = 0
        danger_left = 0

        # Already exploding
        if explosions[i, j] > 0:
            danger_here = 1

        if j - 1 >= 0 and explosions[i, j - 1] > 0:
            danger_up = 1

        if i + 1 < width and explosions[i + 1, j] > 0:
            danger_right = 1

        if j + 1 < height and explosions[i, j + 1] > 0:
            danger_down = 1

        if i - 1 >= 0 and explosions[i - 1, j] > 0:
            danger_left = 1

        # Future explosions
        for (bx, by), countdown in game_state["bombs"]:

            danger_value = min(countdown, 3)

            if (bx, by) == (i, j):
                danger_here = max(danger_here, danger_value)

            directions = [
                (0, -1),  # UP
                (1, 0),  # RIGHT
                (0, 1),  # DOWN
                (-1, 0)  # LEFT
            ]

            for dx, dy in directions:

                for distance in range(1, 4):

                    fx = bx + dx * distance
                    fy = by + dy * distance

                    if fx < 0 or fx >= width or fy < 0 or fy >= height:
                        break

                    if field[fx, fy] == -1:
                        break

                    if (fx, fy) == (i, j):
                        danger_here = max(danger_here, danger_value)

                    elif (fx, fy) == (i, j - 1):
                        danger_up = max(danger_up, danger_value)

                    elif (fx, fy) == (i + 1, j):
                        danger_right = max(danger_right, danger_value)

                    elif (fx, fy) == (i, j + 1):
                        danger_down = max(danger_down, danger_value)

                    elif (fx, fy) == (i - 1, j):
                        danger_left = max(danger_left, danger_value)

                    if field[fx, fy] == 1:
                        break

        return (
            danger_up,
            danger_right,
            danger_down,
            danger_left,
            danger_here
        )


    d_up, d_right, d_down, d_left, d_here = get_bomb_danger(x, y)
    tile_up = get_tile_type(x, y - 1)
    tile_right = get_tile_type(x + 1, y)
    tile_down = get_tile_type(x, y + 1)
    tile_left = get_tile_type(x - 1, y)

    bomb_active = len(game_state["bombs"]) > 0

    def can_escape_bomb(i, j):
        countdown = 3

        # Calculate the tiles affected by a bomb at (x, y)
        blast = {(i, j)}

        directions = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0)
        ]

        for dx, dy in directions:
            for distance in range(1, 4):
                fx = i + dx * distance
                fy = j + dy * distance

                if fx < 0 or fx >= width or fy < 0 or fy >= height:
                    break

                if field[fx, fy] == -1:
                    break

                blast.add((fx, fy))

                if field[fx, fy] == 1:
                    break

        # BFS
        queue = deque()
        queue.append(((i, j), 0))

        visited = {(i, j)}

        while queue:
            (cx, cy), time = queue.popleft()
            if (cx, cy) not in blast and time <= countdown:
                return True

            if time >= countdown:
                continue

            for dx, dy in directions:
                nx = cx + dx
                ny = cy + dy

                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue

                if field[nx, ny] != 0:
                    continue

                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), time + 1))

        return False

    bomb_trap = not can_escape_bomb(x, y)

    return (
        int(coin_visible),
        coin_dx,
        coin_dy,
        coin_distance_x,
        coin_distance_y,

        tile_up,
        tile_right,
        tile_down,
        tile_left,

        d_here,
        d_up,
        d_right,
        d_down,
        d_left,

        int(bomb_possible),
        int(bomb_active),
        int(bomb_trap)
    )
