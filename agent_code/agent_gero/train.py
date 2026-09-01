from collections import namedtuple, deque

import pickle
from typing import List

import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS, ALPHA, GAMMA, MIN_EPSILON

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100_000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def update_q_value(self, transition):
    state = transition.state
    action = transition.action
    next_state = transition.next_state
    reward = transition.reward

    action_idx = ACTIONS.index(action)
    current_q = self.model[state][action_idx]

    if next_state is None:
        target = reward
    else:
        next_max_q = np.max(self.model[next_state])
        target = reward + GAMMA * next_max_q

    self.model[state][action_idx] += ALPHA * (target - current_q)



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.round_reward = 0
    self.round_coins = 0
    self.round_steps = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)

    reward = reward_from_events(self, events)
    transition = Transition(old_state, self_action, new_state, reward)

    self.round_reward += reward
    self.round_steps += 1
    if e.COIN_COLLECTED in events:
        self.round_coins += 1

    # state_to_features is defined in callbacks.py
    self.transitions.append(transition)
    update_q_value(self, transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    transition = Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events))
    self.transitions.append(transition)
    update_q_value(self, transition)

    self.epsilon = max(self.epsilon * 0.995, MIN_EPSILON)

    reward = reward_from_events(self, events)
    self.round_reward += reward
    if e.COIN_COLLECTED in events:
        self.round_coins += 1


    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(dict(self.model), file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 2,
        e.KILLED_OPPONENT: 5,
        e.WAITED: -1,
        e.INVALID_ACTION: -1,
        e.MOVED_UP: -0.05,
        e.MOVED_DOWN: -0.05,
        e.MOVED_LEFT: -0.05,
        e.MOVED_RIGHT: -0.05,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
