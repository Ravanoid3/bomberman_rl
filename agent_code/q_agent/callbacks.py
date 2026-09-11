"""Tournament entry the configuration is pinned, not read from the environment to make sure we aren't
mis-configured by a stray environment variable when playing in the tournament

It always plays the ``tournament`` configuration (best checkpoint, safety veto on, no
exploration).
"""
from agent_code.q_agent import api

CONFIG = 'tournament'


def setup(self):
    api.setup(self, CONFIG)


def act(self, game_state: dict) -> str:
    return api.act(self, game_state)
