"""A second learning agent, so two variants can meet in one match and actually learn from another

Identical code to ``q_agent`` and just uses that one.

We can technically run mutliple Q-Agents, but we cannot have them have different configs.
So we add this one to essentially give it a differently configured enemy.

Q_AGENT_CONFIG=task4 Q_SPARRING_CONFIG=task3 python main.py play --no-gui --agents q_agent q_sparring --train 1
"""

from agent_code.q_agent import api

CONFIG_ENV = 'Q_SPARRING_CONFIG'
DEFAULT_CONFIG = 'tournament'


def setup(self):
    api.setup(self, api.resolve_config(CONFIG_ENV, DEFAULT_CONFIG))


def act(self, game_state: dict) -> str:
    return api.act(self, game_state)
