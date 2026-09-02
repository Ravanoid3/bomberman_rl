"""The learning agent.
Which variant it plays is chosen per match and can be configured using the Q_AGENT_CONFIG environment variable.

The variable must match an entry in the config.py file.

E.g. Q_AGENT_CONFIG=task2 python main.py play --no-gui --agents q_agent --train 1

The framework identifies agents by folder name and runs them all in a single
process, so a folder can only carry one configuration *per match* -- but which
one it carries is chosen from outside.  That is why there are three folders here
rather than one per variant: ``q_agent`` for whatever is under test,
``q_sparring`` so a *second*, differently configured variant can play in the same
match (self-play), and ``q_tournament`` with the configuration pinned.
"""

from agent_code.q_agent import api

CONFIG_ENV = 'Q_AGENT_CONFIG'
DEFAULT_CONFIG = 'tournament'


def setup(self):
    api.setup(self, api.resolve_config(CONFIG_ENV, DEFAULT_CONFIG))


def act(self, game_state: dict) -> str:
    return api.act(self, game_state)
