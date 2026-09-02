"""A random agent that only ever takes provably survivable actions.

It validates the danger model and is an informative baseline.
It doubles for testing the agent actually knows how to stay alive really.
It has no strategy beyond that really, but only picks
uniformly among the actions that ``gamestate.survivable_moves`` certifies
as survivable and drops bombs whenever ``with_own_bomb`` says an escape route
would remain.

If the timing model in :mod:`agent_code.q_agent.gamestate` were
off by a single step, this agent would blow itself up.  ``experiments/selftest``
runs it for 40 crate-heavy rounds and asserts zero suicides.
"""

import logging
import os

import numpy as np

from agent_code.q_agent import gamestate as gs

# How often we consider bombing when a safe bomb is available.
BOMB_PROBABILITY = 0.3


def setup(self):
    self.rng = np.random.default_rng()
    if os.environ.get('BOMBERMAN_VERBOSE', '') in ('', '0', 'false'):
        # see the note in agent_code/q_agent/api.py: per-step logging inside the
        # agent folder is both slow and a sync-client lock magnet
        self.logger.setLevel(logging.WARNING)
    self.logger.warning('safe_random_agent ready')


def act(self, game_state: dict) -> str:
    p = gs.parse(game_state)
    field, pos, lethal, blocked = p['field'], p['pos'], p['lethal'], p['blocked']

    if p['bombs_left'] and self.rng.random() < BOMB_PROBABILITY:
        lethal_b = gs.with_own_bomb(lethal, field, pos)
        blocked_b = blocked.copy()
        blocked_b[:, pos[0], pos[1]] = True  # our own bomb blocks re-entry
        if gs.survivable_moves(field, lethal_b, pos, blocked_b) != 0:
            return 'BOMB'

    mask = gs.survivable_moves(field, lethal, pos, blocked)
    safe = [a for a in range(5) if (mask >> a) & 1]
    if not safe:
        # Provably doomed; nothing to choose. Wait so the log stays readable.
        self.logger.debug('no survivable action available')
        return 'WAIT'

    return gs.ACTIONS[int(self.rng.choice(safe))]
