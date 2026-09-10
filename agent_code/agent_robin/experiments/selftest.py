"""
Basically sanity checks.

Run with ``python -m experiments.selftest`` from the repository root.

Checks a bunch of assumptions made later on to ensure its actually correct.
It's not actually testing if the agent learns.

1. the reward table keys really are the framework's event names;
2. To train the agent we calculate the blast geometry which must match items.Bomb.get_blast_coords
3. the danger timing is right, i.e. a bomb with ``timer = t`` kills after
   moves ``t`` and ``t + 1`` and no others;
4. the escape search is sound, which we validate based on  ``safe_random_agent``, which does nothing but obey it and must therefore
   never blow itself up;
5. the feature encoder is D4-equivariant, which is what licenses the 8x data augmentation used in training;
6. n-step returns are the discounted sums they claim to be.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from agent_code.agent_robin.experiments import runner

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import events as game_events                          # noqa: E402
import settings as s                                  # noqa: E402
from items import Bomb                                # noqa: E402
from agent_code.agent_robin import config as cfg_module   # noqa: E402
from agent_code.agent_robin import features as feat       # noqa: E402
from agent_code.agent_robin import gamestate as gs        # noqa: E402
from agent_code.agent_robin import replay as replay_mod   # noqa: E402
from agent_code.agent_robin import rewards as rewards_mod # noqa: E402
from agent_code.agent_robin import symmetry               # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    status = 'ok  ' if condition else 'FAIL'
    print(f'  [{status}] {name}' + (f'  -- {detail}' if detail and not condition else ''))
    if not condition:
        FAILURES.append(name)


# --------------------------------------------------------------------------
def empty_arena(crate_density=0.0, seed=0):
    """A board built exactly like ``BombeRLeWorld.build_arena``."""
    rng = np.random.default_rng(seed)
    arena = np.zeros((s.COLS, s.ROWS), int)
    if crate_density:
        arena[rng.random((s.COLS, s.ROWS)) < crate_density] = gs.CRATE
    arena[:1, :] = arena[-1:, :] = arena[:, :1] = arena[:, -1:] = gs.WALL
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = gs.WALL
    for (x, y) in [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]:
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if arena[xx, yy] == gs.CRATE:
                arena[xx, yy] = gs.FREE
    return arena


def make_state(field, pos, coins=(), bombs=(), others=(), explosion_map=None,
               bombs_left=True, step=1, rnd=1):
    return {
        'round': rnd, 'step': step, 'field': field,
        'self': ('me', 0, bombs_left, tuple(pos)),
        'others': [(f'o{i}', 0, True, tuple(p)) for i, p in enumerate(others)],
        'bombs': [(tuple(p), t) for p, t in bombs],
        'coins': [tuple(c) for c in coins],
        'explosion_map': (np.zeros(field.shape) if explosion_map is None
                          else explosion_map),
        'user_input': None,
    }


# --------------------------------------------------------------------------
def test_event_names():
    print('1. reward table keys are real event names')
    known = {v for k, v in vars(game_events).items() if k.isupper()}
    known |= set(rewards_mod.AUX_EVENTS)
    for cfg_name in cfg_module.CONFIGS:
        cfg = cfg_module.get(cfg_name)
        unknown = sorted(set(cfg['rewards']) - known)
        check(f'{cfg_name}: no unknown reward keys', not unknown, str(unknown))
    for name, value in vars(game_events).items():
        if name.isupper():
            check(f'events.{name} == its own name', name == value)
            break  # one representative is enough for the printout
    mismatched = [n for n, v in vars(game_events).items() if n.isupper() and n != v]
    check('every events.py constant equals its name', not mismatched, str(mismatched))


def test_blast_geometry():
    print('2. blast geometry matches items.Bomb')
    field = empty_arena(crate_density=0.5, seed=3)
    rng = np.random.default_rng(1)
    free = np.argwhere(field != gs.WALL)
    ok = True
    for _ in range(200):
        x, y = free[rng.integers(len(free))]
        reference = Bomb.get_blast_coords(
            _FakeBomb(int(x), int(y), s.BOMB_POWER), field)
        ours = gs.blast_coords(field, int(x), int(y))
        ok &= sorted(reference) == sorted(ours)
    check('200 random positions agree with items.Bomb.get_blast_coords', ok)


class _FakeBomb:
    """Minimal stand-in so we can call the framework's own blast routine."""

    def __init__(self, x, y, power):
        self.x, self.y, self.power = x, y, power


def test_danger_timing():
    print('3. danger timing')
    field = empty_arena()
    for t in range(s.BOMB_TIMER):
        lethal = gs.lethal_schedule(field, [((1, 1), t)], np.zeros(field.shape))
        ks = [k for k in range(lethal.shape[0]) if lethal[k, 1, 1]]
        check(f'bomb with timer {t} is lethal exactly after moves {t},{t + 1}',
              ks == [t, t + 1], str(ks))

    # an explosion already on the board: map value 1 kills now, 0 is smoke
    expl = np.zeros(field.shape)
    expl[3, 1] = 1
    lethal = gs.lethal_schedule(field, [], expl)
    check('explosion_map == 1 is lethal after move 0', bool(lethal[0, 3, 1]))
    check('explosion_map == 1 is harmless afterwards',
          not lethal[1:, 3, 1].any())
    expl[3, 1] = 0
    lethal = gs.lethal_schedule(field, [], expl)
    check('explosion_map == 0 is never lethal', not lethal[:, 3, 1].any())

    # a bomb dropped now leaves exactly four moves to escape radius three
    lethal = gs.with_own_bomb(
        gs.lethal_schedule(field, [], np.zeros(field.shape)), field, (1, 1))
    ks = [k for k in range(lethal.shape[0]) if lethal[k, 1, 1]]
    check('own bomb dropped now is lethal after moves 4,5',
          ks == [s.BOMB_TIMER, s.BOMB_TIMER + 1], str(ks))


def test_escape_search():
    print('4. escape search')
    field = empty_arena()

    # standing on a bomb that goes off this very move: nothing can save us
    lethal = gs.lethal_schedule(field, [((1, 1), 0)], np.zeros(field.shape))
    blocked = gs.blocked_schedule(field, [((1, 1), 0)], [])
    check('no escape from a bomb with timer 0 under our feet',
          gs.survivable_moves(field, lethal, (1, 1), blocked) == 0)

    # a fresh bomb under our feet: escape must exist
    lethal = gs.lethal_schedule(field, [((1, 1), s.BOMB_TIMER - 1)],
                                np.zeros(field.shape))
    blocked = gs.blocked_schedule(field, [((1, 1), s.BOMB_TIMER - 1)], [])
    mask = gs.survivable_moves(field, lethal, (1, 1), blocked)
    check('escape exists from a fresh bomb under our feet', mask != 0,
          f'mask={mask:05b}')

    # dead-end corridor: walling off the side exits must remove the escape
    trap = empty_arena()
    for y in (2, 3, 4):
        trap[2, y] = gs.WALL          # close the side openings of column x=1
    for x in (2, 3, 4):
        trap[x, 2] = gs.WALL
    trap[1, 5] = gs.WALL              # and cap the corridor beyond blast range
    trap[5, 1] = gs.WALL
    lethal = gs.lethal_schedule(trap, [((1, 1), s.BOMB_TIMER - 1)],
                                np.zeros(trap.shape))
    blocked = gs.blocked_schedule(trap, [((1, 1), s.BOMB_TIMER - 1)], [])
    check('no escape when both corridor exits are capped',
          gs.survivable_moves(trap, lethal, (1, 1), blocked) == 0)


def test_escape_search_empirically(rounds=40):
    print('5. escape search, empirically (safe_random_agent must never suicide)')


    # Uses the harness runner so this check also tolerates the synced-folder
    # chdir hazard described in experiments/README.md.
    data = runner.run(runner.Match(
        agents=['safe_random_agent'], scenario='loot-crate', n_rounds=rounds,
        seed=12345, label='selftest_safe_random',
        stats_path=REPO / 'results' / '_selftest_safe_random.json'), retries=4)
    if data is None:
        check('safe_random_agent ran', False, 'see the runner output above')
        return

    agent = data['by_agent']['safe_random_agent']
    suicides = agent.get('suicides', 0)
    bombs = agent.get('bombs', 0)
    check(f'0 suicides in {rounds} rounds while dropping {bombs} bombs',
          suicides == 0, f'suicides={suicides}')
    check('the test was not vacuous (it actually dropped bombs)', bombs > rounds)


def test_symmetry():
    print('6. feature encoder is D4-equivariant')
    spec = feat.FeatureSpec(cfg_module.FULL_BLOCKS)
    field = empty_arena(crate_density=0.4, seed=7)
    state = make_state(field, (3, 5), coins=[(7, 5), (3, 9)],
                       bombs=[((5, 5), 2)], others=[(9, 9)])
    phi, _ = feat.extract(spec, state)

    n = s.COLS - 1
    for gi, gmap in enumerate(symmetry.DIR_MAPS):
        rotated = _apply_symmetry_to_state(state, gi, n)
        phi_rot, _ = feat.extract(spec, rotated)
        expected = phi[spec.index_perms[gi]]
        check(f'g{gi} {gmap}: phi(g.s) == g.phi(s)',
              np.array_equal(phi_rot, expected),
              f'{np.flatnonzero(phi_rot != expected).tolist()}')

    perms = spec.action_perms
    check('WAIT and BOMB are symmetry fixed points',
          bool((perms[:, 4] == 4).all() and (perms[:, 5] == 5).all()))
    check('index permutations are bijections',
          all(sorted(p.tolist()) == list(range(spec.size))
              for p in spec.index_perms))


def _apply_symmetry_to_state(state, gi, n):
    """Map a game state through the ``gi``-th element of D4.

    Point maps chosen so that the induced direction relabelling is exactly
    ``symmetry.DIR_MAPS[gi]``; the checks in :func:`test_symmetry` would fail
    loudly if that correspondence were off.
    """
    def rot(p):      # quarter turn: UP -> RIGHT
        return (n - p[1], p[0])

    def mirror(p):   # reflect about the vertical axis: RIGHT <-> LEFT
        return (n - p[0], p[1])

    # DIR_MAPS is generated as [r^k, r^k . m] for k = 0..3
    k, mirrored = divmod(gi, 2)

    def apply(p):
        q = mirror(p) if mirrored else tuple(p)
        for _ in range(k):
            q = rot(q)
        return q

    field = state['field']
    new_field = np.zeros_like(field)
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            nx, ny = apply((x, y))
            new_field[nx, ny] = field[x, y]

    new_expl = np.zeros_like(state['explosion_map'])
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            nx, ny = apply((x, y))
            new_expl[nx, ny] = state['explosion_map'][x, y]

    out = dict(state)
    out['field'] = new_field
    out['explosion_map'] = new_expl
    out['self'] = state['self'][:3] + (apply(state['self'][3]),)
    out['others'] = [o[:3] + (apply(o[3]),) for o in state['others']]
    out['bombs'] = [(apply(p), t) for p, t in state['bombs']]
    out['coins'] = [apply(c) for c in state['coins']]
    return out


def test_features_sanity():
    print('7. feature sanity')
    spec = feat.FeatureSpec(cfg_module.FULL_BLOCKS)
    field = empty_arena()

    state = make_state(field, (1, 1), coins=[(1, 4)])
    phi, info = feat.extract(spec, state)
    base, _ = spec.offset['dir_coin']
    check('coin straight below -> dir_coin points DOWN',
          phi[base + 2] == 1 and phi[base + 0] == 0,
          f'dir_coin={phi[base:base + 4].tolist()}')
    check('coin distance is the walking distance', info['coin_dist'] == 3,
          str(info['coin_dist']))

    base, _ = spec.offset['dir_free']
    check('walls are not free', phi[base + 0] == 0 and phi[base + 3] == 0,
          f'dir_free={phi[base:base + 4].tolist()}')

    crated = empty_arena()
    crated[1, 2] = gs.CRATE
    state = make_state(crated, (1, 1))
    _phi, info = feat.extract(spec, state)
    check('crate next to us is counted by crate_gain', info['crate_gain'] >= 1,
          str(info['crate_gain']))

    state = make_state(field, (1, 1), others=[(1, 2)])
    _phi, info = feat.extract(spec, state)
    check('an adjacent opponent is in our blast', info['enemy_gain'] == 1,
          str(info['enemy_gain']))
    check('we cannot walk into an opponent', not info['legal'][2])


def test_nstep_returns():
    print('8. n-step returns')
    gamma = 0.9
    acc = replay_mod.NStepAccumulator(3, gamma)
    phis = [np.full(4, i, dtype=np.int8) for i in range(5)]
    rewards = [1.0, 2.0, 3.0, 4.0]

    emitted = []
    for i, r in enumerate(rewards):
        out = acc.push(phis[i], 0, r, phis[i + 1])
        if out is not None:
            emitted.append(out)
    emitted += acc.flush()

    expected_first = 1.0 + gamma * 2.0 + gamma ** 2 * 3.0
    check('first 3-step return is r0 + g r1 + g^2 r2',
          abs(emitted[0][2] - expected_first) < 1e-12,
          f'{emitted[0][2]} vs {expected_first}')
    check('first transition bootstraps from s_3',
          np.array_equal(emitted[0][3], phis[3]))
    check('first transition is not terminal', emitted[0][5] is False)
    check('every step is emitted exactly once', len(emitted) == len(rewards),
          str(len(emitted)))
    # windows that closed while the episode ran bootstrap; only the tails that
    # the flush had to truncate are terminal
    n_full = len(rewards) - 3 + 1
    check('full windows bootstrap', not any(t[5] for t in emitted[:n_full]))
    check('flushed tails are terminal', all(t[5] for t in emitted[n_full:]))

    tail = emitted[-1]
    check('last tail return is just r3', abs(tail[2] - 4.0) < 1e-12, str(tail[2]))


def test_potential_shaping():
    print('9. potential-based shaping')
    cfg = cfg_module.get('task2')
    near = dict(n_coins=1, coin_dist=2, crate_dist=99, enemy_dist=99)
    far = dict(n_coins=1, coin_dist=5, crate_dist=99, enemy_dist=99)
    closer = (cfg['gamma'] * rewards_mod.potential(cfg, near)
              - rewards_mod.potential(cfg, far))
    further = (cfg['gamma'] * rewards_mod.potential(cfg, far)
               - rewards_mod.potential(cfg, near))
    check('moving towards the objective is rewarded', closer > 0, str(closer))
    check('moving away from the objective is penalised', further < 0, str(further))
    check('terminal potential is zero', rewards_mod.potential(cfg, None) == 0.0)

    off = cfg_module.get('task2_noshaping')
    check('shaping can be switched off',
          rewards_mod.potential(off, near) == 0.0)


# --------------------------------------------------------------------------
def main():
    print('q_agent self-test\n' + '=' * 60)
    test_event_names()
    test_blast_geometry()
    test_danger_timing()
    test_escape_search()
    test_symmetry()
    test_features_sanity()
    test_nstep_returns()
    test_potential_shaping()
    test_escape_search_empirically()

    print('=' * 60)
    if FAILURES:
        print(f'{len(FAILURES)} FAILED: {FAILURES}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
