"""Named agent configurations

Every agent variant in this project is the same code with a different config really.
That is deliberate: it makes each experiment a controlled comparison in which exactly the listed config differs.

Names ending in a bare task number (``task1`` ... ``task4``) are the curriculum agents.
"""

import copy

from .features import ALL_BLOCKS

# Full feature set, used from stage 2 onwards so checkpoints stay compatible
FULL_BLOCKS = tuple(name for name, _w, _d in ALL_BLOCKS)

# Bare minimum for "walk to the nearest coin" -- a state space small enough
# that a tabular model converges within minutes.
MINIMAL_BLOCKS = ('dir_free', 'dir_coin', 'coin_dist', 'bias')

# No opponent-related blocks, needed if the agent plays alone in training e.g.
SOLO_BLOCKS = tuple(b for b in FULL_BLOCKS
                    if b not in ('dir_enemy', 'enemy_dist', 'enemy_gain'))


BASE = dict(
    # ---- representation -------------------------------------------------
    feature_blocks=FULL_BLOCKS,
    model='linear',              # 'linear' | 'tabular'

    # ---- learning -------------------------------------------------------
    gamma=0.95,
    n_step=3,
    lr=0.02,
    lr_decay=0.0,                # tabular: alpha = lr / (1 + lr_decay * visits)
    batch_size=128,
    updates_per_step=1,
    grad_clip=5.0,
    memory=100_000,
    rare_fraction=0.25,          # share of each batch drawn from the rare pool
    target_sync=1_000,           # gradient steps between target-network syncs
    symmetry_augment=True,       # 8x D4 data augmentation

    # ---- action space ---------------------------------------------------
    # Stage 1 is defined as a pure navigation task ("does not require dropping
    # any bombs").  Offering BOMB there is not merely unnecessary, it is
    # unlearnable with the minimal feature set: the agent dies four steps after
    # the drop, and a representation without danger features cannot tell the
    # fatal state from a safe one, so the penalty never reaches the BOMB action.
    # The 'task1_bombs' arm documents that failure; every other stage-1 arm
    # drops BOMB from the action space and stage 2 restores it.
    enable_bomb=True,

    # ---- exploration ----------------------------------------------------
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_rounds=500,      # time constant of the exponential decay
    eps_greedy_bias=True,        # explore over legal actions only
    safe_explore=0.9,            # share of exploration restricted to safe actions

    # ---- reward shaping -------------------------------------------------
    shaping=True,
    shaping_weight=0.15,         # weight of the potential Phi = -w * distance
    shaping_cap=15,              # distances are clipped here
    step_penalty=-0.05,
    rewards={},                  # merged into DEFAULT_REWARDS

    # ---- inference ------------------------------------------------------
    # The feature encoding is deliberately memoryless, which makes the greedy
    # policy vulnerable to cycles -- and WAIT is an *absorbing* cycle of length
    # one, so a single stale over-estimate can freeze the agent for the rest of
    # the round.  Measured on stage 1: 50.0 coins/round while training (residual
    # exploration kept breaking the cycle) versus 9.1 at eval with eps = 0.
    # The loop breaker adds the minimum missing state -- where have I just been --
    # as a decision-time penalty on revisiting recent tiles.  It is scaled by the
    # spread of the Q-values so one setting works across stages, and it does not
    # touch the learning targets, only action selection.
    loop_penalty=0.34,           # fraction of the Q-spread charged per recent visit
    loop_memory=8,               # how many past positions count as "recent"
    safety_mask=False,           # veto provably fatal actions at decision time
    eval_eps=0.0,                # exploration outside training mode
    greedy_tie_break='random',

    # ---- checkpointing --------------------------------------------------
    save_every=200,              # rounds between checkpoint writes

    # ---- bookkeeping ----------------------------------------------------
    checkpoint='task1.pt',
    init_from=None,              # checkpoint to warm-start from
    train_scenario='coin-heaven',
    log_csv=None,                # defaults to '<checkpoint stem>_train.csv'
)


# Reward values in "internal" units.
# Real game rewards:
# Coin: 1
# Kill: 5
# We scale those up and add auxiliary terms. Auxiliary terms are *not*
# present in official games, so every one of them is an ablation candidate.
DEFAULT_REWARDS = {
    # genuine game events
    'COIN_COLLECTED': 17.0,
    'KILLED_OPPONENT': 25.0,
    'KILLED_SELF': -150.0,
    'GOT_KILLED': -100.0,
    'CRATE_DESTROYED': 1.3,
    'COIN_FOUND': 0.7,
    'SURVIVED_ROUND': 3.0,
    'OPPONENT_ELIMINATED': 0.5,
    'INVALID_ACTION': -1.0,
    'WAITED': -0.2,
    'IN_LEAD': 0.1,
    'IN_CORNER': -1,

    # auxiliary events raised by rewards.py
    'SUICIDAL_BOMB': -30.0,       # dropped a bomb with no escape route
    'UNSAFE_MOVE': -1.5,         # stepped somewhere with no escape route
    'USELESS_BOMB': -0.6,        # bomb that can hit neither crate nor opponent
    'GOOD_BOMB': 1.5,            # bomb next to crates, escape available
    'ATTACK_BOMB': 2.0,          # bomb that can catch an opponent
    'ESCAPED_DANGER': 0.8,
    'LINGERED_IN_DANGER': -0.5,
    'ENTERED_DANGER': -0.4,
}

def _cfg(**overrides):
    cfg = copy.deepcopy(BASE)
    rewards = copy.deepcopy(DEFAULT_REWARDS)
    rewards.update(overrides.pop('rewards', {}))
    cfg.update(overrides)
    cfg['rewards'] = rewards
    return cfg

CONFIGS = {
    # ------------------------------------------------------------------
    # Stage 1 -- collect revealed coins on an empty board.
    # ------------------------------------------------------------------
    'task1': _cfg(
        feature_blocks=MINIMAL_BLOCKS,
        model='tabular',
        lr=0.15,
        batch_size=32,
        n_step=3,
        eps_decay_rounds=150,
        eps_end=0.02,
        enable_bomb=False,
        checkpoint='task1.pt',
        train_scenario='coin-heaven',
        # We explicitly add MORE emphasis to coin collection
        rewards={'COIN_COLLECTED': 3.0, 'INVALID_ACTION': -1.0, 'WAITED': -0.5},
    ),
    # same task, richer representation -- seeds the stage-2 warm start
    'task1_full': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.02,
        eps_decay_rounds=250,
        eps_end=0.02,
        checkpoint='task1_full.pt',
        train_scenario='coin-heaven',
    ),
    # ablation arms for stage 1
    'task1_linear': _cfg(
        feature_blocks=MINIMAL_BLOCKS, model='linear', lr=0.05,
        eps_decay_rounds=150, eps_end=0.02,
        enable_bomb=False,
        checkpoint='task1_linear.pt', train_scenario='coin-heaven',
    ),
    'task1_noshaping': _cfg(
        feature_blocks=MINIMAL_BLOCKS, model='tabular', lr=0.15, batch_size=32,
        eps_decay_rounds=150, eps_end=0.02, shaping=False,
        enable_bomb=False,
        checkpoint='task1_noshaping.pt', train_scenario='coin-heaven',
    ),
    'task1_nosym': _cfg(
        feature_blocks=MINIMAL_BLOCKS, model='tabular', lr=0.15, batch_size=32,
        eps_decay_rounds=150, eps_end=0.02, symmetry_augment=False,
        enable_bomb=False,
        checkpoint='task1_nosym.pt', train_scenario='coin-heaven',
    ),
    'task1_1step': _cfg(
        feature_blocks=MINIMAL_BLOCKS, model='tabular', lr=0.15, n_step=1, batch_size=32,
        eps_decay_rounds=150, eps_end=0.02,
        enable_bomb=False,
        checkpoint='task1_1step.pt', train_scenario='coin-heaven',
    ),

    # documents why stage 1 drops BOMB: same agent, bombs allowed, cannot learn
    # not to blow itself up because its features cannot express danger
    'task1_bombs': _cfg(
        feature_blocks=MINIMAL_BLOCKS, model='tabular', lr=0.15, batch_size=32,
        eps_decay_rounds=150, eps_end=0.02, enable_bomb=True,
        checkpoint='task1_bombs.pt', train_scenario='coin-heaven',
    ),

    # ------------------------------------------------------------------
    # Stage 2 -- crates, bombs, no opponents.
    # ------------------------------------------------------------------
    'task2': _cfg(
        feature_blocks=SOLO_BLOCKS,
        model='linear',
        lr=0.02,
        eps_decay_rounds=1_000,
        eps_end=0.05,
        checkpoint='task2.pt',
        init_from='task1_full.pt',
        train_scenario='loot-crate',
    ),
    'task2_scratch': _cfg(
        feature_blocks=SOLO_BLOCKS, model='linear', lr=0.02,
        eps_decay_rounds=1_000, checkpoint='task2_scratch.pt',
        init_from=None, train_scenario='loot-crate',
    ),
    'task2_tabular': _cfg(
        feature_blocks=SOLO_BLOCKS, model='tabular', lr=0.1, batch_size=32,
        eps_decay_rounds=1_000, checkpoint='task2_tabular.pt',
        train_scenario='loot-crate',
    ),
    'task2_noshaping': _cfg(
        feature_blocks=SOLO_BLOCKS, model='linear', lr=0.02, shaping=False,
        eps_decay_rounds=1_000, checkpoint='task2_noshaping.pt',
        init_from='task1_full.pt', train_scenario='loot-crate',
    ),
    'task2_nosym': _cfg(
        feature_blocks=SOLO_BLOCKS, model='linear', lr=0.02,
        symmetry_augment=False, eps_decay_rounds=1_000,
        checkpoint='task2_nosym.pt', init_from='task1_full.pt',
        train_scenario='loot-crate',
    ),
    'task2_noaux': _cfg(
        # only the events the real game rewards, plus the step penalty
        feature_blocks=SOLO_BLOCKS, model='linear', lr=0.02,
        eps_decay_rounds=1_000, checkpoint='task2_noaux.pt',
        init_from='task1_full.pt', train_scenario='loot-crate',
        rewards={'SUICIDAL_BOMB': 0.0, 'UNSAFE_MOVE': 0.0, 'USELESS_BOMB': 0.0,
                 'GOOD_BOMB': 0.0, 'ATTACK_BOMB': 0.0, 'ESCAPED_DANGER': 0.0,
                 'LINGERED_IN_DANGER': 0.0, 'ENTERED_DANGER': 0.0,
                 'CRATE_DESTROYED': 0.0, 'COIN_FOUND': 0.0},
    ),

    # documents why stage 2 needs guided exploration: uniform random actions
    # end the episode within ~10 steps, so nothing is ever learned
    'task2_unsafe_explore': _cfg(
        feature_blocks=SOLO_BLOCKS, model='linear', lr=0.02, safe_explore=0.0,
        eps_decay_rounds=1_000, checkpoint='task2_unsafe_explore.pt',
        init_from='task1_full.pt', train_scenario='loot-crate',
    ),

    # ------------------------------------------------------------------
    # Stage 3 -- hunt weak opponents.
    # ------------------------------------------------------------------
    'task3': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.015,
        eps_decay_rounds=2_500,
        eps_end=0.05,
        checkpoint='task3.pt',
        init_from=['task2.pt', 'task1_full.pt'],
        train_scenario='classic',
    ),

    # ------------------------------------------------------------------
    # Stage 4 -- full game against rule_based_agent / self-play.
    # ------------------------------------------------------------------
    'task4': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        lr=0.01,
        eps_decay_rounds=3_000,
        eps_end=0.03,
        checkpoint='task4.pt',
        init_from=['task3.pt', 'task2.pt', 'task1_full.pt'],
        train_scenario='classic',
    ),

    # ------------------------------------------------------------------
    # Tournament entry: best checkpoint plus the safety veto.
    # ------------------------------------------------------------------
    'tournament': _cfg(
        feature_blocks=FULL_BLOCKS,
        model='linear',
        safety_mask=True,
        eps_end=0.0,
        checkpoint='task4.pt',
        train_scenario='classic',
    ),
    'tournament_nosafety': _cfg(
        feature_blocks=FULL_BLOCKS, model='linear', safety_mask=False,
        eps_end=0.0, checkpoint='task4.pt', train_scenario='classic',
    ),
    # ablation: how much of the score is the loop breaker rather than the policy
    'tournament_noloop': _cfg(
        feature_blocks=FULL_BLOCKS, model='linear', safety_mask=True,
        loop_penalty=0.0, eps_end=0.0, checkpoint='task4.pt',
        train_scenario='classic',
    ),
}


def get(name):
    """Return a deep copy of the named configuration."""
    if name not in CONFIGS:
        raise KeyError(f'unknown config {name!r}; known: {sorted(CONFIGS)}')
    cfg = copy.deepcopy(CONFIGS[name])
    cfg['name'] = name
    if cfg.get('log_csv') is None:
        cfg['log_csv'] = cfg['checkpoint'].replace('.pt', '') + '_train.csv'
    return cfg
