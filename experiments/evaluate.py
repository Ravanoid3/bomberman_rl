"""Evaluation suites: run matchups over several seeds and tabulate metrics.

    python -m experiments.evaluate --suite stage1
    python -m experiments.evaluate --suite stage4 --rounds 200 --seeds 5

Protocol
--------
Every arm is played for ``--rounds`` rounds under each of ``--seeds``
different world seeds. We then batch these seeds into one observation and the reported uncertainty about an action
is the standard error over seeds (NOT ROUNDS).
All runs within a batch share the same coin and crate layout sequence.

Two files land in ``results/eval/``:

``<suite>_raw.csv``
    one row per (arm, agent, seed) with every metric
``<suite>_summary.csv``
    mean and standard error over seeds, one row per (arm, agent, metric)
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import runner  # noqa: E402

EVAL_DIR = REPO / 'results' / 'eval'

# Our learning agent lives in one folder; which variant it plays is chosen per
# match through the environment (see ``agent_code/q_agent/api.py``).
# ``q_sparring`` is the same code in a second folder, which is the only way to
# get two differently configured variants into a single match.
AGENT = 'q_agent'
SPARRING = 'q_sparring'


def ours(config, *opponents, label=None):
    """An arm in which our agent plays ``config``.

    ``opponents`` are folder names, or ``(SPARRING, config)`` for one of our own
    variants.
    """
    # config goes in through env vars, the folder itself never changes
    folders, env = [], {'Q_AGENT_CONFIG': config}
    for entry in opponents:
        if isinstance(entry, tuple):
            folders.append(entry[0])
            env['Q_SPARRING_CONFIG'] = entry[1]
        else:
            folders.append(entry)
    return (label or config, [AGENT] + folders, env)


def baseline(name, *opponents, label=None):
    """An arm in which a provided agent is the one under test."""
    # no env needed here, the given agents don't read any of our config
    return (label or name, [name] + list(opponents), {})


#: Headline columns for the console table, per suite kind.
#:
#: ``alive_fraction`` is deliberately absent from the solo suites. See comment in analyze, but a solo agent can only kill itself XD
# Also agent steps = world steps and the metric is pinned at 1.0 no matter how often the agent blew itself up.
# For solo play ``suicides`` and ``steps`` carry that information instead; ``alive_fraction``
# only becomes meaningful once other agents keep the round running after a death.

SOLO_HEADLINE = ('score', 'coins', 'suicides', 'steps')
VERSUS_HEADLINE = ('score', 'kills', 'suicides', 'alive_fraction')

RULE_BASED_3 = ['rule_based_agent'] * 3

SUITES = {
    # ------------------------------------------------------------------
    # Stage 1 -- pure navigation.  Success = collect all 50 coins fast.
    # ------------------------------------------------------------------
    'stage1': dict(
        scenario='coin-heaven', rounds=100, headline=SOLO_HEADLINE,
        arms=[ours(c) for c in ('task1', 'task1_linear', 'task1_noshaping',
                                'task1_nosym', 'task1_1step', 'task1_bombs',
                                'task1_full')]
             + [baseline(b) for b in ('coin_collector_agent', 'rule_based_agent',
                                      'random_agent', 'safe_random_agent')],
    ),

    # ------------------------------------------------------------------
    # Stage 2 -- crates and bombs, alone.  Success = survive and loot.
    # ------------------------------------------------------------------
    'stage2': dict(
        scenario='loot-crate', rounds=100, headline=SOLO_HEADLINE,
        arms=[ours(c) for c in ('task2', 'task2_scratch', 'task2_tabular',
                                'task2_noshaping', 'task2_nosym', 'task2_noaux',
                                'task2_unsafe_explore')]
             + [baseline(b) for b in ('coin_collector_agent', 'rule_based_agent',
                                      'safe_random_agent')],
    ),

    # ------------------------------------------------------------------
    # Stage 3 -- hunting.  peaceful_agent is easy prey, coin_collector hard.
    # ------------------------------------------------------------------
    'stage3': dict(
        scenario='classic', rounds=100, headline=VERSUS_HEADLINE,
        arms=[
            ours('task3', 'peaceful_agent', label='task3 vs peaceful'),
            ours('task3', 'coin_collector_agent', label='task3 vs coin_collector'),
            baseline('rule_based_agent', 'peaceful_agent',
                     label='rule_based vs peaceful'),
            baseline('rule_based_agent', 'coin_collector_agent',
                     label='rule_based vs coin_collector'),
        ],
    ),

    # ------------------------------------------------------------------
    # Stage 4 -- the tournament setting, plus its controls and ablations.
    # ------------------------------------------------------------------
    'stage4': dict(
        scenario='classic', rounds=100, headline=VERSUS_HEADLINE,
        arms=[
            ours('task4', *RULE_BASED_3, label='task4 vs 3 rule_based'),
            ours('task4_selfplay', *RULE_BASED_3,
                 label='task4_selfplay vs 3 rule_based'),
            ours('tournament', *RULE_BASED_3, label='tournament vs 3 rule_based'),
            ours('tournament_nosafety', *RULE_BASED_3,
                 label='tournament, no safety veto'),
            ours('tournament_noloop', *RULE_BASED_3,
                 label='tournament, no loop breaker'),
            # Control: what one rule_based_agent scores against three of itself.
            # In a four-way game the coins are shared, so this is what "par"
            # means -- comparing against its solo score would flatter us.
            baseline('rule_based_agent', *RULE_BASED_3,
                     label='rule_based vs 3 rule_based'),
        ],
    ),

    # ------------------------------------------------------------------
    # Curriculum progression: every checkpoint on the same stage-4 task.
    # ------------------------------------------------------------------
    'progression': dict(
        scenario='classic', rounds=100, headline=VERSUS_HEADLINE,
        arms=[ours(c, *RULE_BASED_3, label=f'{c} vs 3 rule_based')
              for c in ('task1_full', 'task2', 'task3', 'task4')]
             + [baseline('rule_based_agent', *RULE_BASED_3,
                         label='rule_based vs 3 rule_based')],
    ),

    # ------------------------------------------------------------------
    # Head-to-head between our own curriculum stages.
    # ------------------------------------------------------------------
    'selfplay': dict(
        scenario='classic', rounds=100, headline=VERSUS_HEADLINE,
        arms=[
            ours('tournament', (SPARRING, 'task2'), label='tournament vs task2'),
            ours('tournament', (SPARRING, 'task3'), label='tournament vs task3'),
            ours('task3', (SPARRING, 'task2'), label='task3 vs task2'),
        ],
    ),
}

METRICS = ['score', 'coins', 'kills', 'suicides', 'crates', 'bombs', 'steps',
           'invalid_rate', 'crates_per_bomb', 'alive_fraction', 'think_ms']


def unique_names(agents):
    """Reproduce ``BombeRLeWorld.setup_agents`` naming for duplicated folders."""
    # three rule_based_agents become _0/_1/_2, a single one keeps its plain name.
    # has to match the game exactly or we look up the wrong agent in the stats.
    counts = defaultdict(int)
    names = []
    for folder in agents:
        if agents.count(folder) > 1:
            names.append(f'{folder}_{counts[folder]}')
        else:
            names.append(folder)
        counts[folder] += 1
    return names


def build_matches(suite_name, suite, rounds, seeds):
    # one match per (arm, seed), plus a parallel list of keys so we can find our
    # way back from a result to the arm it came from
    matches, keys = [], []
    for arm_label, agents, env in suite['arms']:
        for seed in seeds:
            label = f'{suite_name}__{_slug(arm_label)}__seed{seed}'
            matches.append(runner.Match(
                agents=list(agents), scenario=suite['scenario'],
                n_rounds=rounds, seed=seed, label=label, env=dict(env),
                stats_path=EVAL_DIR / 'raw' / f'{label}.json'))
            keys.append((arm_label, tuple(agents), seed))
    return matches, keys


def _slug(text):
    # labels have spaces and commas in them, filenames shouldn't
    return ''.join(c if c.isalnum() else '_' for c in text).strip('_')


def collect(keys, results):
    # flatten everything into one long table, one row per agent in a run.
    # position 0 is always the agent we actually care about.
    rows = []
    for (arm_label, agents, seed), stats in zip(keys, results):
        if stats is None:  # match crashed or timed out, just skip it
            continue
        names = unique_names(list(agents))
        for position, (folder, name) in enumerate(zip(agents, names)):
            metrics = runner.agent_metrics(stats, name)
            if metrics is None:
                continue
            rows.append({
                'arm': arm_label, 'agent': name, 'folder': folder,
                'position': position, 'under_test': position == 0,
                'seed': seed, **{m: metrics[m] for m in METRICS},
            })
    return rows

def summarise(rows):
    # mean +- standard error over the seeds (so n = number of seeds, not rounds)
    grouped = defaultdict(list)
    for row in rows:
        for metric in METRICS:
            grouped[(row['arm'], row['agent'], row['under_test'], metric)].append(
                row[metric])

    out = []
    for (arm, agent, under_test, metric), values in sorted(grouped.items()):
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            var = sum((v - mean) ** 2 for v in values) / (n - 1)
            sem = math.sqrt(var / n)
        else:
            sem = float('nan')  # one seed -> no error bar to give
        out.append({'arm': arm, 'agent': agent, 'under_test': under_test,
                    'metric': metric, 'n_seeds': n, 'mean': mean, 'sem': sem})
    return out


def write_csv(path, rows, fields):
    # nothing fancy, we just want something to put in the report or somethign idk XD
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  wrote {path.relative_to(REPO)} ({len(rows)} rows)')


def print_table(summary, suite_name, headline=VERSUS_HEADLINE):
    # because csvs ain't enough XP
    print(f'\n{suite_name}: agents under test')
    width = max((len(r['arm']) for r in summary), default=10)
    print('  ' + 'arm'.ljust(width) + ''.join(f'{m:>18s}' for m in headline))
    seen = set()
    for row in summary:
        if not row['under_test'] or row['arm'] in seen:
            continue
        seen.add(row['arm'])  # one line per arm, not per metric row
        cells = []
        for metric in headline:
            match = next((r for r in summary
                          if r['arm'] == row['arm'] and r['agent'] == row['agent']
                          and r['metric'] == metric), None)
            cells.append(f'{match["mean"]:>10.3f}+-{match["sem"]:<6.3f}'
                         if match else f'{"-":>18s}')
        print('  ' + row['arm'].ljust(width) + ''.join(cells))


def main(argv=None):
    # run all the matches of a suite, dump the csvs, print the table
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True, choices=sorted(SUITES))
    parser.add_argument('--rounds', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=None)
    args = parser.parse_args(argv)

    suite = SUITES[args.suite]
    rounds = args.rounds or suite['rounds']
    seeds = list(range(1, args.seeds + 1))

    matches, keys = build_matches(args.suite, suite, rounds, seeds)
    results = runner.run_all(matches, workers=args.workers, timeout=args.timeout)

    rows = collect(keys, results)
    if not rows:
        print('no results collected')
        return 1

    raw_fields = ['arm', 'agent', 'folder', 'position', 'under_test', 'seed'] + METRICS
    write_csv(EVAL_DIR / f'{args.suite}_raw.csv', rows, raw_fields)

    summary = summarise(rows)
    write_csv(EVAL_DIR / f'{args.suite}_summary.csv', summary,
              ['arm', 'agent', 'under_test', 'metric', 'n_seeds', 'mean', 'sem'])
    print_table(summary, args.suite, suite.get('headline', VERSUS_HEADLINE))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
