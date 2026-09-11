"""
Little module to generate figures for the report using matplotlib.

    python -m experiments.analyze curves --stage 1
    python -m experiments.analyze bars   --suite stage1
    python -m experiments.analyze all

Just gotta make sure the csv files exist. Usually done via training in results/train/*.csv
Bar charts come from the seed-aggregated evaluation summaries (``results/eval/*_summary.csv``).

All figures are written next to their ``.csv`` holding exactly the numbers plotted,
so the figures never have to be trusted on their own and the report can quote
the table instead of the picture.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAIN_DIR = REPO / 'results' / 'train'
EVAL_DIR = REPO / 'results' / 'eval'
FIG_DIR = REPO / 'figures'

# --- colors ----------
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
        'font.size': 9,
        'text.color': INK, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8,
        'grid.color': GRID, 'grid.linewidth': 0.8,
        'legend.frameon': False, 'figure.dpi': 140,
    })


def tidy(ax, title=None, xlabel=None, ylabel=None):
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc='left', pad=8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def color_for(index):
    """Colour by series identity. Past eight slots we would fold to 'Other'."""
    if index >= len(SERIES):
        raise ValueError('more than 8 series: fold into "Other" or facet')
    return SERIES[index]


def write_table(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# learning curves
# --------------------------------------------------------------------------
def read_train_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            try:
                out[key].append(float(value))
            except (TypeError, ValueError):
                pass
    return out


def rolling(values, window):
    if window <= 1 or len(values) < 2:
        return list(values)
    out, total = [], 0.0
    from collections import deque
    buf = deque()
    for v in values:
        buf.append(v)
        total += v
        if len(buf) > window:
            total -= buf.popleft()
        out.append(total / len(buf))
    return out


#: stage -> (variant order, metrics to plot)
CURVE_SETS = {
    '1': (['task1', 'task1_1step', 'task1_nosym', 'task1_noshaping',
           'task1_linear', 'task1_bombs'],
          [('coins', 'coins collected per round'),
           ('steps', 'round length (steps)'),
           ('suicides', 'suicides per round'),
           ('invalid', 'invalid actions per round')]),
    '1full': (['task1_full'],
              [('coins', 'coins collected per round'),
               ('suicides', 'suicides per round')]),
    '2': (['task2', 'task2_scratch', 'task2_tabular', 'task2_noshaping',
           'task2_nosym', 'task2_noaux', 'task2_unsafe_explore'],
          [('coins', 'coins collected per round'),
           ('crates', 'crates destroyed per round'),
           ('suicides', 'suicides per round'),
           ('steps', 'round length (steps)')]),
    '3': (['task3'], [('score', 'score per round'),
                      ('kills', 'kills per round'),
                      ('coins', 'coins per round'),
                      ('suicides', 'suicides per round')]),
    '4': (['task4'], [('score', 'score per round'),
                      ('kills', 'kills per round'),
                      ('coins', 'coins per round'),
                      ('survived', 'survival rate')]),
}


def plot_curves(stage, window=100, out_name=None):
    if stage not in CURVE_SETS:
        raise SystemExit(f'unknown curve set {stage!r}; known: {sorted(CURVE_SETS)}')
    variants, metrics = CURVE_SETS[stage]

    available = [(v, TRAIN_DIR / f'{v}_train.csv') for v in variants]
    available = [(v, p) for v, p in available if p.is_file()]
    if not available:
        print(f'  (no training logs for stage {stage}, skipping)')
        return None

    style()
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.1 * nrows),
                             squeeze=False)

    table_rows = []
    for panel, (metric, label) in enumerate(metrics):
        ax = axes[panel // ncols][panel % ncols]
        for i, (variant, path) in enumerate(available):
            data = read_train_csv(path)
            if metric not in data:
                continue
            y = rolling(data[metric], window)
            x = data.get('round', list(range(1, len(y) + 1)))
            ax.plot(x, y, color=color_for(i), linewidth=2.0, label=variant,
                    solid_capstyle='round')
            if panel == 0:
                for r, v in zip(x, y):
                    table_rows.append({'variant': variant, 'round': r,
                                       'metric': metric, 'value': round(v, 4)})
        tidy(ax, title=label, xlabel='training round', ylabel=None)

    for panel in range(len(metrics), nrows * ncols):
        axes[panel // ncols][panel % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 4),
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Stage {stage}: training progress '
                 f'(rolling mean over {window} rounds)',
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    out_name = out_name or f'stage{stage}_learning'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{out_name}.png', bbox_inches='tight')
    plt.close(fig)
    write_table(FIG_DIR / f'{out_name}.csv', table_rows,
                ['variant', 'round', 'metric', 'value'])
    print(f'  wrote figures/{out_name}.png (+ .csv)')
    return FIG_DIR / f'{out_name}.png'


# --------------------------------------------------------------------------
# Evaluation Bar Charts
# --------------------------------------------------------------------------
#: Frankl,y alive ``alive_fraction`` is meaningless in solo suites. Because the round ends when the lone
#: agent dies, so it is always 1.0); ``suicides`` and ``steps`` carry survival there instead.
BAR_METRICS = {
    'stage1': ['score', 'coins', 'suicides', 'steps'],
    'stage2': ['score', 'coins', 'suicides', 'crates_per_bomb'],
    'stage3': ['score', 'kills', 'suicides', 'alive_fraction'],
    'stage4': ['score', 'kills', 'suicides', 'alive_fraction'],
    'progression': ['score', 'kills', 'coins', 'alive_fraction'],
}
DEFAULT_BAR_METRICS = ['score', 'coins', 'suicides', 'alive_fraction']


def plot_bars(suite, metrics=None, out_name=None):
    path = EVAL_DIR / f'{suite}_summary.csv'
    if not path.is_file():
        print(f'  (no evaluation summary for {suite}, skipping)')
        return None
    metrics = metrics or BAR_METRICS.get(suite, DEFAULT_BAR_METRICS)

    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r['under_test'] in ('True', 'true')]
    if not rows:
        print(f'  (no rows under test in {path.name})')
        return None

    arms, seen = [], set()
    for row in rows:
        if row['arm'] not in seen:
            seen.add(row['arm'])
            arms.append(row['arm'])

    lookup = {(r['arm'], r['metric']): r for r in rows}

    style()
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5, 3.4 * nrows),
                             squeeze=False)

    table_rows = []
    for panel, metric in enumerate(metrics):
        ax = axes[panel // ncols][panel % ncols]
        values, errors, labels = [], [], []
        for arm in arms:
            rec = lookup.get((arm, metric))
            if rec is None:
                continue
            mean = float(rec['mean'])
            sem = float(rec['sem']) if rec['sem'] not in ('', 'nan') else 0.0
            values.append(mean)
            errors.append(sem)
            labels.append(arm)
            table_rows.append({'arm': arm, 'metric': metric,
                               'mean': round(mean, 4), 'sem': round(sem, 4),
                               'n_seeds': rec['n_seeds']})

        positions = range(len(values))

        bars = ax.bar(positions, values, color=SERIES[0], width=0.68,
                      yerr=errors, capsize=3,
                      error_kw={'ecolor': INK_2, 'elinewidth': 1.2})
        for rect, value in zip(bars, values):
            ax.annotate(f'{value:.2f}',
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords='offset points', xytext=(0, 3),
                        ha='center', fontsize=8, color=INK_2)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=8)
        tidy(ax, title=metric.replace('_', ' '))

    for panel in range(len(metrics), nrows * ncols):
        axes[panel // ncols][panel % ncols].axis('off')

    fig.suptitle(f'{suite}: evaluation (mean +- SEM over seeds)',
                 fontsize=11, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_name = out_name or f'{suite}_eval'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f'{out_name}.png', bbox_inches='tight')
    plt.close(fig)
    write_table(FIG_DIR / f'{out_name}.csv', table_rows,
                ['arm', 'metric', 'mean', 'sem', 'n_seeds'])
    print(f'  wrote figures/{out_name}.png (+ .csv)')
    return FIG_DIR / f'{out_name}.png'


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    c = sub.add_parser('curves')
    c.add_argument('--stage', default='1')
    c.add_argument('--window', type=int, default=100)

    b = sub.add_parser('bars')
    b.add_argument('--suite', default='stage1')
    b.add_argument('--metrics', nargs='+', default=None)

    sub.add_parser('all')

    args = parser.parse_args(argv)

    if args.command == 'curves':
        plot_curves(args.stage, args.window)
    elif args.command == 'bars':
        plot_bars(args.suite, args.metrics)
    else:
        print('learning curves:')
        for stage in CURVE_SETS:
            plot_curves(stage)
        print('evaluation bars:')
        for suite in ('stage1', 'stage2', 'stage3', 'stage4', 'progression'):
            plot_bars(suite)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
