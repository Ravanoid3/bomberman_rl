"""
plot_training.py  --  visualise training progress from training_stats.csv
Place in the bomberman_rl/ root and run:

    python plot_training.py
    python plot_training.py --csv agent_code/agent_test/training_stats.csv
    python plot_training.py --smooth 100
    python plot_training.py --save      # saves PNG instead of showing

CSV columns written by train.py:
  round, score, avg_score_200, reward, avg_reward_200,
  coins, kills, suicides, crates_destroyed, alive_fraction
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def smooth(series, window):
    return series.rolling(window, min_periods=1).mean()


def print_table(df, last_n):
    """Print a summary table like the one in the mate's curriculum runner."""
    last = df.tail(last_n)
    print()
    print(f"{'Metric':<22} {'Last round':>12} {'Avg last ' + str(last_n):>15}")
    print("-" * 52)
    rows = [
        ("Score",          df['score'].iloc[-1],       last['score'].mean()),
        ("Reward",         df['reward'].iloc[-1],      last['reward'].mean()),
        ("Coins",          df['coins'].iloc[-1],       last['coins'].mean()),
        ("Kills",          df['kills'].iloc[-1],       last['kills'].mean()),
        ("Suicides",       df['suicides'].iloc[-1],    last['suicides'].mean()),
        ("Crates destroyed", df['crates_destroyed'].iloc[-1], last['crates_destroyed'].mean()),
        ("Alive fraction", df['alive_fraction'].iloc[-1], last['alive_fraction'].mean()),
    ]
    for name, last_val, avg_val in rows:
        print(f"  {name:<20} {last_val:>12.2f} {avg_val:>15.3f}")
    print()


def plot(csv_path: pathlib.Path, smooth_window: int, save: bool):
    df = pd.read_csv(csv_path)
    n  = len(df)
    print(f"Loaded {n} rounds from {csv_path}")
    if n == 0:
        print("No data yet -- train some rounds first.")
        return

    w = smooth_window

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Training stats  ({n} rounds,  smoothing window = {w})", fontsize=13)
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. game score (top, full width)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(df['round'], smooth(df['score'], w),     label=f'score (smoothed {w})', color='steelblue')
    ax.plot(df['round'], df['avg_score_200'],         label='200-round avg',         color='orange', linewidth=1.8)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_title('Game score per round  (1 pt per coin, 5 pts per kill)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)

    # 2. shaped reward
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['round'], smooth(df['reward'], w),  color='mediumpurple', label=f'reward (smoothed {w})')
    ax2.plot(df['round'], df['avg_reward_200'],      color='purple',       linewidth=1.4, label='200-round avg')
    ax2.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax2.set_title('Shaped reward per round')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Reward')
    ax2.legend(fontsize=7)

    # 3. coins per round
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['round'], smooth(df['coins'], w), color='gold')
    ax3.set_title('Coins collected per round')
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Coins')

    # 4. kills and suicides
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['round'], smooth(df['kills'],    w), color='crimson', label='kills')
    ax4.plot(df['round'], smooth(df['suicides'], w), color='salmon',  label='suicides', linestyle='--')
    ax4.set_title('Kills and suicides per round')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Count')
    ax4.legend(fontsize=8)

    # 5. survival rate
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df['round'], smooth(df['alive_fraction'], w) * 100, color='seagreen')
    ax5.set_title('Survival rate (%)')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('%')
    ax5.set_ylim(0, 105)

    if save:
        out = csv_path.with_suffix('.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {out}")
    else:
        plt.show()

    print_table(df, min(200, n))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv',    default='agent_code/agent_test/training_stats.csv',
                        help='path to training_stats.csv (default: agent_code/agent_test/training_stats.csv)')
    parser.add_argument('--smooth', type=int, default=50,
                        help='rolling-average window in rounds (default: 50)')
    parser.add_argument('--save',   action='store_true',
                        help='save the plot as PNG next to the CSV instead of displaying it')
    args = parser.parse_args()
    plot(pathlib.Path(args.csv), args.smooth, args.save)