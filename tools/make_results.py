#!/usr/bin/env python3
# Copyright 2026 Mahmud
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Turn the raw benchmark output (results/raw/, from tools/run_benchmarks.sh) into tables and plots.

Reads, for every run k found:
    raw/semantic_map_run<k>.yaml   semantic map saved at the end of exploration
    raw/explore_run<k>.csv         exploration_monitor.py log
    raw/nav_run<k>.csv             nav_benchmark.py log
Writes into results/:
    RESULTS.md                     every table (Markdown, pasted into the README)
    semantic_map_per_class.csv     one row per (run, class)
    *.png                          plots
Every "mean (var)" is the mean over runs and the SAMPLE variance (n - 1).

Usage (workspace sourced):  python3 tools/make_results.py
(RESULTS_DIR=<dir> reads <dir>/raw and writes into <dir> instead.)
"""

import csv
import glob
import importlib.util
import math
import os
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from semantic_nav_bringup.benchmark_stats import mean_var, OUTCOMES  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.environ.get('RESULTS_DIR', os.path.join(ROOT, 'results'))
RAW = os.path.join(RESULTS, 'raw')
SCRIPTS = os.path.join(ROOT, 'src', 'semantic_nav_bringup', 'scripts')
TRUTH = os.path.join(ROOT, 'src', 'semantic_nav_bringup', 'config', 'ground_truth.yaml')
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
OUTCOME_COLORS = {'reached': '#2ca02c', 'wrong_place': '#ff7f0e', 'not_in_map': '#7f7f7f',
                  'no_goal': '#9467bd', 'nav_failed': '#d62728', 'timeout': '#8c564b'}


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def runs_available():
    found = [re.search(r'run(\d+)', p).group(1)
             for p in glob.glob(os.path.join(RAW, 'semantic_map_run*.yaml'))]
    return sorted(found, key=int)


def mv(values, spec='.2f', sd=True):
    """'mean ± sd' over runs (sd = sqrt of the sample variance)."""
    mean, var, n = mean_var(values)
    if mean is None:
        return '–'
    if var is None:
        return f'{mean:{spec}}'
    return f'{mean:{spec}} ± {math.sqrt(var):{spec}}' if sd else f'{var:{spec}}'


def f(value, spec='.2f'):
    return '–' if value is None or (isinstance(value, float) and math.isnan(value)) \
        else format(value, spec)


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out) + '\n'


# ------------------------------------------------------------------ semantic map
def semantic_map_section(runs, md):
    ev = load_module('evaluate_semantic_map')
    truth = ev.load_truth(TRUTH)
    per_run = {k: ev.evaluate(os.path.join(RAW, f'semantic_map_run{k}.yaml'), truth, 0.5)
               for k in runs}
    with open(os.path.join(RESULTS, 'semantic_map_per_class.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=ev.FIELDS)
        w.writeheader()
        for k, rows in per_run.items():
            for label, r in rows.items():
                w.writerow({'run': k, 'label': label, **{c: r[c] for c in ev.FIELDS[2:]}})

    md.append('## Semantic map vs ground truth\n')
    md.append('Matched one-to-one (same class, within 0.5 m of the object footprint). '
              'Precision = TP / map objects, recall = TP / real objects, error = '
              'distance from the map position to the object centre.\n')
    rows = []
    for k, r in per_run.items():
        a = r['ALL']
        rows.append([f'run {k}', a['mapped'], a['tp'], a['fp'], a['fn'], f(a['precision']),
                     f(a['recall']), f(a['mean_error'], '.3f'), f(a['median_error'], '.3f')])
    alls = [r['ALL'] for r in per_run.values()]
    for name, sd in (('mean ± sd', True), ('variance', False)):
        rows.append([f'**{name}**'] + [mv([a[c] for a in alls], '.1f', sd)
                                       for c in ('mapped', 'tp', 'fp', 'fn')] +
                    [mv([a['precision'] for a in alls], '.2f', sd),
                     mv([a['recall'] for a in alls], '.2f', sd),
                     mv([a['mean_error'] for a in alls], '.3f', sd),
                     mv([a['median_error'] for a in alls], '.3f', sd)])
    md.append(table(['run', 'map objects', 'TP', 'FP', 'FN', 'precision', 'recall',
                     'mean err (m)', 'median err (m)'], rows))

    labels = [lbl for lbl in next(iter(per_run.values())) if lbl != 'ALL']
    for rows_k in per_run.values():
        labels += [lbl for lbl in rows_k if lbl not in labels and lbl != 'ALL']
    rows = []
    for label in labels:
        rs = [r[label] for r in per_run.values() if label in r]
        errs = [e for r in rs for e in r['errors']]
        rows.append([label, rs[0]['truth'], sum(r['tp'] for r in rs), sum(r['fp'] for r in rs),
                     sum(r['fn'] for r in rs), mv([r['precision'] for r in rs]),
                     mv([r['recall'] for r in rs]),
                     f(statistics.mean(errs), '.3f') if errs else '–',
                     f(statistics.median(errs), '.3f') if errs else '–'])
    md.append(f'\nPer class (TP/FP/FN summed over {len(per_run)} runs; precision and recall '
              'mean ± sd over runs, "–" = undefined in every run; errors pooled over runs):\n')
    md.append(table(['class', 'real objects', 'TP', 'FP', 'FN', 'precision', 'recall',
                     'mean err (m)', 'median err (m)'], rows))

    # Plot: precision and recall per class, bars = mean over runs, dots = runs.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    width = 0.38
    for j, (metric, color) in enumerate((('precision', '#1f77b4'), ('recall', '#ff7f0e'))):
        for i, label in enumerate(labels):
            vals = [r[label][metric] for r in per_run.values()
                    if label in r and r[label][metric] is not None]
            x = i + (j - 0.5) * width
            if vals:
                mean, var, _ = mean_var(vals)
                ax.bar(x, mean, width, color=color, alpha=0.75,
                       yerr=math.sqrt(var) if var else 0, capsize=3,
                       label=metric if i == 0 else None)
                ax.scatter([x] * len(vals), vals, color='k', s=10, zorder=3)
            else:
                ax.text(x, 0.02, 'n/a', ha='center', fontsize=7, rotation=90)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('fraction')
    ax.set_title(f'Semantic map precision and recall per class ({len(per_run)} runs; '
                 'bar = mean, whisker = sd, dots = runs)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'semantic_map_precision_recall.png'), dpi=130)
    plt.close(fig)

    # Plot: position error of every match, per class, coloured by run.
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for n, (k, r) in enumerate(per_run.items()):
        for i, label in enumerate(labels):
            errs = r[label]['errors'] if label in r else []
            ax.scatter([i + (n - 1) * 0.15] * len(errs), errs, color=COLORS[n % 5], s=18,
                       label=f'run {k}' if i == 0 else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel('position error (m)')
    ax.set_title('Position error of every matched object (map position to object centre)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'semantic_map_position_error.png'), dpi=130)
    plt.close(fig)
    return per_run


# ------------------------------------------------------------------ navigation
def navigation_section(runs, md, reach=1.0):
    data = {k: read_csv(os.path.join(RAW, f'nav_run{k}.csv')) for k in runs
            if os.path.exists(os.path.join(RAW, f'nav_run{k}.csv'))}
    if not data:
        return
    md.append('\n## Navigation benchmark (GoTo)\n')
    md.append(f'{len(next(iter(data.values())))} GoTo commands per run, random labels '
              '(seed = run number), chained from wherever the previous one ended. '
              f'**Success** = the commander reported arrival AND the robot\'s true final '
              f'position is within {reach} m of the footprint of a real object of that '
              'class. Time-to-goal and distance are over successful commands.\n')
    rows, per_run = [], {}
    for k, rs in data.items():
        n = len(rs)
        ok = [r for r in rs if r['outcome'] == 'reached']
        claimed = [r for r in rs if r['success'] == '1']
        s = {'success': len(ok) / n, 'claimed': len(claimed) / n,
             'time': statistics.mean(float(r['time_s']) for r in ok) if ok else None,
             'dist': statistics.mean(float(r['dist_m']) for r in ok) if ok else None,
             'centre': statistics.mean(float(r['centre_m']) for r in ok) if ok else None,
             'bearing': statistics.mean(float(r['bearing_deg']) for r in ok) if ok else None}
        per_run[k] = s
        counts = {o: sum(r['outcome'] == o for r in rs) for o in OUTCOMES}
        rows.append([f'run {k}', f'{s["success"]:.0%}', f'{s["claimed"]:.0%}',
                     f(s['time'], '.1f'), f(s['dist'], '.2f'), f(s['centre'], '.2f'),
                     f(s['bearing'], '.0f')] + [counts[o] for o in OUTCOMES])
    vals = list(per_run.values())
    for name, sd in (('mean ± sd', True), ('variance', False)):
        rows.append([f'**{name}**', mv([v['success'] for v in vals], '.2f', sd),
                     mv([v['claimed'] for v in vals], '.2f', sd),
                     mv([v['time'] for v in vals], '.1f', sd),
                     mv([v['dist'] for v in vals], '.2f', sd),
                     mv([v['centre'] for v in vals], '.2f', sd),
                     mv([v['bearing'] for v in vals], '.0f', sd)] + [''] * len(OUTCOMES))
    md.append(table(['run', 'success', 'commander said OK', 'time-to-goal (sim s)',
                     'final dist to footprint (m)', 'to centre (m)', 'heading err (deg)'] +
                    list(OUTCOMES), rows))

    allrows = [r for rs in data.values() for r in rs]
    labels = sorted({r['label'] for r in allrows})
    rows = []
    for label in labels:
        rs = [r for r in allrows if r['label'] == label]
        ok = [r for r in rs if r['outcome'] == 'reached']
        common = statistics.mode([r['outcome'] for r in rs if r['outcome'] != 'reached']) \
            if len(ok) < len(rs) else ''
        rows.append([label, len(rs), f'{len(ok)}/{len(rs)}',
                     f(statistics.mean(float(r['time_s']) for r in ok), '.1f') if ok else '–',
                     f(statistics.mean(float(r['dist_m']) for r in ok), '.2f') if ok else '–',
                     common])
    md.append(f'\nPer label (pooled over {len(data)} runs):\n')
    md.append(table(['label', 'commands', 'success', 'mean time (sim s)',
                     'mean final dist (m)', 'most common failure'], rows))

    # Plot: outcomes per run (stacked).
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1, 2]})
    ax = axes[0]
    bottoms = [0] * len(data)
    for o in OUTCOMES:
        counts = [sum(r['outcome'] == o for r in rs) for rs in data.values()]
        if any(counts):
            ax.bar([f'run {k}' for k in data], counts, bottom=bottoms, label=o,
                   color=OUTCOME_COLORS[o])
            bottoms = [b + c for b, c in zip(bottoms, counts)]
    ax.set_ylabel('GoTo commands')
    ax.set_title('Outcome of every command')
    ax.legend(fontsize=7, loc='lower left')
    ax = axes[1]
    for i, label in enumerate(labels):
        rs = [r for r in allrows if r['label'] == label]
        ax.bar(i, sum(r['outcome'] == 'reached' for r in rs) / len(rs), color='#2ca02c',
               alpha=0.8)
        ax.text(i, 0.02, f'n={len(rs)}', ha='center', fontsize=7, color='w')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('success rate')
    ax.set_title('Success rate per label (all runs pooled)')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'navigation_outcomes.png'), dpi=130)
    plt.close(fig)

    # Plot: time-to-goal and final distance of every command that the commander claimed.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for n, (k, rs) in enumerate(data.items()):
        claimed = [r for r in rs if r['success'] == '1']
        axes[0].scatter([float(r['path_m']) for r in claimed],
                        [float(r['time_s']) for r in claimed], color=COLORS[n % 5], s=18,
                        label=f'run {k}')
        axes[1].scatter([labels.index(r['label']) + (n - 1) * 0.15 for r in claimed],
                        [float(r['dist_m']) for r in claimed], color=COLORS[n % 5], s=18,
                        label=f'run {k}')
    axes[0].set_xlabel('distance driven (m)')
    axes[0].set_ylabel('time-to-goal (sim s)')
    axes[0].set_title('Time vs distance driven')
    axes[0].legend(fontsize=8)
    axes[1].axhline(reach, color='r', ls='--', lw=1, label=f'success threshold {reach} m')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel('final distance to footprint (m)')
    axes[1].set_title('Where the robot stopped (commander said "arrived")')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'navigation_time_distance.png'), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ exploration
def exploration_section(runs, md):
    data = {k: read_csv(os.path.join(RAW, f'explore_run{k}.csv')) for k in runs
            if os.path.exists(os.path.join(RAW, f'explore_run{k}.csv'))}
    if not data:
        return
    md.append('\n## Exploration (input to the semantic map)\n')
    rows, finals = [], []
    for k, rs in data.items():
        last = rs[-1]
        finals.append({c: float(last[c]) for c in ('sim_s', 'wall_s', 'coverage', 'path_m',
                                                   'objects', 'matched')})
        fl = finals[-1]
        rows.append([f'run {k}', f'{fl["sim_s"]:.0f}', f'{fl["wall_s"]:.0f}',
                     f'{fl["coverage"]:.1%}', f'{fl["path_m"]:.1f}', f'{fl["objects"]:.0f}',
                     f'{fl["matched"]:.0f}/18'])
    for name, sd in (('mean ± sd', True), ('variance', False)):
        rows.append([f'**{name}**'] + [mv([fl[c] for fl in finals], s, sd) for c, s in
                                       (('sim_s', '.0f'), ('wall_s', '.0f'),
                                        ('coverage', '.3f'), ('path_m', '.1f'),
                                        ('objects', '.1f'), ('matched', '.1f'))])
    md.append(table(['run', 'time (sim s)', 'time (wall s)', 'map coverage', 'driven (m)',
                     'map objects', 'matched'], rows))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for n, (k, rs) in enumerate(data.items()):
        t = [float(r['sim_s']) for r in rs]
        axes[0].plot(t, [float(r['coverage']) for r in rs], color=COLORS[n % 5],
                     label=f'run {k}')
        axes[1].plot(t, [float(r['matched']) for r in rs], color=COLORS[n % 5],
                     label=f'run {k} matched')
        axes[1].plot(t, [float(r['objects']) for r in rs], color=COLORS[n % 5], ls=':',
                     label=f'run {k} map objects')
    axes[0].set_ylabel('map coverage')
    axes[1].axhline(18, color='k', lw=0.8, ls='--')
    axes[1].set_ylabel('objects')
    for ax in axes:
        ax.set_xlabel('sim time (s)')
        ax.grid(alpha=0.3)
    axes[0].set_title('Occupancy map coverage')
    axes[1].set_title('Semantic map: matched (solid) / all confirmed (dotted)')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, 'exploration_progress.png'), dpi=130)
    plt.close(fig)


def main():
    runs = runs_available()
    if not runs:
        raise SystemExit(f'no results in {RAW}; run tools/run_benchmarks.sh first')
    md = [f'# Results ({len(runs)} runs)\n',
          'Generated by `tools/make_results.py` from `results/raw/` '
          '(`tools/run_benchmarks.sh`). Values are "mean ± sd" over runs; the '
          'variance row is the sample variance (n - 1).\n']
    exploration_section(runs, md)
    semantic_map_section(runs, md)
    navigation_section(runs, md)
    with open(os.path.join(RESULTS, 'RESULTS.md'), 'w') as fh:
        fh.write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
