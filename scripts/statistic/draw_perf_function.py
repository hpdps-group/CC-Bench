#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Draw perf_function CSV data as a heatmap.

Rows are ordered by rank then by operation type.
X-axis is time, color shows the selected metric.

Usage:
    python2.7 scripts/draw_perf_function.py
    python2.7 scripts/draw_perf_function.py --metric duration
    python2.7 scripts/draw_perf_function.py --metric compression_ratio
    python2.7 scripts/draw_perf_function.py --metric input_bytes --save heat.png
    python2.7 scripts/draw_perf_function.py --summary
"""

from __future__ import print_function
import os
import re
import sys
import csv
import argparse
from collections import OrderedDict

import numpy as np


# -- filename parsing ---------------------------------------------------------

FUNC_RE = re.compile(r'perf_function_(\d+)\.csv$')


def parse_func_filename(fname):
    m = FUNC_RE.search(fname)
    if not m:
        return None
    return int(m.group(1))


# -- CSV reading --------------------------------------------------------------

def read_func_csv(filepath):
    """Read a perf_function CSV, return list of dicts (one per row)."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        rows = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                k = k.strip() if k else k
                v = v.strip() if v else v
                clean[k] = v
            rows.append(clean)
        return rows


# -- metrics helper -----------------------------------------------------------

def compute_stats(values):
    a = np.array(values, dtype=float)
    return np.mean(a), np.min(a), np.max(a), np.std(a)


# -- heatmap drawing ----------------------------------------------------------

def draw_heatmap(all_data, metric, output_path=None, bounds=None):
    """Draw a heatmap: rows = rank|func_name, columns = time bins.

    all_data: dict of rank -> list of (func_name, timestamp, value)
    """
    import matplotlib
    if output_path:
        matplotlib.use('Agg')
    else:
        try:
            matplotlib.use('GTKAgg')
        except:
            pass
    import matplotlib.pyplot as plt

    # Build row list: sorted by rank, then by operation
    OPERATION_ORDER = ['ZCCL_compress_mt', 'ZCCL_decompress_mt',
                       'MPI_Allreduce', 'MPI_Allgather']
    ranks = sorted(all_data.keys())
    row_labels = []
    row_data = []  # each entry = list of (timestamp, value)
    all_ts = set()

    for r in ranks:
        # Group by func_name
        by_func = OrderedDict()
        for fn in OPERATION_ORDER:
            by_func[fn] = []
        for ts, val, fn in all_data[r]:
            if fn in by_func:
                by_func[fn].append((ts, val))
                all_ts.add(ts)

        for fn in OPERATION_ORDER:
            pts = by_func[fn]
            if pts:
                row_labels.append('r{}|{}'.format(r, fn))
                row_data.append(sorted(pts, key=lambda x: x[0]))

    if not all_ts:
        print('No timestamp data', file=sys.stderr)
        return

    # Build unified time axis (sorted unique timestamps)
    time_axis = np.array(sorted(all_ts))
    t0 = time_axis[0]
    time_rel = time_axis - t0

    n_rows = len(row_data)
    n_cols = len(time_axis)
    if n_rows == 0 or n_cols == 0:
        print('No data to plot', file=sys.stderr)
        return

    # Downsample if too many columns
    max_cols = 4000
    if n_cols > max_cols:
        step = n_cols // max_cols
        keep = np.arange(0, n_cols, step)
        time_axis = time_axis[keep]
        time_rel = time_rel[keep]
        n_cols = len(keep)
        print('Downsampled {} -> {} time columns'.format(n_cols * step, n_cols),
              file=sys.stderr)

    # Build matrix: forward-fill values onto time axis
    matrix = np.empty((n_rows, n_cols), dtype=float)
    matrix[:] = np.nan
    for i, pts in enumerate(row_data):
        idx = 0
        for j in range(n_cols):
            t = time_axis[j]
            while idx < len(pts) - 1 and pts[idx + 1][0] <= t:
                idx += 1
            if idx < len(pts):
                matrix[i, j] = pts[idx][1]

    # Filter all-NaN rows
    valid_rows = ~np.all(np.isnan(matrix), axis=1)
    if not np.all(valid_rows):
        matrix = matrix[valid_rows, :]
        row_labels = [l for l, v in zip(row_labels, valid_rows) if v]
        n_rows = len(row_labels)
        print('Removed {} empty row(s)'.format(int(np.sum(~valid_rows))),
              file=sys.stderr)

    if n_rows == 0:
        print('No data after filtering', file=sys.stderr)
        return

    # Normalize to [0, 1]
    if bounds:
        vmin, vmax = bounds
        if vmax > vmin:
            normed = np.clip((matrix - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            normed = np.zeros_like(matrix)
        print('Using global bounds [{}, {}] for "{}"'.format(vmin, vmax, metric),
              file=sys.stderr)
    else:
        # Per-row min-max normalization (fallback)
        normed = np.zeros_like(matrix)
        for i in range(n_rows):
            row = matrix[i, :]
            valid = row[~np.isnan(row)]
            if len(valid) < 1:
                continue
            rmin, rmax = float(np.min(valid)), float(np.max(valid))
            if rmax > rmin:
                normed[i, :] = (row - rmin) / (rmax - rmin)
            else:
                normed[i, :] = 0.0

    # Plot
    fig_height = max(6, n_rows * 0.25)
    fig_width = max(10, n_cols * 0.018)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.hot
    im = ax.imshow(normed, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0, vmax=1)

    # Y-axis labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_ylabel('Rank|Operation', fontsize=8)

    # X-axis
    n_xticks = min(30, n_cols)
    tick_step = max(1, n_cols // n_xticks)
    xticks = range(0, n_cols, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:.3f}s'.format(time_rel[i]) for i in xticks],
                       fontsize=6, rotation=45)
    ax.set_xlabel('Time (seconds from first call)', fontsize=8)

    # Separators between ranks
    sep_color = '#1f77b4'
    prev = None
    for i, label in enumerate(row_labels):
        r = label.split('|')[0]
        if prev is not None and r != prev:
            ax.axhline(y=i - 0.5, color=sep_color, linewidth=1.0, linestyle='-')
        prev = r

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    if bounds:
        cbar_vmin, cbar_vmax = bounds
        cbar.set_label('{} (clamped to [{}, {}])'.format(metric, cbar_vmin, cbar_vmax),
                       fontsize=7)
    else:
        cbar.set_label(metric + ' (per-row normalized)', fontsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(6)

    if bounds:
        bmin, bmax = bounds
        ax.set_title('Per-rank {} (global bounds [{}, {}])'.format(metric, bmin, bmax),
                     fontsize=9)
    else:
        ax.set_title('Per-rank {} (each row normalized to [0,1])'.format(metric),
                     fontsize=9)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.grid(False)

    plt.subplots_adjust(left=0.15, right=0.88)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Saved to {}'.format(output_path), file=sys.stderr)
    plt.close(fig)


# -- summary mode -------------------------------------------------------------

def draw_summary(all_raw, output_path=None):
    """Bar chart of mean +/- std per rank for each metric."""
    import matplotlib
    if output_path:
        matplotlib.use('Agg')
    else:
        try:
            matplotlib.use('GTKAgg')
        except:
            pass
    import matplotlib.pyplot as plt

    metrics = ['duration', 'compression_ratio', 'input_bytes', 'output_bytes']
    ranks = sorted(all_raw.keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2.5 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i]
        means, mins, maxs, stds = [], [], [], []
        for r in ranks:
            vals = []
            for ts, val, fn in all_raw[r]:
                if val is not None:
                    vals.append(val)
            if vals:
                m, lo, hi, s = compute_stats(vals)
                means.append(m)
                mins.append(lo)
                maxs.append(hi)
                stds.append(s)
            else:
                means.append(0)
                mins.append(0)
                maxs.append(0)
                stds.append(0)

        x = range(len(ranks))
        ax.errorbar(x, means, yerr=stds, fmt='o-', capsize=3,
                    markersize=3, linewidth=0.8, color='#1f77b4')
        ax.fill_between(x, mins, maxs, alpha=0.15, color='#1f77b4')
        ax.set_xticks(x)
        ax.set_xticklabels(['r{}'.format(r) for r in ranks],
                           fontsize=6, rotation=45)
        ax.set_ylabel(metric, fontsize=8)
        ax.set_title('{} per rank (mean +- std, shaded = min~max)'.format(metric),
                     fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Saved to {}'.format(output_path), file=sys.stderr)
    plt.close(fig)


# -- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Draw perf_function heatmap.')
    parser.add_argument('--dir', default='perf_files',
                        help='Directory containing perf_function_*.csv')
    parser.add_argument('--metric', default='duration',
                        help='Metric to plot (duration, compression_ratio, '
                             'input_bytes, output_bytes, nbEle)')
    parser.add_argument('--save', default=None,
                        help='Save plot to file (default: perf_function_<metric>.png)')
    parser.add_argument('--summary', action='store_true',
                        help='Show per-rank statistics instead of heatmap')
    parser.add_argument('--bounds', action='append', default=None,
                        help='Explicit value range for the metric, repeatable. '
                             'Format: keyword=vmin,vmax (e.g. "duration=0,500"). '
                             'Only the first match for the selected metric is used. '
                             'Values outside [vmin, vmax] are clamped. '
                             '(default: per-row min-max normalization)')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print('Error: directory not found: {}'.format(args.dir), file=sys.stderr)
        sys.exit(1)

    # Discover files
    func_files = []
    for fname in os.listdir(args.dir):
        rank = parse_func_filename(fname)
        if rank is not None:
            func_files.append((rank, fname))

    if not func_files:
        print('No perf_function_*.csv files found in {}'.format(args.dir),
              file=sys.stderr)
        sys.exit(1)

    func_files.sort(key=lambda x: x[0])
    print('Found {} rank file(s)'.format(len(func_files)), file=sys.stderr)

    # Read all data
    metric = args.metric
    all_raw = {}   # rank -> list of (timestamp_us, metric_value, func_name)

    for rank, fname in func_files:
        fpath = os.path.join(args.dir, fname)
        rows = read_func_csv(fpath)
        if not rows:
            continue
        entries = []
        for row in rows:
            fn = row.get('func_name', '').strip()
            ts_str = row.get('timestamp_us', '').strip()
            val_str = row.get(metric, '').strip()
            if fn and ts_str and val_str:
                try:
                    ts = float(ts_str)
                    val = float(val_str)
                    entries.append((ts, val, fn))
                except ValueError:
                    continue
        if entries:
            all_raw[rank] = entries
            print('  rank {}: {} rows'.format(rank, len(entries)),
                  file=sys.stderr)

    if not all_raw:
        print('No data read.', file=sys.stderr)
        sys.exit(1)

    if args.summary:
        out = args.save if args.save else 'perf_function_summary.png'
        if not out.endswith('.png'):
            base, _ = os.path.splitext(out)
            out = base + '_summary.png'
        draw_summary(all_raw, output_path=out)
    else:
        out = args.save if args.save else 'perf_function_{}.png'.format(metric)

        # Parse --bounds for the selected metric
        bounds = None
        if args.bounds:
            for b in args.bounds:
                try:
                    key, rest = b.split('=', 1)
                    vmin_str, vmax_str = rest.split(',', 1)
                    if key.strip().lower() in metric.lower():
                        bounds = (float(vmin_str.strip()), float(vmax_str.strip()))
                        print('Bounds matched: {} -> [{}, {}] for "{}"'.format(
                            key.strip(), bounds[0], bounds[1], metric), file=sys.stderr)
                        break
                except ValueError:
                    print('Warning: ignoring malformed --bounds "{}"'.format(b),
                          file=sys.stderr)

        draw_heatmap(all_raw, metric, output_path=out, bounds=bounds)


if __name__ == '__main__':
    main()
