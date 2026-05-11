#!/usr/bin/env python2.7
"""
Draw hardware utilization timeline from perf_files data.

Usage:
    python2.7 scripts/draw_hardware.py
    python2.7 scripts/draw_hardware.py --dir perf_files --save timeline.png
    python2.7 scripts/draw_hardware.py --dir perf_files --from-ts 45238589 --to-ts 45238610
    python2.7 scripts/draw_hardware.py --dir perf_files --metric "cpu_util, rcv_bw_Bps, counter_1"
    python2.7 scripts/draw_hardware.py --dir perf_files --abstract --save compact.png
    python2.7 scripts/draw_hardware.py --dir perf_files --node 191 --save node191.png

Each device independently picks its first non-timestamp column as its metric
(or the first match from the --metric list if given) and normalizes to [0,1].

Scale detection (auto):
    Column name contains "util", "pct", "percent", "%" -> percent (0-100 -> 0-1)
    Otherwise -> per-device min-max normalization
"""

from __future__ import print_function
import os
import re
import sys
import csv
import argparse
import itertools
from collections import OrderedDict, defaultdict

import numpy as np


# -- filename parsing ---------------------------------------------------------

PERF_RE = re.compile(r'perf_(.*)_node(\d+)\.csv$')


def parse_perf_filename(fname):
    m = PERF_RE.search(fname)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def classify_device(devname):
    if devname == 'cpu':
        return 'cpu_agg'
    if devname.startswith('cpu'):
        return 'cpu'
    if devname.startswith('ib'):
        return 'ib'
    if devname.startswith('pcie'):
        return 'pcie'
    return 'other'


def cpu_number(devname):
    try:
        return int(devname[3:])
    except (IndexError, ValueError):
        return -1


# -- bounds parsing -----------------------------------------------------------

def parse_bounds(bounds_args):
    """Parse --bounds arguments into a dict of keyword -> (vmin, vmax).

    Input: ['cpu_util=0,100', 'rcv_bw_Bps=0,5000000000']
    Output: {'cpu_util': (0.0, 100.0), 'rcv_bw_Bps': (0.0, 5000000000.0)}
    """
    result = {}
    if not bounds_args:
        return result
    for arg in bounds_args:
        try:
            key, rest = arg.split('=', 1)
            vmin_str, vmax_str = rest.split(',', 1)
            result[key.strip()] = (float(vmin_str.strip()), float(vmax_str.strip()))
        except ValueError:
            print('Warning: ignoring malformed --bounds "{}" '
                  '(expected keyword=vmin,vmax)'.format(arg), file=sys.stderr)
    return result


def match_bounds(metric_name, bounds):
    """Find bounds for a metric name by case-insensitive keyword matching.

    Returns (vmin, vmax) or None.
    """
    if not bounds or not metric_name:
        return None
    name_lower = metric_name.lower()
    for keyword, (vmin, vmax) in bounds.items():
        if keyword.lower() in name_lower:
            return (vmin, vmax)
    return None


# -- metric / scale detection -------------------------------------------------

def auto_select_metric(headers):
    """Return the first non-timestamp column."""
    for h in headers:
        if h != 'timestamp_sec':
            return h
    return None


def pick_metric(headers, metric_col):
    """Resolve which metric column to use.

    metric_col can be:
      - None: auto-select first non-timestamp column
      - str:  use this exact column (must exist in headers)
      - list: try each in order, use first that exists
    Returns column name or None.
    """
    if metric_col is None:
        return auto_select_metric(headers)
    if isinstance(metric_col, list):
        for name in metric_col:
            name = name.strip()
            if name and name in headers:
                return name
        return auto_select_metric(headers)
    if metric_col in headers:
        return metric_col
    return auto_select_metric(headers)


def detect_scale(col_name, scale_override=None):
    if scale_override:
        return scale_override
    name = col_name.lower()
    if 'util' in name or 'pct' in name or 'percent' in name or '%' in name:
        return 'percent'
    return 'minmax'


def normalize_values(values, scale, bounds=None):
    if bounds is not None:
        vmin, vmax = bounds
        if vmax > vmin:
            return np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.zeros_like(values)
    valid = values[~np.isnan(values)]
    if len(valid) < 1:
        return np.zeros_like(values)
    if scale == 'percent':
        return np.clip(values / 100.0, 0.0, 1.0)
    vmin, vmax = float(np.min(valid)), float(np.max(valid))
    if vmax > vmin:
        return (values - vmin) / (vmax - vmin)
    return np.zeros_like(values)


# -- CSV reading --------------------------------------------------------------

def read_perf_csv(filepath, metric_col=None):
    """Read a perf CSV.

    Returns (timestamps, values, metric_column_name) or (None, None, None).
    metric_col can be None, str, or list -- see pick_metric().
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, None, None
        headers = reader.fieldnames
        chosen = pick_metric(headers, metric_col)
        if chosen is None:
            return None, None, None
        rows = []
        for row in reader:
            try:
                ts = float(row['timestamp_sec'])
                val = float(row[chosen])
            except (ValueError, KeyError):
                continue
            rows.append((ts, val))
    if not rows:
        return None, None, None
    timestamps, values = zip(*rows)
    return np.array(timestamps), np.array(values, dtype=float), chosen


# -- map.csv reading ----------------------------------------------------------

def read_map_csv(filepath):
    mapping = {}
    node_ranks = defaultdict(list)
    if not os.path.isfile(filepath):
        return mapping, node_ranks
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rank = int(row['rank'].strip())
                node_str = row['node'].strip()
                node_id = int(node_str.replace('node', ''))
                dtype = row['device_type'].strip()
                dnum = int(row['device_number'].strip())
            except (ValueError, KeyError):
                continue
            devname = '{}{}'.format(dtype, dnum)
            mapping[(node_id, devname)] = rank
            node_ranks[node_id].append((rank, devname))
    for nid in node_ranks:
        node_ranks[nid].sort(key=lambda x: x[0])
    return mapping, node_ranks


# -- device ordering ----------------------------------------------------------

def order_devices(grouped, mapping, node_ranks):
    node_ids = sorted(grouped.keys())
    ordered = []
    for nid in node_ids:
        devs = grouped[nid]
        mapped_devs = set()
        mapping_order = {}
        if nid in node_ranks:
            for rank, dname in node_ranks[nid]:
                mapped_devs.add(dname)
                mapping_order[dname] = rank

        mapped = [d for d in devs if d in mapped_devs]
        mapped.sort(key=lambda d: mapping_order[d])
        for d in mapped:
            ordered.append((nid, d))

        unmapped = [d for d in devs if d not in mapped_devs]

        def sort_key(d):
            cls = classify_device(d)
            if cls == 'cpu_agg':
                return (0, 0, d)
            elif cls == 'cpu':
                return (1, cpu_number(d), d)
            elif cls == 'ib':
                return (2, 0, d)
            elif cls == 'pcie':
                return (3, 0, d)
            else:
                return (4, 0, d)

        unmapped.sort(key=sort_key)
        for d in unmapped:
            ordered.append((nid, d))

    return ordered


# -- data loading -------------------------------------------------------------

def load_perf_data(perf_dir, metric_col=None, scale_override=None,
                   time_start=None, time_end=None, bounds=None):
    """Read all perf CSVs and return structured data.

    Returns: (ordered_rows, time_rel, data_matrix, raw_matrix, row_meta)
    """
    # Pass 1: discover all devices from filenames
    all_devices = []
    for fname in os.listdir(perf_dir):
        if not fname.endswith('.csv') or fname == 'map.csv':
            continue
        parsed = parse_perf_filename(fname)
        if not parsed:
            continue
        devname, node_id = parsed
        all_devices.append((node_id, devname, fname))

    if not all_devices:
        return None, None, None, None, None

    device_set = OrderedDict()
    for nid, dname, _ in all_devices:
        device_set[(nid, dname)] = classify_device(dname)

    grouped = defaultdict(list)
    for (nid, dname) in device_set:
        grouped[nid].append(dname)

    map_path = os.path.join(perf_dir, 'map.csv')
    mapping, node_ranks = read_map_csv(map_path)
    ordered = order_devices(grouped, mapping, node_ranks)

    # Pass 2: read each file
    all_data = {}
    per_device_metric = {}
    for nid, dname, fname in all_devices:
        fpath = os.path.join(perf_dir, fname)
        ts, vals, mcol = read_perf_csv(fpath, metric_col)
        if ts is None or len(ts) < 2:
            continue
        all_data[(nid, dname)] = (ts, vals)
        per_device_metric[(nid, dname)] = mcol

    if not all_data:
        return None, None, None, None, None

    # Build unified time axis
    all_ts = set()
    for (ts, vals) in all_data.values():
        for t in ts:
            all_ts.add(t)
    if not all_ts:
        return None, None, None, None, None

    time_axis = np.array(sorted(all_ts))
    t0 = time_axis[0]
    time_rel = time_axis - t0

    # Time range filter
    if time_start is not None or time_end is not None:
        mask = np.ones(len(time_rel), dtype=bool)
        if time_start is not None:
            mask &= (time_rel + t0 >= time_start)
        if time_end is not None:
            mask &= (time_rel + t0 <= time_end)
        if np.any(mask):
            time_axis = time_axis[mask]
            time_rel = time_rel[mask]

    n_devices = len(ordered)
    n_times = len(time_axis)
    if n_times == 0:
        return None, None, None, None, None

    data_matrix = np.zeros((n_devices, n_times), dtype=float)
    raw_matrix = np.zeros((n_devices, n_times), dtype=float)
    row_meta = []

    for row_idx, (nid, dname) in enumerate(ordered):
        entry = all_data.get((nid, dname))
        if entry is None:
            rank = mapping.get((nid, dname), None)
            is_mapped = (nid, dname) in mapping
            data_matrix[row_idx, :] = 0.0
            raw_matrix[row_idx, :] = np.nan
            row_meta.append({'node': nid, 'devname': dname,
                             'rank': rank, 'mapped': is_mapped,
                             'metric': None, 'scale': 'minmax'})
            continue

        ts, vals = entry
        mcol = per_device_metric.get((nid, dname), '?')
        scale = detect_scale(mcol, scale_override)
        dev_bounds = match_bounds(mcol, bounds)

        aligned = np.ones(n_times) * np.nan
        j = 0
        for i, t in enumerate(time_axis):
            while j < len(ts) - 1 and ts[j + 1] <= t:
                j += 1
            if j < len(ts):
                aligned[i] = vals[j]

        raw_matrix[row_idx, :] = aligned
        data_matrix[row_idx, :] = normalize_values(aligned, scale, dev_bounds)

        rank = mapping.get((nid, dname), None)
        is_mapped = (nid, dname) in mapping
        row_meta.append({
            'node': nid,
            'devname': dname,
            'rank': rank,
            'mapped': is_mapped,
            'metric': mcol,
            'scale': scale,
            'bounds': dev_bounds,
        })

    # Downsample if too many time points
    max_samples = 2000
    if n_times > max_samples:
        block = n_times // max_samples
        if block >= 2:
            n_bins = n_times // block
            down_idx = np.arange(0, n_bins * block, block)
            time_axis_down = np.array([np.mean(time_axis[i:i+block]) for i in down_idx])
            time_rel = time_axis_down - time_axis_down[0]
            data_matrix = data_matrix[:, down_idx]
            raw_matrix = raw_matrix[:, down_idx]
            print('Downsampled {} -> {} time points'.format(n_times, n_bins), file=sys.stderr)

    return ordered, time_rel, data_matrix, raw_matrix, row_meta


# -- plotting -----------------------------------------------------------------

def draw_heatmap(time_rel, data_matrix, raw_matrix, ordered, row_meta,
                 output_path=None, title=None):
    n_devices, n_times = data_matrix.shape
    if n_devices == 0 or n_times == 0:
        print('No data to plot.', file=sys.stderr)
        return

    import matplotlib
    if output_path:
        matplotlib.use('Agg')
    else:
        try:
            matplotlib.use('GTKAgg')
        except:
            pass

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig_height = max(6, n_devices * 0.35)
    fig_width = max(10, n_times * 0.02)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.hot
    im = ax.imshow(data_matrix, aspect='auto', cmap=cmap,
                   interpolation='nearest', vmin=0, vmax=1)

    # Y-axis labels
    labels = []
    for meta in row_meta:
        label = 'node{}|{}'.format(meta['node'], meta['devname'])
        if meta['rank'] is not None:
            label += ' (r{})'.format(meta['rank'])
        if meta['metric'] is not None:
            label = label + ' [{}]'.format(meta['metric'])
        b = meta.get('bounds')
        if b is not None:
            label += ' {{{},{}}}'.format(b[0], b[1])
        labels.append(label)

    ax.set_yticks(range(n_devices))
    ax.set_yticklabels(labels, fontsize=6)
    for idx, meta in enumerate(row_meta):
        if meta['mapped']:
            ax.get_yticklabels()[idx].set_weight('bold')
            ax.get_yticklabels()[idx].set_color('black')
        else:
            ax.get_yticklabels()[idx].set_color('gray')
    ax.set_ylabel('Device', fontsize=8)

    # X-axis
    n_xticks = min(30, n_times)
    tick_step = max(1, n_times // n_xticks)
    xticks = range(0, n_times, tick_step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:.1f}s'.format(time_rel[i]) for i in xticks],
                       fontsize=6, rotation=45)
    ax.set_xlabel('Time (seconds from start)', fontsize=8)

    # Node separators
    sep_color = '#1f77b4'
    prev_node = None
    for idx, meta in enumerate(row_meta):
        if prev_node is not None and meta['node'] != prev_node:
            ax.axhline(y=idx - 0.5, color=sep_color, linewidth=1.5, linestyle='-')
        prev_node = meta['node']

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)

    # Detect if any row uses explicit bounds
    has_bounds = any(m.get('bounds') is not None for m in row_meta)
    if has_bounds:
        cbar.set_label('Normalized activity (clamped to bounds)', fontsize=7)
    else:
        cbar.set_label('Normalized activity (0=min, 1=max)', fontsize=7)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(6)

    if title:
        ax.set_title(title, fontsize=10)

    ax.set_ylim(n_devices - 0.5, -0.5)
    ax.grid(False)

    # Time-range slider
    if n_times > 50 and not output_path:
        plt.subplots_adjust(bottom=0.12, left=0.18, right=0.88)
        ax_slider = plt.axes([0.15, 0.02, 0.7, 0.03])
        time_slider = Slider(ax_slider, 'Window', 0, 1.0,
                             valinit=1.0, valfmt='%1.2f')

        def update_window(val):
            frac = time_slider.val
            half = int(n_times * (1.0 - frac) * 0.5)
            left = max(0, half)
            right = min(n_times - 1, n_times - 1 - half)
            if right > left:
                ax.set_xlim(left - 0.5, right - 0.5)
            fig.canvas.draw_idle()

        time_slider.on_changed(update_window)
        ax.set_xlim(-0.5, n_times - 0.5)
    else:
        plt.subplots_adjust(left=0.18, right=0.88)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print('Saved to {}'.format(output_path), file=sys.stderr)
        plt.close(fig)
    else:
        plt.show()


# -- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Draw hardware utilization timeline from perf_files data.')
    parser.add_argument('--dir', default='perf_files',
                        help='Directory containing perf_*.csv and map.csv')
    parser.add_argument('--metric', default=None,
                        help='Comma-separated preferred columns (e.g. "cpu_util, rcv_bw_Bps"). '
                             'Each device uses the first match from its own headers.')
    parser.add_argument('--scale', default=None,
                        choices=['percent', 'minmax'],
                        help='Scale override (default: auto per column name)')
    parser.add_argument('--from-ts', type=float, default=None,
                        help='Filter: start timestamp (absolute)')
    parser.add_argument('--to-ts', type=float, default=None,
                        help='Filter: end timestamp (absolute)')
    parser.add_argument('--save', default=None,
                        help='Save to file instead of interactive display')
    parser.add_argument('--title', default=None,
                        help='Plot title')
    parser.add_argument('--abstract', action='store_true',
                        help='Compact mode: skip devices that are all-NaN (no data) '
                             'or in map.csv but carry no actual measurement.')
    parser.add_argument('--node', type=int, default=None,
                        help='Only show devices from this node ID (e.g. 191).')
    parser.add_argument('--bounds', action='append', default=None,
                        help='Explicit value bounds for a metric, repeatable. '
                             'Format: keyword=vmin,vmax  (e.g. "cpu_util=0,100"). '
                             'Substring-matched case-insensitively against column names. '
                             'Values outside [vmin, vmax] are clamped. '
                             '(default: auto-detect percent or min-max per device)')
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print('Error: directory not found: {}'.format(args.dir), file=sys.stderr)
        sys.exit(1)

    # Parse --metric into list if it contains commas
    metric_col = args.metric
    if metric_col and ',' in metric_col:
        metric_col = [m.strip() for m in metric_col.split(',') if m.strip()]

    # Parse --bounds
    bounds_dict = parse_bounds(args.bounds)

    result = load_perf_data(args.dir, metric_col, args.scale,
                            args.from_ts, args.to_ts, bounds_dict)
    ordered, time_rel, data_matrix, raw_matrix, row_meta = result

    if time_rel is None:
        print('No valid perf data found in {}'.format(args.dir), file=sys.stderr)
        sys.exit(1)

    metrics_used = sorted(set(m['metric'] for m in row_meta if m['metric']))
    scales_used = sorted(set(m['scale'] for m in row_meta if m['scale']))
    bounds_used = sorted(set(m['bounds'] for m in row_meta if m.get('bounds') is not None))

    print('Devices: {}'.format(len(ordered)), file=sys.stderr)
    print('Timestamps: {}'.format(len(time_rel)), file=sys.stderr)
    print('Metrics:  {}'.format(', '.join(metrics_used) if metrics_used else 'N/A'), file=sys.stderr)
    print('Scales:   {}'.format(', '.join(scales_used) if scales_used else 'N/A'), file=sys.stderr)
    if bounds_used:
        bounds_str = ', '.join('[{},{}]'.format(b[0], b[1]) for b in bounds_used)
        print('Bounds:   {}'.format(bounds_str), file=sys.stderr)
    print('Time range: {:.1f}s - {:.1f}s ({} samples)'.format(
        time_rel[0], time_rel[-1], len(time_rel)), file=sys.stderr)

    # --abstract and --node filtering
    drop = np.zeros(len(ordered), dtype=bool)
    if args.abstract:
        all_nan = np.all(np.isnan(raw_matrix), axis=1)
        all_zero = np.all(data_matrix == 0, axis=1)
        mapped_no_data = np.array([m['mapped'] and m['metric'] is None for m in row_meta])
        drop = all_nan | all_zero | mapped_no_data
    if args.node is not None:
        node_mask = np.array([m['node'] == args.node for m in row_meta])
        drop = drop | ~node_mask
    keep_count = int(np.sum(~drop))
    if keep_count < len(ordered):
        ordered = list(itertools.compress(ordered, ~drop))
        data_matrix = data_matrix[~drop, :]
        raw_matrix = raw_matrix[~drop, :]
        row_meta = list(itertools.compress(row_meta, ~drop))
        print('Filtered: {} -> {} devices kept'.format(
            len(drop), keep_count), file=sys.stderr)

    draw_heatmap(time_rel, data_matrix, raw_matrix, ordered, row_meta,
                 output_path=args.save, title=args.title)


if __name__ == '__main__':
    main()
