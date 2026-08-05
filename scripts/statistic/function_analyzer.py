#!/usr/bin/env python
"""
function_analyzer.py -- Analyze perf_function trace data.

Classifies function entries into communication / compression categories,
estimates missing durations for async operations via pingpong benchmark
linear interpolation, and reports: total time, comm time, comp time,
overlap, and idle.
"""

from __future__ import print_function
import argparse
import csv
import os
import sys
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COMM_PATTERNS = [
    "MPI_Send", "MPI_Recv", "MPI_Isend", "MPI_Irecv",
    "MPI_Sendrecv",
    "MPI_Allreduce", "MPI_Reduce", "MPI_Bcast", "MPI_Allgather",
    "MPI_Gather", "MPI_Scatter", "MPI_Alltoall", "MPI_Reduce_scatter",
    "MPI_Barrier",
    "MPI_Wait", "MPI_Waitall", "MPI_Test",
    "PMPI_Send", "PMPI_Recv", "PMPI_Isend", "PMPI_Irecv",
    "PMPI_Allreduce", "PMPI_Allgather",
]

DEFAULT_COMP_PATTERNS = [
    "ZCCL_compress", "ZCCL_decompress",
    "SZ3_compress", "SZ3_decompress",
    "ZCCL_compression", "ZCCL_decompression",
    "compress", "decompress",
]

DEFAULT_PERF_DIR = "perf_files"
DEFAULT_PINGPONG = "results/pingpong.csv"

# Entries with recorded duration below this threshold (microseconds) are
# treated as async (missing) and will be estimated via interpolation.
ASYNC_DURATION_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Pingpong helpers
# ---------------------------------------------------------------------------

def load_pingpong(path):
    """Load pingpong CSV and average across all partners.

    Returns:
      pp_avg:  dict  (comm_type, msg_size) -> {"send_us": avg, "recv_us": avg}
      sizes_by_type: dict  comm_type -> sorted list of msg_sizes
    """
    sums = {}
    counts = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ct = row.get("comm_type", "").strip().lower()
                ms = int(row.get("msg_size", 0))
                send0 = float(row.get("send0_us", 0))
                recv0 = float(row.get("recv0_us", 0))
            except (ValueError, TypeError):
                continue
            if ct not in ("intra", "inter"):
                continue
            key = (ct, ms)
            if key not in sums:
                sums[key] = {"send_us": 0.0, "recv_us": 0.0}
                counts[key] = 0
            sums[key]["send_us"] += send0
            sums[key]["recv_us"] += recv0
            counts[key] += 1

    pp_avg = {}
    sizes_by_type = {}
    for (ct, ms), total in sums.items():
        n = counts[(ct, ms)]
        pp_avg[(ct, ms)] = {
            "send_us": total["send_us"] / n,
            "recv_us": total["recv_us"] / n,
        }
        sizes_by_type.setdefault(ct, []).append(ms)

    for ct in sizes_by_type:
        sizes_by_type[ct] = sorted(set(sizes_by_type[ct]))
    return pp_avg, sizes_by_type


def interpolate_oneway(pp_avg, sizes_by_type, msg_bytes, is_intra, func_name=""):
    """Estimate one-way latency (microseconds) via linear interpolation.

    Uses ``send_us`` if *func_name* contains "send" (case-insensitive),
    otherwise ``recv_us``.
    Averages are partner-averaged values from the pingpong CSV.
    """
    comm_type = "intra" if is_intra else "inter"
    sizes = sizes_by_type.get(comm_type, [])
    if not sizes:
        return 0.0

    # Pick send or recv column based on function name
    use_send = "send" in func_name.lower()

    def _oneway(ms):
        row = pp_avg.get((comm_type, ms))
        if row is None:
            return 0.0
        val = row["send_us"] if use_send else row["recv_us"]
        return val

    # Clamp / exact match
    if msg_bytes <= sizes[0]:
        return _oneway(sizes[0])
    if msg_bytes >= sizes[-1]:
        return _oneway(sizes[-1])
    if msg_bytes in sizes:
        return _oneway(msg_bytes)

    # Interpolate between bracketing sizes
    low = None
    high = None
    for s in sizes:
        if s <= msg_bytes:
            low = s
        if s >= msg_bytes and high is None:
            high = s

    if low is None or high is None:
        return 0.0

    t_low = _oneway(low)
    t_high = _oneway(high)
    if t_low <= 0 or t_high <= 0:
        return 0.0

    # T = T_low + (T_high - T_low) * (S - S_low) / (S_high - S_low)
    if high == low:
        return t_low
    t = t_low + (t_high - t_low) * float(msg_bytes - low) / float(high - low)
    return t


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def matches_any(name, patterns):
    """Case-insensitive substring match against a list of patterns."""
    name_lower = name.lower()
    for p in patterns:
        if p.lower() in name_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-rank analysis
# ---------------------------------------------------------------------------

def analyze_rank(rank, args, pp_data, sizes_by_type):
    """Load and classify perf_function entries for a given rank.

    Returns (all_entries, comm_entries, comp_entries, other_entries).
    Each entry is a dict with keys:
      name, timestamp, duration, msg_bytes, intra, estimated, category
    """
    pf_path = os.path.join(args.perf_dir, "perf_function_{}.csv".format(rank))
    if not os.path.isfile(pf_path):
        sys.stderr.write("[error] perf_function file not found: {}\n".format(pf_path))
        sys.exit(1)

    # Read CSV
    raw = []
    with open(pf_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw.append(row)

    sys.stderr.write("[info] rank {}: read {} entries from {}\n".format(rank, len(raw), pf_path))

    fn_field = args.func_name_field
    ts_field = args.timestamp_field
    dur_field = args.duration_field
    intra_field = args.intra_field
    msg_field = args.msg_bytes_field

    all_entries = []
    comm_entries = []
    comp_entries = []
    other_entries = []

    for row in raw:
        name = row.get(fn_field, "").strip()
        if not name or name == fn_field:
            continue

        # -- category --
        is_comm = matches_any(name, args.comm_funcs)
        is_comp = matches_any(name, args.comp_funcs)
        if is_comm and is_comp:
            is_comp = False  # comm takes priority

        # -- timestamp --
        try:
            ts = float(row.get(ts_field, 0))
        except (ValueError, TypeError):
            ts = 0.0

        # -- duration --
        dur_str = row.get(dur_field, "").strip()
        has_explicit_duration = bool(dur_str)
        try:
            duration = float(dur_str) * 1e6 if dur_str else 0.0
        except (ValueError, TypeError):
            duration = 0.0

        # -- msg bytes --
        msg_str = row.get(msg_field, "").strip()
        try:
            msg_bytes = int(float(msg_str)) if msg_str else 0
        except (ValueError, TypeError):
            msg_bytes = 0

        # -- intra flag --
        intra_str = row.get(intra_field, "").strip()
        is_intra = False
        if intra_str:
            try:
                is_intra = abs(float(intra_str) - 1.0) < 0.5
            except (ValueError, TypeError):
                is_intra = False

        # -- estimate duration for async comm ops --
        estimated = False
        need_estimate = (
            is_comm
            and (not has_explicit_duration or duration <= ASYNC_DURATION_THRESHOLD)
            and msg_bytes > 0
            and pp_data is not None
        )
        if need_estimate:
            est = interpolate_oneway(pp_data, sizes_by_type, msg_bytes,
                                     is_intra, name)
            if est > 0:
                duration = est
                estimated = True

        entry = {
            "name": name,
            "timestamp": ts,
            "duration": duration,
            "msg_bytes": msg_bytes,
            "intra": is_intra,
            "estimated": estimated,
            "category": "comm" if is_comm else ("comp" if is_comp else "other"),
        }
        all_entries.append(entry)
        if is_comm:
            comm_entries.append(entry)
        elif is_comp:
            comp_entries.append(entry)
        else:
            other_entries.append(entry)

    return all_entries, comm_entries, comp_entries, other_entries


# ---------------------------------------------------------------------------
# Timeline overlap computation (event-sweep)
# ---------------------------------------------------------------------------

def compute_timeline_stats(all_entries, comm_entries, comp_entries):
    """Compute comm_total, comp_total, overlap, idle via event sweep.

    Returns (total_wall, comm_total, comp_total, overlap, idle) where each
    value is in microseconds.
    """
    if not all_entries:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Wall-clock range
    timestamps = [e["timestamp"] for e in all_entries]
    t_min = min(timestamps)
    t_max = max(timestamps)
    for e in all_entries:
        end = e["timestamp"] + e["duration"]
        if end > t_max:
            t_max = end
    total_wall = t_max - t_min

    if total_wall <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Build events: (time, delta_comm, delta_comp)
    events = []
    for e in comm_entries:
        t0 = e["timestamp"]
        t1 = t0 + e["duration"]
        events.append((t0, 1, 0))
        events.append((t1, -1, 0))
    for e in comp_entries:
        t0 = e["timestamp"]
        t1 = t0 + e["duration"]
        events.append((t0, 0, 1))
        events.append((t1, 0, -1))

    if not events:
        return total_wall, 0.0, 0.0, 0.0, total_wall

    events.sort(key=lambda x: x[0])

    comm_active = 0
    comp_active = 0
    comm_only_time = 0.0
    comp_only_time = 0.0
    both_time = 0.0
    idle_time = 0.0

    prev_t = events[0][0]
    if prev_t > t_min:
        idle_time += prev_t - t_min

    for t, dc, dac in events:
        dt = t - prev_t
        if dt > 0:
            if comm_active > 0 and comp_active == 0:
                comm_only_time += dt
            elif comp_active > 0 and comm_active == 0:
                comp_only_time += dt
            elif comm_active > 0 and comp_active > 0:
                both_time += dt
            else:
                idle_time += dt
        comm_active += dc
        comp_active += dac
        prev_t = t

    # Remaining idle after last event
    if total_wall > (prev_t - t_min):
        idle_time += total_wall - (prev_t - t_min)

    comm_total = comm_only_time + both_time
    comp_total = comp_only_time + both_time
    overlap = both_time
    idle = idle_time

    return total_wall, comm_total, comp_total, overlap, idle


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(rank, total_wall, comm_total, comp_total, overlap, idle,
                 all_entries, comm_entries, comp_entries, other_entries):
    """Print formatted analysis report to stdout."""
    num_estimated = sum(1 for e in comm_entries if e["estimated"])

    print("")
    print("=" * 65)
    print("  Function Analysis Report  --  rank {}".format(rank))
    print("=" * 65)
    print("")
    print("  Entries:")
    print("    Total:                  {:>9d}".format(len(all_entries)))
    print("    Communication:          {:>9d}".format(len(comm_entries)))
    print("    Compression:            {:>9d}".format(len(comp_entries)))
    print("    Other / Uncategorized:  {:>9d}".format(len(other_entries)))
    if num_estimated:
        print("    (async durations estimated: {:>7d})".format(num_estimated))
    print("")
    print("  Timeline (us):")
    print("    Total wall-clock:       {:>15.3f}".format(total_wall))
    print("    Communication time:     {:>15.3f}  ({:>6.2f}%)".format(
        comm_total, (comm_total / total_wall * 100) if total_wall > 0 else 0))
    print("    Compression time:       {:>15.3f}  ({:>6.2f}%)".format(
        comp_total, (comp_total / total_wall * 100) if total_wall > 0 else 0))
    print("    Overlap (comm+comp):    {:>15.3f}  ({:>6.2f}%)".format(
        overlap, (overlap / total_wall * 100) if total_wall > 0 else 0))
    print("    Idle (neither):         {:>15.3f}  ({:>6.2f}%)".format(
        idle, (idle / total_wall * 100) if total_wall > 0 else 0))
    print("")
    print("  Verification:")
    union = comm_total + comp_total - overlap
    print("    comm + comp - overlap = {:>15.3f}  (union)".format(union))
    print("    union + idle           = {:>15.3f}  (should equal total)".format(
        union + idle))
    print("")

    # Per-function breakdown
    if comm_entries or comp_entries:
        print("  --- Per-function duration breakdown ---")
        print("  {:<25s}  {:>12s}  {:>10s}  {:>10s}".format(
            "Function", "Count", "Total(us)", "Avg(us)"))
        print("  " + "-" * 62)

        def show_group(entries, label):
            if not entries:
                return
            by_name = OrderedDict()
            for e in entries:
                by_name.setdefault(e["name"], []).append(e)
            for name in sorted(by_name.keys()):
                elist = by_name[name]
                total = sum(e["duration"] for e in elist)
                avg = total / len(elist)
                print("  {:<25s}  {:>12d}  {:>10.3f}  {:>10.3f}".format(
                    name, len(elist), total, avg))

        print("  -- Communication --")
        show_group(comm_entries, "comm")
        if comp_entries:
            print("  -- Compression --")
            show_group(comp_entries, "comp")
        print("")


# ---------------------------------------------------------------------------
# Timeline plot
# ---------------------------------------------------------------------------

def plot_timeline(rank, all_entries, comm_entries, comp_entries,
                  total_wall, args):
    """Generate a timeline visualization with matplotlib."""
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["text.usetex"] = False
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        sys.stderr.write("[error] matplotlib not available, cannot plot\n")
        return

    # Group entries by category then function name
    # Plot order: comm first, then comp
    categories = OrderedDict()
    categories["Communication"] = OrderedDict()
    for e in comm_entries:
        categories["Communication"].setdefault(e["name"], []).append(e)
    categories["Compression"] = OrderedDict()
    for e in comp_entries:
        categories["Compression"].setdefault(e["name"], []).append(e)

    # Count total rows
    total_rows = sum(len(funcs) for funcs in categories.values())
    if total_rows == 0:
        sys.stderr.write("[warn] no comm or comp entries to plot\n")
        return

    # Convert wall time to seconds for axis readability
    scale = 1000000.0  # us -> seconds
    t_min = min(e["timestamp"] for e in all_entries)
    if t_min < 0:
        # shift to make earliest timestamp 0
        t_shift = -t_min
    else:
        t_shift = 0

    # Build figure
    row_height = 0.8
    header_height = 0.3
    fig_height = max(4, total_rows * row_height + 2)
    fig, ax = plt.subplots(1, 1, figsize=(14, fig_height))

    colors = plt.cm.Set1(np.linspace(0, 1, 9))  # 9 distinct colors
    color_idx = 0
    func_colors = {}

    y = total_rows  # top-to-bottom: row 0 at top

    for cat_name, func_dict in categories.items():
        # Category header
        y -= header_height
        ax.text(-0.01, y + row_height / 2.0, cat_name,
                transform=ax.get_yaxis_transform(),
                fontsize=11, fontweight="bold", va="center",
                ha="right")
        y += header_height  # adjust visual only

        for func_name, entries in func_dict.items():
            y -= row_height
            if func_name not in func_colors:
                func_colors[func_name] = colors[color_idx % len(colors)]
                color_idx += 1
            color = func_colors[func_name]

            # Build interval list: [(start_sec, width_sec), ...]
            xranges = []
            for e in entries:
                x0 = (e["timestamp"] + t_shift) / scale
                w = e["duration"] / scale
                if w > 0:
                    xranges.append((x0, w))

            if xranges:
                ax.broken_barh(xranges, (y, row_height * 0.85),
                               facecolors=color, edgecolors="none",
                               linewidth=0)

            # Label
            label = "  " + func_name
            ax.text(-0.01, y + row_height * 0.4, label,
                    transform=ax.get_yaxis_transform(),
                    fontsize=8, va="center", ha="right",
                    family="monospace")

    ax.set_xlim(0, (total_wall + t_shift) / scale)
    ax.set_ylim(0, total_rows + len(categories) * header_height)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Function Timeline -- rank {}".format(rank))
    ax.grid(True, axis="x", alpha=0.3)

    # Move y-axis to left
    ax.spines["left"].set_position(("axes", 0.0))
    ax.yaxis.set_visible(False)

    if args.output_plot:
        out_path = args.output_plot
    else:
        out_path = "function_timeline_rank{}.png".format(rank)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sys.stderr.write("[info] timeline saved to {}\n".format(out_path))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze perf_function trace data.")

    # Rank / I/O
    parser.add_argument("--rank", type=int, default=0,
                        help="MPI rank to analyze (default: 0)")
    parser.add_argument("--perf-dir", default=DEFAULT_PERF_DIR,
                        help="Directory containing perf_function_*.csv "
                             "(default: {})".format(DEFAULT_PERF_DIR))
    parser.add_argument("--pingpong", default=DEFAULT_PINGPONG,
                        help="Path to pingpong results CSV for async "
                             "duration estimation "
                             "(default: {})".format(DEFAULT_PINGPONG))

    # Function classification sets
    parser.add_argument("--comm", metavar="PATTERN", dest="comm_funcs",
                        action="append", default=[],
                        help="Communication function pattern (substring "
                             "match, repeatable). Default: common MPI "
                             "functions")
    parser.add_argument("--comp", metavar="PATTERN", dest="comp_funcs",
                        action="append", default=[],
                        help="Compression function pattern (substring "
                             "match, repeatable). Default: ZCCL/SZ3 "
                             "functions")

    # Field name overrides (defaults match current perf_function schema)
    parser.add_argument("--func-name-field", default="func_name",
                        help="CSV column for function name (default: "
                             "func_name)")
    parser.add_argument("--timestamp-field", default="timestamp_us",
                        help="CSV column for timestamp (default: "
                             "timestamp_us)")
    parser.add_argument("--duration-field", default="duration",
                        help="CSV column for duration in microseconds "
                             "(default: duration)")
    parser.add_argument("--intra-field", default="intra",
                        help="CSV column for intra-node flag "
                             "(default: intra)")
    parser.add_argument("--msg-bytes-field", default="msg_bytes",
                        help="CSV column for message size in bytes "
                             "(default: msg_bytes)")

    # Plot
    parser.add_argument("--plot", action="store_true",
                        help="Generate timeline visualization")
    parser.add_argument("--output-plot", default=None,
                        help="Path to save the timeline plot image "
                             "(default: function_timeline_rank{rank}.png)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Use defaults if user didn't supply any patterns
    if not args.comm_funcs:
        args.comm_funcs = DEFAULT_COMM_PATTERNS
    if not args.comp_funcs:
        args.comp_funcs = DEFAULT_COMP_PATTERNS

    sys.stderr.write("[info] comm patterns: {}\n".format(args.comm_funcs))
    sys.stderr.write("[info] comp patterns: {}\n".format(args.comp_funcs))

    # Load pingpong data
    pp_data = None
    sizes_by_type = None
    if args.pingpong and os.path.isfile(args.pingpong):
        pp_data, sizes_by_type = load_pingpong(args.pingpong)
        sys.stderr.write("[info] loaded pingpong data ({} entries) from "
                         "{}\n".format(len(pp_data), args.pingpong))
    else:
        sys.stderr.write("[warn] pingpong file not found: {}; async "
                         "estimation disabled\n".format(args.pingpong))

    # Analyse the requested rank
    all_entries, comm_entries, comp_entries, other_entries = \
        analyze_rank(args.rank, args, pp_data, sizes_by_type)

    # Timeline stats
    total_wall, comm_total, comp_total, overlap, idle = \
        compute_timeline_stats(all_entries, comm_entries, comp_entries)

    # Report
    print_report(args.rank, total_wall, comm_total, comp_total, overlap,
                 idle, all_entries, comm_entries, comp_entries,
                 other_entries)

    # Plot
    if args.plot:
        if not all_entries:
            sys.stderr.write("[warn] no entries to plot\n")
        else:
            # Import numpy here (needed for plot only)
            global np
            try:
                import numpy as np
            except ImportError:
                sys.stderr.write("[error] numpy required for --plot\n")
                sys.exit(1)
            plot_timeline(args.rank, all_entries, comm_entries,
                          comp_entries, total_wall, args)


if __name__ == "__main__":
    main()
