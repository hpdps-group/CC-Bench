#!/bin/bash
# build_script.sh -- generate a benchmark workflow script from JSONC configs
#
# Usage: ./scripts/build_script.sh [--rebuild-bench]
#   --rebuild-bench   Full rebuild via compile_all.sh (no args), then generate
#                     the run script (equivalent to
#                     compile_all.sh && build_script.sh)
#   -> scripts/run/run_benchmark.slurm   (sbatch)
#   -> scripts/run/run_benchmark.srun.sh (salloc terminal)
#   -> scripts/run/run_benchmark.sh      (local single-node)

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_DIR="userconfig/config_in_jsonc"
OUTPUT_DIR="scripts/run"
mkdir -p "$OUTPUT_DIR"

# Parse flags
REBUILD_BENCH=false
for arg in "$@"; do
    case "$arg" in
        --rebuild-bench) REBUILD_BENCH=true ;;
        *) echo "Unknown option: $arg (valid: --rebuild-bench)"; exit 1 ;;
    esac
done

# Detect Python 2.7+ or 3
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Error: Python not found"
    exit 1
fi

# ============================================================
# Parse all JSONC configs in one Python invocation
# ============================================================

eval "$($PYTHON << 'PYEOF'
import json, os, re

def load_jsonc(path):
    with open(path) as f:
        text = f.read()
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)

base = "userconfig/config_in_jsonc"

cfg = {}
cfg['job']    = load_jsonc(base + '/job_config.jsonc')
cfg['basic']  = load_jsonc(base + '/bench_basic_config.jsonc')
cfg['env']    = load_jsonc(base + '/environment_variables.jsonc')
cfg['daemon'] = load_jsonc(base + '/daemon_config.jsonc')
cfg['deviation'] = load_jsonc(base + '/deviation_config.jsonc')
cfg['comm']   = load_jsonc(base + '/communication_lib_selection.jsonc')
cfg['comp']   = load_jsonc(base + '/compression_kernel_selection.jsonc')
cfg['perf']   = load_jsonc(base + '/perf_kernel_selection.jsonc')

def emit(k, v):
    if isinstance(v, bool):
        v = 'true' if v else 'false'
    elif isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=True)
    else:
        v = str(v)
    escaped = v.replace("'", "'\\''")
    print(k + "='" + escaped + "'")

emit('JOB_CHOICE', cfg['job']['job_choice'])
cross_nodes = cfg['job'].get('cross_alloc_nodes', [])
if isinstance(cross_nodes, list):
    emit('CROSS_ALLOC_NODES', ','.join(str(x) for x in cross_nodes) if cross_nodes else '')
else:
    emit('CROSS_ALLOC_NODES', '')
emit('PHASE1_WARMUP', cfg['basic']['iterations']['performance']['warmup'])
emit('PHASE1_MEASURE', cfg['basic']['iterations']['performance']['measurement'])
emit('PHASE2_WARMUP', cfg['basic']['iterations']['perf_round']['warmup'])
emit('PHASE2_MEASURE', cfg['basic']['iterations']['perf_round']['measurement'])

slurm = cfg['job'].get('slurm', {})
emit('SLURM_JOB_NAME', slurm.get('job', {}).get('name', ''))
emit('SLURM_OUTPUT', slurm.get('job', {}).get('output', ''))
emit('SLURM_ERROR', slurm.get('job', {}).get('error', ''))
emit('SLURM_TIME', slurm.get('job', {}).get('time', ''))
emit('SLURM_PARTITION', slurm.get('job', {}).get('partition', ''))
emit('SLURM_QOS', slurm.get('job', {}).get('qos', ''))
emit('SLURM_ACCOUNT', slurm.get('job', {}).get('account', ''))

res = slurm.get('resources', {})
emit('SLURM_NODES', res.get('nodes', 1))
emit('SLURM_TASKS_PER_NODE', res.get('tasks_per_node', 1))
emit('SLURM_CPUS_PER_TASK', res.get('cpus_per_task', 1))
emit('SLURM_GPUS_PER_NODE', res.get('gpus_per_node', 0))
emit('SLURM_GPU_TYPE', res.get('gpu_type', ''))
emit('SLURM_MEM_GB', res.get('memory_gb', 0))
emit('SLURM_EXCLUSIVE', res.get('exclusive', False))

emit('ARCH', cfg['basic']['communication_arch'])
emit('BENCHMARK_TYPE', cfg['basic']['benchmark_type'])
emit('APP_COMMAND', cfg['basic'].get('app_command', ''))

# Gendata capture config -- auto-enabled when benchmark_type=app_gendata and targets are set
gc = cfg['basic'].get('gendata_capture', {})
targets = gc.get('targets', [])
if targets:
    parts = []
    for t in targets:
        op = t.get('operation', 'all')
        occs = t.get('occurrence', [])
        if isinstance(occs, list):
            occ_str = ','.join(str(x) for x in occs)
        else:
            occ_str = str(occs)
        parts.append('{}:{}'.format(op, occ_str))
    emit('GENDATA_TARGETS', ';'.join(parts))
else:
    emit('GENDATA_TARGETS', '')
emit('DS_TYPE', cfg['basic']['data_source']['type'])
emit('DS_FILE', cfg['basic']['data_source']['file_path'])
emit('DS_PATTERN', cfg['basic']['data_source']['pattern_type'])
emit('DS_FORMAT', cfg['basic']['data_source']['file_format'])
emit('DS_SUFFIX', cfg['basic']['data_source'].get('file_suffix', ''))

oc = cfg['basic']['data_source'].get('offset_config', {})
emit('DS_BASE_OFFSET', oc.get('base_offset', 0))
emit('DS_PER_RANK_OFFSET', oc.get('per_rank_offset', True))
custom = oc.get('custom_offsets', [])
emit('DS_CUSTOM_OFFSETS', ','.join(str(x) for x in custom) if custom else '')
ms = cfg['basic']['message_sizes']
mode = ms.get('mode', 'range')
if mode == 'list':
    msg_list = ms.get('list', [])
    emit('MSG_MODE', 'list')
    emit('MSG_LIST', ','.join(str(x) for x in msg_list) if msg_list else '')
    emit('MSG_MIN', '0')
    emit('MSG_MAX', '0')
    emit('MSG_INCR', '0')
    emit('MSG_INCR_TYPE', 'multiply')
else:
    emit('MSG_MODE', 'range')
    emit('MSG_LIST', '')
    emit('MSG_MIN', ms['min'])
    emit('MSG_MAX', ms['max'])
    emit('MSG_INCR', ms['increment'])
    emit('MSG_INCR_TYPE', ms.get('increment_type', 'multiply'))
emit('DATATYPE', cfg['basic']['mpi_operation']['datatype'].replace('MPI_', '').lower())

emit('VAL_ENABLED', cfg['deviation']['validation']['enabled'])
emit('VAL_METRICS', cfg['deviation']['validation']['selected_metrics'])

out = cfg['basic'].get('output', {})
emit('OUTPUT_CSV', out.get('csv', False))
emit('OUTPUT_PATH', out.get('csv_path', ''))
emit('OUTPUT_BINARY', out.get('binary', False))
emit('OUTPUT_BIN_PATH', out.get('bin_path', ''))

emit('DAEMON_ENABLED', cfg['daemon']['performance']['enabled'])
emit('DAEMON_PHASE1_LIST', cfg['daemon']['performance']['phase1_daemons'])
emit('DAEMON_PHASE2_LIST', cfg['daemon']['performance']['phase2_daemons'])
emit('DAEMON_INTERVAL', cfg['daemon']['performance']['poll_interval_sec'])

stress = cfg['daemon'].get('stress', {})
emit('STRESS_CPU_PERCENT', stress.get('cpu_percent', 50))
emit('STRESS_GPU_PERCENT', stress.get('gpu_percent', 50))

debug = cfg['basic'].get('debug', {})
emit('GDB_ENABLED', debug.get('gdb_enabled', False))
emit('GDB_PATH', debug.get('gdb_path', 'gdb'))

for pair in [('COMM','comm'), ('COMP','comp'), ('PERF','perf')]:
    key = pair[0]
    ckey = pair[1]
    c = cfg[ckey]
    emit(key + '_MODE', c['mode'])
    emit(key + '_LIBPATHS', json.dumps(c.get('library_paths', []), ensure_ascii=True))
    emit(key + '_OUTPUT', c.get('output_name', ''))

# Collect library directories for LD_LIBRARY_PATH.
# Mode 1 (direct prebuilt): library_paths are final .so files that go into
# LD_PRELOAD, so their directories should NOT be added to LD_LIBRARY_PATH.
all_lib_dirs = set()
for ckey in ['comm', 'comp', 'perf']:
    m = cfg[ckey].get('mode', 0)
    if m == 0 or m == 1:
        continue   # mode 0: bare (no paths needed) ; mode 1: LD_PRELOAD only
    for p in cfg[ckey].get('library_paths', []):
        d = os.path.dirname(p)
        if d:
            all_lib_dirs.add(d)
emit('LIB_DIRS', ':'.join(sorted(all_lib_dirs)))

# Env vars as shell export lines
print("ENV_EXPORT_LINES='''")
env = cfg['env']
for key in sorted(env.keys()):
    val = env[key]
    if val:
        escaped = val.replace('"', '\\"')
        print('export ' + key + '="' + escaped + '"')
print("'''")

all_env_names = sorted(env.keys()) + ["LD_PRELOAD", "LD_LIBRARY_PATH", "PATH"]
print("ENV_VAR_NAMES=\"" + " ".join(all_env_names) + "\"")
PYEOF
)"

echo "[build_script] Config parsed: arch=$ARCH, benchmark=$BENCHMARK_TYPE, job=$JOB_CHOICE"

# ============================================================
# Build step
#   --rebuild-bench: full rebuild via compile_all.sh (findso,
#                    wrappers, validation.so, daemons, mapper,
#                    benchmark binaries), then generate the run script
#   (default)      : build wrapper libraries only, then generate
# ============================================================
if [ "$REBUILD_BENCH" = "true" ]; then
    echo "[build_script] Full rebuild: calling scripts/compile_all.sh ..."
    scripts/compile_all.sh
    echo ""
else
    echo "[build_script] Building wrapper libraries..."
    for s in build_comm_wrapper build_comp_wrapper build_perf_wrapper; do
        if [ -x "scripts/intermediate/${s}.sh" ]; then
            "scripts/intermediate/${s}.sh"
        else
            echo "[build_script] WARNING: scripts/intermediate/${s}.sh not found"
        fi
    done
    echo "[build_script] Wrapper libraries done"
    echo ""
fi

# ============================================================
# Derived values
# ============================================================

TEST_NAME="${BENCHMARK_TYPE#*_}"

case "$ARCH" in
    mpi)  COMPILER="mpicc"  ;;
    nccl) COMPILER="nvcc"   ;;
    rccl) COMPILER="hipcc"  ;;
esac

NPROCS=$(( SLURM_NODES * SLURM_TASKS_PER_NODE ))

case "$JOB_CHOICE" in
    slurm) OUTPUT_SCRIPT="$OUTPUT_DIR/run_benchmark.slurm"      ;;
    srun)  OUTPUT_SCRIPT="$OUTPUT_DIR/run_benchmark.srun.sh"    ;;
    mpirun) OUTPUT_SCRIPT="$OUTPUT_DIR/run_benchmark.mpirun.sh" ;;
    local) OUTPUT_SCRIPT="$OUTPUT_DIR/run_benchmark.sh"          ;;
esac

# ── Launcher selection ──────────────────────────────────
# JOB_CHOICE determines the launcher; verify it exists at build time.
# MPI arch always uses mpirun regardless of JOB_CHOICE.
if [ "$ARCH" = "mpi" ]; then
    command -v mpirun >/dev/null 2>&1 || { echo "Error: ARCH=mpi requires mpirun (not found)" >&2; exit 1; }
    LAUNCH_CMD="mpirun -np $NPROCS"
    if [ "$JOB_CHOICE" = "srun" ]; then
        LAUNCH_CMD="mpirun -np $NPROCS --hostfile \$LSF_HOSTFILE --map-by node"
    elif [ "$JOB_CHOICE" = "mpirun" ] && [ -n "${CROSS_ALLOC_NODES:-}" ]; then
        LAUNCH_CMD="mpirun --hostfile \$CROSS_HOSTFILE -np $NPROCS --map-by node --bind-to none -x LD_LIBRARY_PATH -x LD_PRELOAD"
    fi
else
    case "$JOB_CHOICE" in
        slurm|srun)
            command -v srun >/dev/null 2>&1 || { echo "Error: JOB_CHOICE=$JOB_CHOICE but srun not found" >&2; exit 1; }
            LAUNCH_CMD="stdbuf -oL srun --nodes=$SLURM_NODES --ntasks=$NPROCS --ntasks-per-node=$SLURM_TASKS_PER_NODE"
            ;;
        mpirun)
            command -v mpirun >/dev/null 2>&1 || { echo "Error: JOB_CHOICE=mpirun but mpirun not found" >&2; exit 1; }
            if [ -n "${CROSS_ALLOC_NODES:-}" ]; then
                LAUNCH_CMD="mpirun --hostfile \$CROSS_HOSTFILE -np $NPROCS --map-by node --bind-to none -x LD_LIBRARY_PATH -x LD_PRELOAD"
            else
                LAUNCH_CMD="mpirun -np $NPROCS --bind-to none"
            fi
            ;;
        local)
            LAUNCH_CMD=""
            ;;
        *)
            echo "Error: unknown JOB_CHOICE=$JOB_CHOICE" >&2
            exit 1
            ;;
    esac
fi

# For mpirun launchers: append -x for every config env var (forcing remote export)
if [[ "$LAUNCH_CMD" = *mpirun* ]] && [ -n "${ENV_VAR_NAMES:-}" ]; then
  for _v in $ENV_VAR_NAMES; do
    LAUNCH_CMD="$LAUNCH_CMD -x $_v"
  done
fi

echo "[build_script] generating: $OUTPUT_SCRIPT"

# Clear the output file (do not append to stale content)
> "$OUTPUT_SCRIPT"

# ============================================================
# All writes go through a single function that always appends
# to OUTPUT_SCRIPT.  No heredocs, no stdout redirect tricks.
# ============================================================

write_line() { echo "$1" >> "$OUTPUT_SCRIPT"; }

# -- Daemon start/stop code generators -------------------------

# $1: daemon list (baked into generated script at build time)
# $2: PID array variable name (e.g., PHASE1_DAEMON_PIDS)
emit_daemon_start_block() {
    local list="$1"
    local pid_var="$2"
    write_line "if [ \"\$DAEMON_ENABLED\" = \"true\" ]; then"
    write_line '  # mpirun -x forwards only SET vars; default it so the stress'
    write_line '  # daemon always receives a value (0 = start immediately).'
    write_line '  export STRESS_GPU_START_DELAY="${STRESS_GPU_START_DELAY:-0}"'
    write_line "  echo \"[bench] Starting daemons...\""
    write_line "  IFS=\",\" read -ra DAEMONS <<< \"$list\""
    write_line '  for daemon in "${DAEMONS[@]}"; do'
    write_line '    d=$(echo "$daemon" | xargs)'
    write_line '    [ -z "$d" ] && continue'
    write_line '    # Clear any stale signal file (e.g. "EXIT" left over from a'
    write_line '    # previous phase stop) so the daemon does NOT exit immediately.'
    write_line '    rm -f "${BENCH_DIR}/daemon_signals/daemon_${d}.signal" 2>/dev/null || true'
    write_line '    if [ -x "$BENCH_DIR/bin/daemons/daemon_$d" ]; then'
    if [ "$JOB_CHOICE" = "slurm" ] || [ "$JOB_CHOICE" = "srun" ]; then
        write_line "      srun --nodes=\$NNODES --ntasks=\$NNODES --ntasks-per-node=1 --overlap \"\$BENCH_DIR/bin/daemons/daemon_\$d\" $DAEMON_INTERVAL &"
        write_line "      echo \"[bench]   daemon_\$d started via srun\""
    elif [ "$JOB_CHOICE" = "mpirun" ] && [ -n "${CROSS_ALLOC_NODES:-}" ]; then
        write_line "      mpirun --hostfile \$CROSS_HOSTFILE -np \$NNODES --map-by node --bind-to none -x LD_LIBRARY_PATH -x LD_PRELOAD -x STRESS_CPU_PERCENT -x STRESS_GPU_PERCENT -x STRESS_GPU_START_DELAY \"\$BENCH_DIR/bin/daemons/daemon_\$d\" $DAEMON_INTERVAL &"
        write_line "      echo \"[bench]   daemon_\$d started via mpirun across \$NNODES nodes\""
    else
        write_line "      \"\$BENCH_DIR/bin/daemons/daemon_\$d\" $DAEMON_INTERVAL &"
        write_line "      echo \"[bench]   daemon_\$d started\""
    fi
    write_line "      ${pid_var}+=(\"\$!\")"
    write_line '    fi'
    write_line '  done'
    write_line '  echo ""'
    write_line 'fi'
}

# $1: daemon list
# $2: PID array variable name
emit_daemon_stop_block() {
    local list="$1"
    local pid_var="$2"
    write_line "if [ \"\$DAEMON_ENABLED\" = \"true\" ]; then"
    write_line '  echo "[bench] Stopping daemons..."'
    write_line "  IFS=\",\" read -ra DAEMONS <<< \"$list\""
    write_line '  for _d in "${DAEMONS[@]}"; do'
    write_line '    d=$(echo "$_d" | xargs)'
    write_line '    [ -z "$d" ] && continue'
    write_line '    echo "EXIT" > "${BENCH_DIR}/daemon_signals/daemon_${d}.signal" 2>/dev/null || true'
    write_line '    echo "[bench]   signal EXIT sent to daemon_${d}"'
    write_line '  done'
    write_line '  echo "[bench]   waiting for daemons to exit gracefully (max 10s)..."'
    write_line "  for pid in \"\${${pid_var}[@]}\"; do"
    write_line '    for i in $(seq 1 20); do'
    write_line '      kill -0 "$pid" 2>/dev/null || break'
    write_line '      sleep 0.5'
    write_line '    done'
    write_line '    kill -0 "$pid" 2>/dev/null && { kill "$pid" 2>/dev/null; echo "[bench]   force-killed PID $pid"; } || echo "[bench]   PID $pid exited cleanly"'
    write_line '  done'
    write_line 'fi'
}

# -- SLURM headers ---------------------------------------------

if [ "$JOB_CHOICE" = "slurm" ]; then
    write_line "#!/bin/bash"
    write_line "# Generated by build_script.sh -- do not edit manually"
    write_line "#SBATCH --job-name=${SLURM_JOB_NAME}"
    write_line "#SBATCH --time=${SLURM_TIME}"
    write_line "#SBATCH --nodes=${SLURM_NODES}"
    write_line "#SBATCH --ntasks-per-node=${SLURM_TASKS_PER_NODE}"
    write_line "#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}"
    [ -n "${SLURM_OUTPUT}" ]     && write_line "#SBATCH --output=${SLURM_OUTPUT}"
    [ -n "${SLURM_ERROR}" ]      && write_line "#SBATCH --error=${SLURM_ERROR}"
    [ -n "${SLURM_PARTITION}" ]  && write_line "#SBATCH --partition=${SLURM_PARTITION}"
    [ -n "${SLURM_QOS}" ]        && write_line "#SBATCH --qos=${SLURM_QOS}"
    [ -n "${SLURM_ACCOUNT}" ]    && write_line "#SBATCH --account=${SLURM_ACCOUNT}"
    if [ "${SLURM_GPUS_PER_NODE}" -gt 0 ]; then
        if [ -n "${SLURM_GPU_TYPE}" ]; then
            write_line "#SBATCH --gpus-per-node=${SLURM_GPU_TYPE}:${SLURM_GPUS_PER_NODE}"
        else
            write_line "#SBATCH --gpus-per-node=${SLURM_GPUS_PER_NODE}"
        fi
    fi
    [ "${SLURM_MEM_GB}" -gt 0 ] && write_line "#SBATCH --mem=${SLURM_MEM_GB}G"
    [ "${SLURM_EXCLUSIVE}" = "true" ] && write_line "#SBATCH --exclusive"
    write_line ""
fi

# -- Preamble ---------------------------------------------------

write_line "# ===================================================================="
write_line "# Benchmark Workflow -- generated by build_script.sh"
write_line "# ===================================================================="
write_line 'set -euo pipefail'
write_line ""
write_line "# Project root (hardcoded at generation time)"
write_line "BENCH_DIR=\"$PWD\""
write_line 'cd "$BENCH_DIR"'
write_line ""
write_line '# Clean up previous perf data'
write_line 'echo "[bench] Cleaning perf_files/ ..."'
write_line 'rm -rf "${BENCH_DIR}/perf_files" && mkdir -p "${BENCH_DIR}/perf_files"'
write_line 'export PERF_OUTPUT_DIR="${BENCH_DIR}/perf_files"'
write_line '# NCCL unique-id file'
write_line 'mkdir -p "${BENCH_DIR}/nccl_id_file"'
write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_bench_id"'
write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_bench_id.meta."*'
write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_bench_id.meta."*.tmp.*'
# Clean stale TCP barrier file from previous runs (NCCL benchmark bootstrap)
if [ "$ARCH" = "nccl" ]; then
    write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_barrier_addr"'
fi
write_line ""
if [ "$OUTPUT_CSV" = "true" ] && [ -n "$OUTPUT_PATH" ]; then
    write_line "# Clean up previous CSV output"
    write_line "rm -f \"\${BENCH_DIR}/$OUTPUT_PATH\""
    write_line 'echo "[bench] Cleaning previous CSV output..."'
fi
write_line 'echo ""'
write_line ''
write_line '# Ensure signal file directory exists'
write_line 'mkdir -p "${BENCH_DIR}/daemon_signals"'
write_line 'echo "[bench]   cleaning old daemon signal files..."'
write_line 'rm -f "${BENCH_DIR}"/daemon_signals/*.signal'
write_line 'rm -f "${BENCH_DIR}"/daemon_signals/gendata_done_*'
write_line 'echo ""'
write_line ""

# -- Config variables -------------------------------------------

write_line "# -- Config ----------------------------------------------------"
write_line "JOB_CHOICE=\"$JOB_CHOICE\""
write_line "CROSS_ALLOC_NODES=\"$CROSS_ALLOC_NODES\""
write_line "ARCH=\"$ARCH\""
write_line "BENCHMARK_TYPE=\"$BENCHMARK_TYPE\""
write_line "TEST_NAME=\"$TEST_NAME\""
write_line "APP_COMMAND=\"$APP_COMMAND\""
write_line "NPROCS=$NPROCS"
write_line "NNODES=$SLURM_NODES"
write_line ""
write_line '# ── Dynamic node detection for daemon deployment (used in app_trace/app_gendata) ──'
write_line '# Falls back through: SLURM_JOB_NODELIST → cross_alloc_nodes → skip with warning'
write_line 'if [ "$BENCHMARK_TYPE" = "app_trace" ] || [ "$BENCHMARK_TYPE" = "app_gendata" ]; then'
write_line '  if [ -n "${SLURM_JOB_NODELIST:-}" ]; then'
write_line '    _NODE_LIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null || true)'
write_line '    if [ -n "$_NODE_LIST" ]; then'
write_line '      NNODES=$(echo "$_NODE_LIST" | wc -l)'
write_line '      echo "[bench]   node detection: SLURM — $NNODES nodes"'
write_line '    fi'
write_line '  elif [ -n "${CROSS_ALLOC_NODES:-}" ]; then'
write_line '    echo "[bench]   WARNING: node detection: no SLURM allocation found."'
write_line '    echo "[bench]   WARNING: falling back to cross_alloc_nodes from config."'
write_line '    echo "[bench]   WARNING: ensure cross_alloc_nodes in job_config.jsonc matches your"'
write_line '    echo "[bench]   WARNING: application'\''s actual node list."'
write_line '    IFS="," read -ra _NODES <<< "$CROSS_ALLOC_NODES"'
write_line '    NNODES=${#_NODES[@]}'
write_line '  else'
write_line '    echo "[bench]   WARNING: node detection failed — no SLURM allocation and no cross_alloc_nodes."'
write_line '    echo "[bench]   WARNING: daemon deployment requires SLURM or cross_alloc_nodes."'
write_line '    echo "[bench]   WARNING: daemon will be skipped. PMPI trace still works."'
write_line '    DAEMON_ENABLED=false'
write_line '  fi'
write_line 'fi'
write_line ""
write_line "COMPILER=\"$COMPILER\""
write_line ""
write_line "DS_TYPE=\"$DS_TYPE\""
write_line "DS_FILE=\"$DS_FILE\""
write_line "DS_PATTERN=\"$DS_PATTERN\""
write_line "DS_FORMAT=\"$DS_FORMAT\""
write_line "DS_SUFFIX=\"$DS_SUFFIX\""
write_line "DS_BASE_OFFSET=$DS_BASE_OFFSET"
write_line "DS_PER_RANK_OFFSET=$DS_PER_RANK_OFFSET"
write_line "DS_CUSTOM_OFFSETS=\"$DS_CUSTOM_OFFSETS\""
write_line 'export DS_TYPE DS_FORMAT DS_SUFFIX DS_BASE_OFFSET DS_PER_RANK_OFFSET DS_CUSTOM_OFFSETS'
write_line "MSG_MODE=\"$MSG_MODE\""
write_line "MSG_LIST=\"$MSG_LIST\""
write_line "MSG_MIN=$MSG_MIN"
write_line "MSG_MAX=$MSG_MAX"
write_line "MSG_INCR=$MSG_INCR"
write_line "MSG_INCR_TYPE=\"$MSG_INCR_TYPE\""
write_line "DATATYPE=\"$DATATYPE\""
write_line ""
write_line "VAL_ENABLED=$VAL_ENABLED"
write_line "VAL_METRICS=\"$VAL_METRICS\""
write_line ""
write_line "OUTPUT_CSV=$OUTPUT_CSV"
write_line "OUTPUT_PATH=\"$OUTPUT_PATH\""
write_line ""
write_line "OUTPUT_BINARY=$OUTPUT_BINARY"
write_line "OUTPUT_BIN_PATH=\"$OUTPUT_BIN_PATH\""
write_line ""
write_line "GENDATA_TARGETS=\"$GENDATA_TARGETS\""
write_line "ENV_VAR_NAMES=\"$ENV_VAR_NAMES\""
write_line "DAEMON_ENABLED=$DAEMON_ENABLED"
write_line "DAEMON_PHASE1_LIST=\"$DAEMON_PHASE1_LIST\""
write_line "DAEMON_PHASE2_LIST=\"$DAEMON_PHASE2_LIST\""
write_line "DAEMON_INTERVAL=$DAEMON_INTERVAL"
write_line ""
write_line "STRESS_CPU_PERCENT=${STRESS_CPU_PERCENT:-$STRESS_CPU_PERCENT}"
write_line 'export STRESS_CPU_PERCENT'
write_line "STRESS_GPU_PERCENT=${STRESS_GPU_PERCENT:-$STRESS_GPU_PERCENT}"
write_line 'export STRESS_GPU_PERCENT'
write_line ""
write_line "PHASE1_WARMUP=$PHASE1_WARMUP"
write_line "PHASE1_MEASURE=$PHASE1_MEASURE"
write_line "PHASE2_WARMUP=$PHASE2_WARMUP"
write_line "PHASE2_MEASURE=$PHASE2_MEASURE"
write_line ""
write_line "COMM_LIBPATHS='$COMM_LIBPATHS'"
write_line "COMM_OUTPUT=\"$COMM_OUTPUT\""
write_line "COMP_LIBPATHS='$COMP_LIBPATHS'"
write_line "COMP_OUTPUT=\"$COMP_OUTPUT\""
write_line "PERF_LIBPATHS='$PERF_LIBPATHS'"
write_line "PERF_OUTPUT=\"$PERF_OUTPUT\""
write_line ""
write_line "# Prepend library directories to LD_LIBRARY_PATH"
write_line "LIB_DIRS=\"$LIB_DIRS\""
write_line '[ -n "$LIB_DIRS" ] && export LD_LIBRARY_PATH="${LIB_DIRS}:${LD_LIBRARY_PATH}"'
# Runtime environment variables must be exported BEFORE LD_PRELOAD is set.
# A preloaded library (e.g. libzccl_comm_wrapper.so) may depend on a shared
# object (e.g. libZCCL.so.1) living in one of these paths; if LD_LIBRARY_PATH
# is not updated first, every tool invoked after LD_PRELOAD (mktemp, wc, ...)
# fails with "cannot open shared object file".
echo "$ENV_EXPORT_LINES" >> "$OUTPUT_SCRIPT"
write_line ""
write_line "# Generate hostfile from SLURM allocation (srun mode)"
if [ "$JOB_CHOICE" = "srun" ]; then
    case "$LAUNCH_CMD" in
    *mpirun*)
        write_line 'if [ -z "${SLURM_NODELIST:-}" ]; then'
        write_line '  echo "Error: srun mode requires a SLURM allocation (run salloc first)"'
        write_line '  exit 1'
        write_line 'fi'
        write_line 'LSF_HOSTFILE=$(mktemp)'
        write_line 'scontrol show hostnames "$SLURM_NODELIST" > "$LSF_HOSTFILE"'
        write_line 'trap "rm -f \"$LSF_HOSTFILE\"" EXIT'
        write_line 'echo "[bench] Hostfile: $LSF_HOSTFILE ($(wc -l < "$LSF_HOSTFILE") nodes)"'
        write_line ""
        ;;
    esac
fi

# ============================================================
# Helper functions that write to the script
# ============================================================

# write_preload_entry name mode libpaths_json output
write_preload_entry() {
    local name="$1" mode="$2" libpaths_json="$3" output="$4"
    [ "$mode" = "0" ] && return 0
    if [ "$mode" = "1" ]; then
        write_line "# mode 1: LD_PRELOAD each prebuilt library"
        write_line "for _lib in $libpaths_json; do"
        write_line '  _lib=$(echo "$_lib" | tr -d "[:space:],\"\\[\\]")'
        write_line '  [ -z "$_lib" ] && continue'
        write_line '  if [ -f "$_lib" ]; then'
        write_line '    LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$_lib"'
        write_line '    echo "[bench]   '"$name"': $_lib"'
        write_line '  fi'
        write_line "done"
    else
        write_line "if [ -f \"\$BENCH_DIR/bin/libs/$output\" ]; then"
        write_line "  LD_PRELOAD=\"\${LD_PRELOAD:+\$LD_PRELOAD:}\$BENCH_DIR/bin/libs/$output\""
        write_line "  echo \"[bench]   $name: \$BENCH_DIR/bin/libs/$output\""
        write_line "fi"
    fi
}

# ============================================================
# Step 1 -- findso
# ============================================================

write_line ""
write_line 'echo "============================================"'
write_line "echo \" Benchmark: $BENCHMARK_TYPE\""
write_line "echo \" Arch:      $ARCH\""
write_line "echo \" Nodes:     $NPROCS\""
write_line 'echo "============================================"'
write_line "echo \"\""
write_line ""
write_line 'echo "[bench] Step 1: Checking base implementation..."'
write_line "if [ -x \"\$BENCH_DIR/bin/$ARCH/findso\" ]; then"
write_line "  \"\$BENCH_DIR/bin/$ARCH/findso\""
write_line "  if [ \$? -ne 0 ]; then"
write_line '    echo "[bench] ERROR: findso failed"'
write_line "    exit 1"
write_line "  fi"
write_line "else"
write_line '  echo "[bench] WARNING: $BENCH_DIR/bin/$ARCH/findso not found"'
write_line "  echo \"[bench]   Run: $PWD/scripts/intermediate/build_checker.sh $ARCH\""
write_line "fi"
write_line "echo \"\""

# ============================================================
# Step 2 -- Wrapper libraries (pre-built locally by intermediate scripts)
# ============================================================

write_line 'echo "[bench] Step 2: Wrapper libraries ready (built locally)"'
write_line ""


# ============================================================
# Step 3 -- LD_PRELOAD (base: compression + communication)
# ============================================================

write_line 'echo "[bench] Step 3: Setting up LD_PRELOAD..."'
write_line 'LD_PRELOAD=""'

write_preload_entry "compression" "$COMP_MODE" "$COMP_LIBPATHS" "$COMP_OUTPUT"
write_preload_entry "communication" "$COMM_MODE" "$COMM_LIBPATHS" "$COMM_OUTPUT"

# Extensions: default ncclAllToAll etc. (NCCL/RCCL only)
write_line 'if [ "$ARCH" = "nccl" ] || [ "$ARCH" = "rccl" ]; then'
write_line '  if [ -f "${BENCH_DIR}/bin/libs/libnccl_extensions.so" ]; then'
write_line '    LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}${BENCH_DIR}/bin/libs/libnccl_extensions.so"'
write_line '    echo "[bench]   extensions: ${BENCH_DIR}/bin/libs/libnccl_extensions.so"'
write_line '  fi'
write_line 'fi'

write_line 'export LD_PRELOAD'
write_line 'echo "[bench] Base LD_PRELOAD (no perf): ${LD_PRELOAD:-"(none)"}"'
write_line "echo \"\""

# ── Cross-allocation hostfile (mpirun + cross_alloc_nodes) ──
write_line '# ── Cross-allocation hostfile ──'
write_line 'if [ -n "${CROSS_ALLOC_NODES:-}" ]; then'
write_line '  echo "[bench] Cross-allocation mode: generating hostfile..."'
write_line '  CROSS_HOSTFILE=$(mktemp)'
write_line '  trap "rm -f $CROSS_HOSTFILE" EXIT'
write_line '  IFS="," read -ra _NODES <<< "$CROSS_ALLOC_NODES"'
write_line '  _NODES=("${_NODES[@]:0:$NNODES}")'
write_line '  for _node in "${_NODES[@]}"; do'
write_line '    echo "$_node slots=$(( NPROCS / NNODES ))" >> "$CROSS_HOSTFILE"'
write_line '  done'
write_line '  echo "[bench]   hostfile: $CROSS_HOSTFILE ($(wc -l < "$CROSS_HOSTFILE") nodes)"'
write_line '  echo "[bench]   cleaning SLURM env vars for cross-allocation..."'
write_line '  unset SLURM_JOB_ID SLURM_NODELIST SLURM_NNODES SLURM_NTASKS'
write_line '  unset SLURM_TASKS_PER_NODE SLURM_NTASKS_PER_NODE SLURM_NTASKS_PER_NODE'
write_line '  unset SLURM_JOB_NODELIST SLURM_JOBID SLURM_JOB_NUM_NODES SLURM_NODEID SLURM_PROCID'
write_line '  echo "[bench]   ready — mpirun will use --hostfile $CROSS_HOSTFILE"'
write_line 'fi'
write_line ""

# Step 4 -- Environment variables (applies to both phases)
# Runtime env vars are exported near the top of the script (before LD_PRELOAD),
# so that preloaded libraries' shared-object dependencies resolve.
write_line ""

# ============================================================
# Phase 1 -- Performance round (with phase1 daemons)
# ============================================================

BENCH_BIN='${BENCH_DIR}/bin/${ARCH}/${TEST_NAME}/${TEST_NAME}'

write_line 'echo "--- Phase 1: Performance round ---"'
write_line "echo \"[bench] Running benchmark $BENCHMARK_TYPE...\""
write_line ""

if [ "$MSG_MODE" = "list" ]; then
    PHASE1_ARGS="-L ${MSG_LIST}"
elif [ "$MSG_INCR_TYPE" = "add" ]; then
    PHASE1_ARGS="-m ${MSG_MIN}:${MSG_MAX}:${MSG_INCR} -A"
else
    PHASE1_ARGS="-m ${MSG_MIN}:${MSG_MAX}:${MSG_INCR}"
fi
PHASE1_ARGS="$PHASE1_ARGS -i $PHASE1_MEASURE -w $PHASE1_WARMUP"
PHASE1_ARGS="$PHASE1_ARGS -p $DS_PATTERN"
[ "$VAL_ENABLED" = "true" ] && PHASE1_ARGS="$PHASE1_ARGS -v -e \"$VAL_METRICS\""
{ [ "$DS_TYPE" = "file" ] || [ "$DS_TYPE" = "folder" ]; } && [ -n "$DS_FILE" ] && PHASE1_ARGS="$PHASE1_ARGS -f \"$DS_FILE\""
PHASE1_ARGS="$PHASE1_ARGS -d $DATATYPE"
[ "$OUTPUT_CSV" = "true" ] && [ -n "$OUTPUT_PATH" ] && PHASE1_ARGS="$PHASE1_ARGS -c -o \"$OUTPUT_PATH\""
[ "$OUTPUT_BINARY" = "true" ] && [ -n "$OUTPUT_BIN_PATH" ] && PHASE1_ARGS="$PHASE1_ARGS -b -B \"$OUTPUT_BIN_PATH\""

write_line ""
write_line '# Start phase1 daemons'
write_line 'PHASE1_DAEMON_PIDS=()'
emit_daemon_start_block "$DAEMON_PHASE1_LIST" PHASE1_DAEMON_PIDS

write_line '# GDB debug wrapper — batch mode, auto bt on crash if enabled'
if [ "$GDB_ENABLED" = "true" ] || [ "$GDB_ENABLED" = "1" ]; then
  write_line "GDB_WRAPPER=\"${GDB_PATH:-gdb} -batch -ex run -ex bt -ex \\\"bt full\\\" --args\""
else
  write_line 'GDB_WRAPPER=""'
fi
write_line ""
write_line '# ── app_trace/app_gendata: exec user command instead of benchmark binary ──'
write_line 'if [ "$BENCHMARK_TYPE" = "app_trace" ] || [ "$BENCHMARK_TYPE" = "app_gendata" ]; then'
write_line '  echo "[bench] Launching app (Phase 1): $APP_COMMAND"'
write_line '  PHASE1_START=$(date +%s.%N)'
write_line '  # ── Auto-detect launcher and forward env vars to remote nodes ──'
write_line '  case "$APP_COMMAND" in'
write_line '    *mpirun*)'
write_line '      _X_ARGS=""'
write_line '      for _v in $ENV_VAR_NAMES; do'
write_line '        _X_ARGS="$_X_ARGS -x $_v"'
write_line '      done'
write_line '      echo "[bench]   mpirun detected — auto-inserting env forwarding flags"'
write_line '      eval "${APP_COMMAND/mpirun/mpirun $_X_ARGS}"'
write_line '      ;;'
write_line '    srun*)'
write_line '      echo "[bench]   srun detected — env vars inherited by default"'
write_line '      eval "$APP_COMMAND"'
write_line '      ;;'
write_line '    *)'
write_line '      eval "$APP_COMMAND"'
write_line '      ;;'
write_line '  esac'
write_line '  PHASE1_END=$(date +%s.%N)'
write_line 'else'
write_line "$LAUNCH_CMD \$GDB_WRAPPER \\"
write_line "    \"\$BENCH_DIR/bin/\${ARCH}/\${TEST_NAME}/\${TEST_NAME}\" $PHASE1_ARGS"
write_line 'fi'

write_line 'BENCH_RC1=$?'
write_line 'echo ""'

write_line '# Stop phase1 daemons'
emit_daemon_stop_block "$DAEMON_PHASE1_LIST" PHASE1_DAEMON_PIDS

# Clean metadata before Phase 2 to prevent stale metadata race
write_line 'echo "[bench] Cleaning metadata files for Phase 2..."'
write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_bench_id.meta."*'
write_line 'rm -f "${BENCH_DIR}/nccl_id_file/nccl_bench_id.meta."*.tmp.*'
write_line ''

write_line ""

# ============================================================
# Phase 2 -- Perf breakdown round (with perf wrapper + daemons)
# ============================================================

write_line 'echo "--- Phase 2: Perf breakdown round ---"'
write_line ""

# Start daemons
write_line "# Step 5 -- Start monitoring daemons"
write_line 'PHASE2_DAEMON_PIDS=()'
write_line ""
emit_daemon_start_block "$DAEMON_PHASE2_LIST" PHASE2_DAEMON_PIDS

# Now add perf to LD_PRELOAD — put it FIRST so it intercepts before
# compression / communication wrappers and can chain via RTLD_NEXT.
write_line 'echo "[bench] Adding perf wrapper to LD_PRELOAD..."'
write_line 'PERF_OLD="$LD_PRELOAD"'
write_line 'LD_PRELOAD=""'
write_preload_entry "perf" "$PERF_MODE" "$PERF_LIBPATHS" "$PERF_OUTPUT"
write_line 'LD_PRELOAD="${LD_PRELOAD}${PERF_OLD:+:$PERF_OLD}"'

write_line 'export LD_PRELOAD'
write_line 'echo "[bench] Final LD_PRELOAD: ${LD_PRELOAD:-"(none)"}"'
write_line "echo \"\""

write_line "echo \"[bench] Running benchmark $BENCHMARK_TYPE...\""
write_line ""

if [ "$MSG_MODE" = "list" ]; then
    PHASE2_ARGS="-L ${MSG_LIST}"
elif [ "$MSG_INCR_TYPE" = "add" ]; then
    PHASE2_ARGS="-m ${MSG_MIN}:${MSG_MAX}:${MSG_INCR} -A"
else
    PHASE2_ARGS="-m ${MSG_MIN}:${MSG_MAX}:${MSG_INCR}"
fi
PHASE2_ARGS="$PHASE2_ARGS -i $PHASE2_MEASURE -w $PHASE2_WARMUP"
PHASE2_ARGS="$PHASE2_ARGS -p $DS_PATTERN"
[ "$VAL_ENABLED" = "true" ] && PHASE2_ARGS="$PHASE2_ARGS -v -e \"$VAL_METRICS\""
{ [ "$DS_TYPE" = "file" ] || [ "$DS_TYPE" = "folder" ]; } && [ -n "$DS_FILE" ] && PHASE2_ARGS="$PHASE2_ARGS -f \"$DS_FILE\""
PHASE2_ARGS="$PHASE2_ARGS -d $DATATYPE"
[ "$OUTPUT_CSV" = "true" ] && [ -n "$OUTPUT_PATH" ] && PHASE2_ARGS="$PHASE2_ARGS -c -o \"$OUTPUT_PATH\""
[ "$OUTPUT_BINARY" = "true" ] && [ -n "$OUTPUT_BIN_PATH" ] && PHASE2_ARGS="$PHASE2_ARGS -b -B \"$OUTPUT_BIN_PATH\""

write_line '# ── app_trace/app_gendata: exec user command ──'
write_line 'if [ "$BENCHMARK_TYPE" = "app_trace" ] || [ "$BENCHMARK_TYPE" = "app_gendata" ]; then'
write_line '  echo "[bench] Launching app (Phase 2): $APP_COMMAND"'
write_line '  # app_gendata: export dataset construction env vars'
write_line '  if [ "$BENCHMARK_TYPE" = "app_gendata" ] && [ -n "$GENDATA_TARGETS" ]; then'
write_line '    export PERF_GENDATA_TARGETS="$GENDATA_TARGETS"'
write_line '    case "$OUTPUT_BIN_PATH" in'
write_line '      /*) export PERF_GENDATA_OUTPUT_DIR="$OUTPUT_BIN_PATH" ;;'
write_line '      *)  export PERF_GENDATA_OUTPUT_DIR="${BENCH_DIR}/${OUTPUT_BIN_PATH}" ;;'
write_line '    esac'
write_line '    export PERF_GENDATA_DONE_DIR="${BENCH_DIR}/daemon_signals"'
write_line '  fi'
write_line '  PHASE2_START=$(date +%s.%N)'
write_line '  # ── Auto-detect launcher and forward env vars to remote nodes ──'
write_line '  case "$APP_COMMAND" in'
write_line '    *mpirun*)'
write_line '      _X_ARGS=""'
write_line '      for _v in $ENV_VAR_NAMES; do'
write_line '        _X_ARGS="$_X_ARGS -x $_v"'
write_line '      done'
write_line '      # Also forward gendata env vars if set'
write_line '      if [ "$BENCHMARK_TYPE" = "app_gendata" ] && [ -n "$GENDATA_TARGETS" ]; then'
write_line '        for _v in PERF_GENDATA_TARGETS PERF_GENDATA_OUTPUT_DIR PERF_GENDATA_DONE_DIR; do'
write_line '          _X_ARGS="$_X_ARGS -x $_v"'
write_line '        done'
write_line '      fi'
write_line '      echo "[bench]   mpirun detected — auto-inserting env forwarding flags"'
write_line '      eval "${APP_COMMAND/mpirun/mpirun $_X_ARGS}"'
write_line '      ;;'
write_line '    srun*)'
write_line '      echo "[bench]   srun detected — env vars inherited by default"'
write_line '      eval "$APP_COMMAND"'
write_line '      ;;'
write_line '    *)'
write_line '      eval "$APP_COMMAND"'
write_line '      ;;'
write_line '  esac'
write_line '  PHASE2_END=$(date +%s.%N)'
write_line '  echo ""'
write_line '  echo "--- Phase 2 app wall time: $(echo "$PHASE2_END - $PHASE2_START" | bc) seconds ---"'
write_line 'else'
write_line "$LAUNCH_CMD \$GDB_WRAPPER \\"
write_line "    \"\$BENCH_DIR/bin/\${ARCH}/\${TEST_NAME}/\${TEST_NAME}\" $PHASE2_ARGS"
write_line 'fi'

write_line 'BENCH_RC2=$?'
write_line 'echo ""'
write_line ""

# ============================================================
# Step 6 -- Stop daemons (after Phase 2)
# ============================================================

emit_daemon_stop_block "$DAEMON_PHASE2_LIST" PHASE2_DAEMON_PIDS

# ============================================================
# Done
# ============================================================

write_line 'BENCH_RC=$(( BENCH_RC1 > BENCH_RC2 ? BENCH_RC1 : BENCH_RC2 ))'
write_line 'echo "============================================"'
write_line "echo \" Benchmark $BENCHMARK_TYPE complete\""
write_line 'if [ "$BENCHMARK_TYPE" = "app_trace" ] || [ "$BENCHMARK_TYPE" = "app_gendata" ]; then'
write_line '  echo "   Phase 1: exit $BENCH_RC1"'
write_line '  echo "   Phase 2: exit $BENCH_RC2"'
write_line 'else'
write_line '  echo "   Phase 1 (perf):  exit $BENCH_RC1"'
write_line '  echo "   Phase 2 (perf round): exit $BENCH_RC2"'
write_line 'fi'
write_line ""
write_line 'exit $BENCH_RC'

chmod +x "$OUTPUT_SCRIPT"

echo "[build_script] Generated: $OUTPUT_SCRIPT"
echo "[build_script] Done"