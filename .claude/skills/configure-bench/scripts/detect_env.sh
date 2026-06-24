#!/bin/bash
# ==============================================================================
# detect_env.sh — CCBench environment detection script
#
# Purpose: Automatically probe hardware, network, compilers, scheduler, etc.
# Output: JSON to stdout, easy for agents/programs to parse
# Errors: Each probe block is independent; one failure does not affect others
# ==============================================================================

set -o pipefail

# Safe command execution: returns empty if command not found, no error
safe_cmd() {
    local cmd="$1"
    local fallback="${2:-}"
    if command -v "${cmd%% *}" &>/dev/null; then
        eval "$cmd" 2>/dev/null || echo "$fallback"
    else
        echo "$fallback"
    fi
}

# Safe file read
safe_read() {
    local file="$1"
    local fallback="${2:-}"
    if [[ -f "$file" && -r "$file" ]]; then
        cat "$file" 2>/dev/null || echo "$fallback"
    else
        echo "$fallback"
    fi
}

# Parse key=value files (e.g., /etc/os-release)
parse_kv_file() {
    local file="$1"
    local key="$2"
    if [[ -f "$file" ]]; then
        grep "^${key}=" "$file" 2>/dev/null | sed 's/^[^=]*=//' | tr -d '"' | head -1
    fi
}

# ==============================================================================
# JSON output builder
# ==============================================================================

json_str() { printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'; }
json_arr() {
    # One item per line
    python3 -c "
import json, sys
items = [line.strip() for line in sys.stdin if line.strip()]
print(json.dumps(items))
"
}
json_arr_field() {
    local field="$1"
    python3 -c "
import json, sys
items = [line.strip() for line in sys.stdin if line.strip()]
print('\"' + field + '\": ' + json.dumps(items))
"
}

# ==============================================================================
# Begin detection
# ==============================================================================

# ---- Basic info ----
HOSTNAME=$(hostname 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")

# ---- CPU ----
NPROC=$(safe_cmd "nproc" "0")
CPU_MODEL=$(safe_cmd "lscpu | grep 'Model name' | head -1 | sed 's/Model name:*//' | sed 's/^[[:space:]]*//'" "unknown")
CPU_ARCH=$(safe_cmd "uname -m" "unknown")
NUMA_NODES=$(safe_cmd "numactl --hardware 2>/dev/null | grep -c available" "1")
NUMA_TOPOLOGY=$(safe_cmd "numactl --hardware 2>/dev/null" "" | head -20)

# ---- Memory ----
MEM_TOTAL_GB=$(free -g 2>/dev/null | awk '/Mem:/{print $2}')
MEM_TOTAL_GB=${MEM_TOTAL_GB:-0}
MEM_AVAIL_GB=$(free -g 2>/dev/null | awk '/Mem:/{print $7}')
MEM_AVAIL_GB=${MEM_AVAIL_GB:-0}

# ---- GPU (NVIDIA) ----
GPU_NVIDIA_PRESENT=false
GPU_NVIDIA_COUNT=0
GPU_NVIDIA_JSON="[]"
if command -v nvidia-smi &>/dev/null; then
    NVIDIA_OUTPUT=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null)
    if [[ -n "$NVIDIA_OUTPUT" ]]; then
        GPU_NVIDIA_PRESENT=true
        GPU_NVIDIA_COUNT=$(echo "$NVIDIA_OUTPUT" | wc -l)
        GPU_NVIDIA_JSON=$(echo "$NVIDIA_OUTPUT" | python3 -c "
import json, sys
gpus = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    parts = [p.strip() for p in line.split(',')]
    if len(parts) >= 2:
        gpus.append({'index': parts[0], 'name': parts[1], 'memory': parts[2] if len(parts) > 2 else 'unknown'})
print(json.dumps(gpus))
" 2>/dev/null || echo "[]")
    fi
fi

# ---- AMD GPU (ROCm) ----
GPU_AMD_PRESENT=false
GPU_AMD_COUNT=0
GPU_AMD_JSON="[]"
if command -v rocminfo &>/dev/null; then
    ROCM_OUTPUT=$(rocminfo 2>/dev/null | grep -i "gfx" | head -5)
    if [[ -n "$ROCM_OUTPUT" ]]; then
        GPU_AMD_PRESENT=true
        GPU_AMD_COUNT=$(echo "$ROCM_OUTPUT" | wc -l)
    fi
fi

# ---- CUDA ----
CUDA_FOUND=false
CUDA_VERSION=""
CUDA_NVCC_PATH=""
CUDA_HOME=""
if command -v nvcc &>/dev/null; then
    CUDA_FOUND=true
    CUDA_NVCC_PATH=$(which nvcc 2>/dev/null)
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //;s/,.*//' | head -1)
fi
# Also check common CUDA_HOME locations
for cuda_root in /usr/local/cuda /usr/lib/cuda /opt/cuda; do
    if [[ -d "$cuda_root" && -f "$cuda_root/bin/nvcc" ]]; then
        CUDA_HOME="$cuda_root"
        [[ -z "$CUDA_VERSION" ]] && CUDA_VERSION=$(safe_read "$cuda_root/version.json" "" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('cuda',{}).get('version',''))" 2>/dev/null || safe_read "$cuda_root/version.txt" "")
        break
    fi
done

# ---- NCCL ----
NCCL_FOUND=false
NCCL_VERSION=""
NCCL_LIBRARY=""
if ldconfig -p 2>/dev/null | grep -q nccl; then
    NCCL_FOUND=true
    NCCL_LIBRARY=$(ldconfig -p 2>/dev/null | grep "libnccl.so" | head -1 | awk '{print $NF}')
fi
# Check common paths
for nccl_path in /usr/lib/x86_64-linux-gnu/libnccl.so /usr/local/nccl/lib/libnccl.so; do
    if [[ -f "$nccl_path" ]]; then
        NCCL_FOUND=true
        NCCL_LIBRARY="$nccl_path"
        break
    fi
done

# ---- MPI ----
MPI_FOUND=false
MPI_IMPLEMENTATION=""
MPI_VERSION=""
MPI_COMPILER=""
if command -v mpirun &>/dev/null; then
    MPI_FOUND=true
    MPI_VERSION_OUT=$(mpirun --version 2>/dev/null | head -5)
    if echo "$MPI_VERSION_OUT" | grep -qi "open.mpi\|openmpi"; then
        MPI_IMPLEMENTATION="OpenMPI"
        MPI_VERSION=$(echo "$MPI_VERSION_OUT" | grep -i "Open MPI" | sed 's/.*Open MPI: *//;s/ .*//' | head -1)
    elif echo "$MPI_VERSION_OUT" | grep -qi "intel"; then
        MPI_IMPLEMENTATION="IntelMPI"
    elif echo "$MPI_VERSION_OUT" | grep -qi "mpich"; then
        MPI_IMPLEMENTATION="MPICH"
    else
        MPI_IMPLEMENTATION="unknown"
    fi
fi
if command -v mpicc &>/dev/null; then
    MPI_FOUND=true
    MPI_COMPILER=$(which mpicc 2>/dev/null)
elif command -v mpicxx &>/dev/null; then
    MPI_COMPILER=$(which mpicxx 2>/dev/null)
fi
if command -v mpiexec &>/dev/null && ! $MPI_FOUND; then
    MPI_FOUND=true
    MPI_IMPLEMENTATION="unknown (mpiexec found)"
fi

# ---- Compilers ----
GCC_VERSION=$(safe_cmd "gcc --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+'" "not found")
GXX_VERSION=$(safe_cmd "g++ --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+'" "not found")
CLANG_VERSION=$(safe_cmd "clang --version 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+'" "not found")

# ---- Python ----
PYTHON_VERSION=$(safe_cmd "python3 --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+'" "")
[[ -z "$PYTHON_VERSION" ]] && PYTHON_VERSION=$(safe_cmd "python --version 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+'" "not found")
PYTHON_CMD="python3"
command -v python3 &>/dev/null || PYTHON_CMD="python"

# ---- OS ----
OS_NAME=$(parse_kv_file /etc/os-release "PRETTY_NAME")
[[ -z "$OS_NAME" ]] && OS_NAME=$(safe_cmd "uname -s -r" "unknown")

# ---- InfiniBand ----
IB_FOUND=false
IB_DEVICES="[]"
if command -v ibstat &>/dev/null; then
    IB_RAW=$(ibstat 2>/dev/null)
    if [[ -n "$IB_RAW" ]]; then
        IB_FOUND=true
        IB_DEVICES=$(echo "$IB_RAW" | python3 -c "
import json, sys, re
lines = sys.stdin.read()
devices = []
current = {}
for line in lines.split('\n'):
    m = re.match(r\"CA '(\S+)'\", line)
    if m:
        if current: devices.append(current)
        current = {'name': m.group(1)}
    if current:
        m2 = re.match(r'\s+CA type:\s*(\S+)', line)
        if m2: current['type'] = m2.group(1)
        m3 = re.match(r'\s+Firmware version:\s*(\S+)', line)
        if m3: current['fw_version'] = m3.group(1)
        m4 = re.match(r'\s+Number of ports:\s*(\d+)', line)
        if m4: current['ports'] = int(m4.group(1))
        m5 = re.match(r'\s+State:\s*(\S+)', line)
        if m5: current['state'] = m5.group(1)
        m6 = re.match(r'\s+Physical state:\s*(\S+)', line)
        if m6: current['phys_state'] = m6.group(1)
        m7 = re.match(r'\s+Rate:\s*(\S+)', line)
        if m7: current['rate'] = m7.group(1)
        m8 = re.match(r'\s+Link layer:\s*(\S+)', line)
        if m8: current['link_layer'] = m8.group(1)
if current: devices.append(current)
print(json.dumps(devices))
" 2>/dev/null || echo "[]")
    fi
fi

# ---- Network interfaces ----
NET_INTERFACES_JSON="[]"
if command -v ip &>/dev/null; then
    NET_INTERFACES_JSON=$(ip -br addr show 2>/dev/null | python3 -c "
import json, sys
ifaces = []
for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) >= 2 and parts[0] != 'lo':
        iface = {'name': parts[0], 'state': parts[1]}
        if len(parts) >= 3:
            iface['ip'] = parts[2] if '/' not in parts[2] else parts[2]
        ifaces.append(iface)
print(json.dumps(ifaces))
" 2>/dev/null || echo "[]")
fi

# Default route interface
DEFAULT_IFACE=""
if command -v ip &>/dev/null; then
    DEFAULT_IFACE=$(ip route 2>/dev/null | grep "^default" | head -1 | grep -oP 'dev \K\S+')
fi

# ---- UCX ----
UCX_FOUND=false
UCX_TRANSPORTS="[]"
if command -v ucx_info &>/dev/null; then
    UCX_FOUND=true
    UCX_TRANSPORTS=$(ucx_info -d 2>/dev/null | grep "Transport:" | sed 's/.*# *Transport: *//' | sort -u | json_arr 2>/dev/null || echo "[]")
fi

# ---- SLURM ----
SLURM_FOUND=false
SLURM_PARTITIONS="[]"
if command -v sinfo &>/dev/null; then
    SLURM_FOUND=true
    SLURM_PARTITIONS=$(sinfo -o "%P %D %N %T %G" -h 2>/dev/null | python3 -c "
import json, sys, re
pars = []
for line in sys.stdin:
    parts = line.strip().split(None, 4)
    if len(parts) >= 5:
        pars.append({
            'partition': parts[0].rstrip('*'),
            'nodes': int(parts[1]) if parts[1].isdigit() else parts[1],
            'nodelist': parts[2],
            'state': parts[3],
            'gres': parts[4]
        })
    elif len(parts) >= 4:
        pars.append({
            'partition': parts[0].rstrip('*'),
            'nodes': int(parts[1]) if parts[1].isdigit() else parts[1],
            'nodelist': parts[2],
            'state': parts[3],
            'gres': ''
        })
print(json.dumps(pars))
" 2>/dev/null || echo "[]")
fi

# Current SLURM allocation (if inside a job)
SLURM_JOB_ID=""
SLURM_NODELIST=""
SLURM_NNODES=""
SLURM_NTASKS=""
SLURM_CPUS_PER_TASK=""
SLURM_JOB_ID=${SLURM_JOB_ID:-$(safe_cmd "echo \$SLURM_JOB_ID" "")}
SLURM_NODELIST=${SLURM_NODELIST:-$(safe_cmd "echo \$SLURM_NODELIST" "")}
SLURM_NNODES=${SLURM_NNODES:-$(safe_cmd "echo \$SLURM_NNODES" "")}
SLURM_NTASKS=${SLURM_NTASKS:-$(safe_cmd "echo \$SLURM_NTASKS" "")}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-$(safe_cmd "echo \$SLURM_CPUS_PER_TASK" "")}

# ---- Modules (lmod/environment-modules) ----
MODULES_LOADED="[]"
if command -v module &>/dev/null; then
    MODULES_TRY=$(module list 2>&1 | tail -n +3 | head -50)
    if [[ -n "$MODULES_TRY" ]]; then
        MODULES_LOADED=$(echo "$MODULES_TRY" | python3 -c "
import json, sys
mods = [m.strip() for m in sys.stdin if m.strip() and not m.strip().startswith('(')]
print(json.dumps(mods))
" 2>/dev/null || echo "[]")
    fi
fi

# ---- Environment Variables of Interest ----
ENV_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
ENV_PATH="${PATH:-}"
ENV_CUDA_HOME="${CUDA_HOME:-}"

# ---- Project plugin_projects ----
CCBENCH_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
CCL_PLUGINS="[]"
COMPRESSION_PLUGINS="[]"
if [[ -d "$CCBENCH_DIR/plugin_projects/ccl" ]]; then
    CCL_PLUGINS=$(ls "$CCBENCH_DIR/plugin_projects/ccl" 2>/dev/null | json_arr)
fi
if [[ -d "$CCBENCH_DIR/plugin_projects/compression" ]]; then
    COMPRESSION_PLUGINS=$(ls "$CCBENCH_DIR/plugin_projects/compression" 2>/dev/null | json_arr)
fi

# ---- Disk space ----
DISK_AVAIL_GB=0
CCBENCH_DISK_GB=$(df -BG "$CCBENCH_DIR" 2>/dev/null | awk 'NR==2{print $4}' | sed 's/G//')
CCBENCH_DISK_GB=${CCBENCH_DISK_GB:-0}
echo "DEBUG: CCBENCH_DIR=$CCBENCH_DIR" >&2
echo "DEBUG: CCBENCH_DISK_GB=$CCBENCH_DISK_GB" >&2

# ---- Datasets ----
DATASET_DIR="$CCBENCH_DIR/dataset"
DATASET_FILES="[]"
if [[ -d "$DATASET_DIR" ]]; then
    DATASET_FILES=$(find "$DATASET_DIR" -type f -name "*.f32" -o -name "*.dat" -o -name "*.bin" 2>/dev/null | head -20 | while read f; do
        size=$(stat --format=%s "$f" 2>/dev/null || echo 0)
        echo "$f;$size"
    done | python3 -c "
import json, sys
files = []
for line in sys.stdin:
    line=line.strip()
    if not line: continue
    parts = line.split(';', 1)
    relpath = parts[0]
    size = int(parts[1]) if len(parts)>1 else 0
    files.append({'path': relpath, 'size_bytes': size})
print(json.dumps(files))
" 2>/dev/null || echo "[]")
fi

# ---- Current user config summary ----
read_config_field() {
    local config_file="$CCBENCH_DIR/userconfig/config_in_jsonc/$1"
    local field="$2"
    if [[ -f "$config_file" ]]; then
        $PYTHON_CMD -c "
import json, sys
with open('$config_file') as f:
    content = f.read()
# Strip C-style comments
import re
content = re.sub(r'//.*', '', content)
content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
try:
    data = json.loads(content)
    val = data
    for k in '$field'.split('.'):
        if isinstance(val, dict):
            val = val.get(k, '__missing__')
        else:
            val = '__missing__'
            break
    if val == '__missing__':
        print('__missing__')
    elif isinstance(val, str):
        print(val)
    else:
        print(json.dumps(val))
except:
    print('__parse_error__')
" 2>/dev/null || echo "__unreadable__"
    fi
}

CURRENT_BENCH_TYPE=$(read_config_field "bench_basic_config.jsonc" "benchmark_type")
CURRENT_COMM_ARCH=$(read_config_field "bench_basic_config.jsonc" "communication_arch")
CURRENT_COMM_MODE=$(read_config_field "communication_lib_selection.jsonc" "mode")
CURRENT_COMP_MODE=$(read_config_field "compression_kernel_selection.jsonc" "mode")
CURRENT_JOB_CHOICE=$(read_config_field "job_config.jsonc" "job_choice")

# ==============================================================================
# ---- Assemble JSON output ----
# ==============================================================================

$PYTHON_CMD -c "
import json
if '$CUDA_VERSION' and '$CUDA_FOUND' == 'false':
    CUDA_FOUND = True
else:
    CUDA_FOUND = $([ "$CUDA_FOUND" = "true" ] && echo "True" || echo "False")

data = {
    'timestamp': '$TIMESTAMP',
    'hostname': '$HOSTNAME',
    'ccbench_dir': '$CCBENCH_DIR',
    'cpu': {
        'cores': $NPROC,
        'model': $(json_str "$CPU_MODEL"),
        'architecture': $(json_str "$CPU_ARCH"),
        'numa_nodes': $NUMA_NODES
    },
    'memory': {
        'total_gb': $MEM_TOTAL_GB,
        'available_gb': $MEM_AVAIL_GB
    },
    'gpu': {
        'nvidia_present': $([ "$GPU_NVIDIA_PRESENT" = "true" ] && echo "True" || echo "False"),
        'nvidia_count': $GPU_NVIDIA_COUNT,
        'nvidia_devices': $GPU_NVIDIA_JSON,
        'amd_present': $([ "$GPU_AMD_PRESENT" = "true" ] && echo "True" || echo "False"),
        'amd_count': $GPU_AMD_COUNT
    },
    'cuda': {
        'found': CUDA_FOUND,
        'version': $(json_str "$CUDA_VERSION"),
        'nvcc_path': $(json_str "$CUDA_NVCC_PATH"),
        'cuda_home': $(json_str "$CUDA_HOME")
    },
    'nccl': {
        'found': $([ "$NCCL_FOUND" = "true" ] && echo "True" || echo "False"),
        'version': $(json_str "$NCCL_VERSION"),
        'library': $(json_str "$NCCL_LIBRARY")
    },
    'mpi': {
        'found': $([ "$MPI_FOUND" = "true" ] && echo "True" || echo "False"),
        'implementation': $(json_str "$MPI_IMPLEMENTATION"),
        'version': $(json_str "$MPI_VERSION"),
        'compiler': $(json_str "$MPI_COMPILER")
    },
    'ib': {
        'found': $([ "$IB_FOUND" = "true" ] && echo "True" || echo "False"),
        'devices': $IB_DEVICES
    },
    'network': {
        'interfaces': $NET_INTERFACES_JSON,
        'default_interface': $(json_str "$DEFAULT_IFACE")
    },
    'ucx': {
        'found': $([ "$UCX_FOUND" = "true" ] && echo "True" || echo "False"),
        'transports': $UCX_TRANSPORTS
    },
    'slurm': {
        'found': $([ "$SLURM_FOUND" = "true" ] && echo "True" || echo "False"),
        'partitions': $SLURM_PARTITIONS
    },
    'slurm_allocation': {
        'job_id': $(json_str "$SLURM_JOB_ID"),
        'nodelist': $(json_str "$SLURM_NODELIST"),
        'nnodes': $(json_str "$SLURM_NNODES"),
        'ntasks': $(json_str "$SLURM_NTASKS"),
        'cpus_per_task': $(json_str "$SLURM_CPUS_PER_TASK")
    },
    'compiler': {
        'gcc': $(json_str "$GCC_VERSION"),
        'gxx': $(json_str "$GXX_VERSION"),
        'clang': $(json_str "$CLANG_VERSION")
    },
    'python': {
        'version': $(json_str "$PYTHON_VERSION"),
        'command': $(json_str "$PYTHON_CMD")
    },
    'os': $(json_str "$OS_NAME"),
    'modules_loaded': $MODULES_LOADED,
    'plugin_projects': {
        'ccl': $CCL_PLUGINS,
        'compression': $COMPRESSION_PLUGINS
    },
    'disk': {
        'ccbench_available_gb': $CCBENCH_DISK_GB
    },
    'datasets': $DATASET_FILES,
    'environment': {
        'ld_library_path': $(json_str "$ENV_LD_LIBRARY_PATH"),
        'cuda_home': $(json_str "$ENV_CUDA_HOME")
    },
    'current_config': {
        'benchmark_type': $(json_str "$CURRENT_BENCH_TYPE"),
        'communication_arch': $(json_str "$CURRENT_COMM_ARCH"),
        'communication_mode': $(json_str "$CURRENT_COMM_MODE"),
        'compression_mode': $(json_str "$CURRENT_COMP_MODE"),
        'job_choice': $(json_str "$CURRENT_JOB_CHOICE")
    }
}

print(json.dumps(data, indent=2, ensure_ascii=False))
"
