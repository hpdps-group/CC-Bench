#!/bin/bash
# build_perf_wrapper.sh — compile the performance wrapper .so locally
#
# Reads perf_kernel_selection.jsonc + bench_basic_config.jsonc.
# Can run standalone or be called by build_script.sh.
#
# Usage: ./scripts/intermediate/build_perf_wrapper.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then PYTHON="$cmd"; break; fi
done
if [ -z "$PYTHON" ]; then echo "Error: Python not found"; exit 1; fi

eval "$($PYTHON << 'PYEOF'
import json, re

def load_jsonc(path):
    with open(path) as f:
        text = f.read()
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)

base = "userconfig/config_in_jsonc"
cfg = load_jsonc(base + '/perf_kernel_selection.jsonc')
basic = load_jsonc(base + '/bench_basic_config.jsonc')

arch  = basic.get('communication_arch', 'mpi')
# Perf wrappers are LD_PRELOAD host-code libraries (NCCL stubs + dlsym).
# They don't compile any CUDA kernels, so always use the system C compiler.
comp  = {'mpi': 'mpicc', 'nccl': 'cc', 'rccl': 'cc'}.get(arch, 'mpicc')

def emit(k, v):
    if isinstance(v, bool):           v = 'true' if v else 'false'
    elif isinstance(v, (list, dict)): v = json.dumps(v, ensure_ascii=True)
    else:                             v = str(v)
    print(k + "='" + v.replace("'", "'\\''") + "'")

emit('PERF_MODE',      cfg.get('mode', 0))
emit('PERF_LIBPATHS',  json.dumps(cfg.get('library_paths', []), ensure_ascii=True))
emit('PERF_OUTPUT',    cfg.get('output_name', ''))
emit('PERF_SRC_FILES', json.dumps(cfg.get('source_files', []), ensure_ascii=True))
emit('PERF_SRC_DIRS',  json.dumps(cfg.get('source_dirs', []), ensure_ascii=True))
emit('PERF_INC_FILES', json.dumps(cfg.get('include_files', []), ensure_ascii=True))
emit('PERF_INC_DIRS',  json.dumps(cfg.get('include_dirs', []), ensure_ascii=True))
emit('PERF_LIBS',      json.dumps(cfg.get('libraries', []), ensure_ascii=True))
emit('PERF_ARCH', arch)
emit('COMPILER', comp)
PYEOF
)"

# ---- Filter environment to exclude custom NCCL implementations ----
filter_env() {
    local var="$1"
    local val="${!var:-}"
    [ -z "$val" ] && return
    local new_val=""
    IFS=':' read -ra parts <<< "$val"
    for part in "${parts[@]}"; do
        local skip=0
        case "$part" in
            *coccl*|*zccl*) skip=1 ;;
        esac
        [ "$skip" -eq 0 ] && new_val="${new_val:+$new_val:}$part"
    done
    export "$var=$new_val"
}

if [ "$PERF_MODE" != "0" ] && { [ "$PERF_ARCH" = "nccl" ] || [ "$PERF_ARCH" = "rccl" ]; }; then
    filter_env CPATH
    filter_env C_INCLUDE_PATH
    filter_env LIBRARY_PATH
    filter_env LD_LIBRARY_PATH
fi

# ---- Mode dispatch ----------------------------------------------------------

case "$PERF_MODE" in
    0) echo "[build_perf] Mode 0: perf wrapper disabled"; exit 0 ;;
    1)
        if [ "$PERF_LIBPATHS" != "[]" ]; then
            echo "[build_perf] Mode 1: prebuilt library(s) configured"
        else
            echo "[build_perf] WARNING: mode 1 but no library_paths set"
        fi
        exit 0
        ;;
esac

# Mode 2 / 3 — compile from source
[ -z "$PERF_OUTPUT" ] && { echo "[build_perf] ERROR: output_name is empty"; exit 1; }

echo "[build_perf] Building perf wrapper: bin/libs/$PERF_OUTPUT"

SRC_FILES=()

# source_files
if [ "$PERF_SRC_FILES" != "[]" ]; then
    for item in $($PYTHON -c "import json, sys; print('|'.join(json.load(sys.stdin)))" <<< "$PERF_SRC_FILES" 2>/dev/null); do
        IFS='|' read -ra parts <<< "$item"
        for p in "${parts[@]}"; do [ -n "$p" ] && SRC_FILES+=("$p"); done
    done
fi

# source_dirs
if [ "$PERF_SRC_DIRS" != "[]" ]; then
    for item in $($PYTHON -c "import json, sys; print('|'.join(json.load(sys.stdin)))" <<< "$PERF_SRC_DIRS" 2>/dev/null); do
        IFS='|' read -ra parts <<< "$item"
        for d in "${parts[@]}"; do
            [ -z "$d" ] && continue
            [ -d "$d" ] && while IFS= read -r -d '' f; do SRC_FILES+=("$f"); done < <(find "$d" \( -name '*.c' -o -name '*.cu' \) -print0 2>/dev/null)
        done
    done
fi

if [ ${#SRC_FILES[@]} -eq 0 ]; then
    echo "[build_perf] WARNING: no source files found, skipping"; exit 0
fi

# include flags
INC_FLAGS="-I run_codes/wrappers/include -I run_codes/wrappers"
if [ "$PERF_INC_DIRS" != "[]" ]; then
    for item in $($PYTHON -c "import json, sys; print('|'.join(json.load(sys.stdin)))" <<< "$PERF_INC_DIRS" 2>/dev/null); do
        IFS='|' read -ra parts <<< "$item"
        for d in "${parts[@]}"; do [ -z "$d" ] || INC_FLAGS="$INC_FLAGS -I $d"; done
    done
fi
if [ "$PERF_INC_FILES" != "[]" ]; then
    for item in $($PYTHON -c "import json, sys; print('|'.join(json.load(sys.stdin)))" <<< "$PERF_INC_FILES" 2>/dev/null); do
        IFS='|' read -ra parts <<< "$item"
        for f in "${parts[@]}"; do [ -z "$f" ] || INC_FLAGS="$INC_FLAGS -I $(dirname "$f")"; done
    done
fi

# lib flags
LIB_FLAGS=""
[ "$PERF_LIBS" != "[]" ] && LIB_FLAGS=$($PYTHON -c "import json, sys; print(' '.join(json.load(sys.stdin)))" <<< "$PERF_LIBS" 2>/dev/null) || true

# Mode 2: link with prebuilt .so
LINK_LIB=""
if [ "$PERF_MODE" = "2" ] && [ "$PERF_LIBPATHS" != "[]" ]; then
    LINK_LIB=$($PYTHON -c "import json, sys; print(' '.join(json.load(sys.stdin)))" <<< "$PERF_LIBPATHS" 2>/dev/null) || true
fi

mkdir -p bin/libs
$COMPILER -O2 -fPIC -shared "${SRC_FILES[@]}" $INC_FLAGS $LIB_FLAGS $LINK_LIB -o "bin/libs/$PERF_OUTPUT"
echo "[build_perf] Built bin/libs/$PERF_OUTPUT"