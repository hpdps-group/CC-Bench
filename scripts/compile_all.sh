#!/bin/bash
# compile_all.sh — build all CCBench components (checker + wrappers + benchmarks)
#                 for the architecture specified in bench_basic_config.jsonc.
#
# Usage: ./scripts/compile_all.sh   (no arguments — always a full build)

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Parse architecture from config ──────────────────────────────
# ── Detect Python ───────────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
  if command -v "$cmd" >/dev/null 2>&1; then PYTHON="$cmd"; break; fi
done
[ -z "$PYTHON" ] && { echo "Error: Python not found"; exit 1; }

# ── Parse architecture from config ──────────────────────────────
ARCH=$($PYTHON -c "
import json, re
with open('userconfig/config_in_jsonc/bench_basic_config.jsonc') as f:
    text = f.read()
text = re.sub(r'//.*', '', text)
text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
text = re.sub(r',\s*([}\]])', r'\1', text)
cfg = json.loads(text)
print(cfg['communication_arch'])
")
echo "═══ compile_all.sh  arch=${ARCH} ═══"

# ── Step 1: findso checker ──────────────────────────────────────
echo ""
echo "--- Step 1: build_checker.sh ${ARCH} ---"
if [ -x "scripts/intermediate/build_checker.sh" ]; then
  scripts/intermediate/build_checker.sh "${ARCH}"
else
  echo "WARNING: scripts/intermediate/build_checker.sh not found, skipping"
fi

# ── Step 2: wrapper libraries ───────────────────────────────────
echo ""
echo "--- Step 2: wrapper libraries ---"
for s in build_comm_wrapper build_comp_wrapper build_perf_wrapper; do
  if [ -x "scripts/intermediate/${s}.sh" ]; then
    echo "  → ${s}.sh"
    "scripts/intermediate/${s}.sh"
  else
    echo "  WARNING: scripts/intermediate/${s}.sh not found, skipping"
  fi
done

# ── Step 3: validation metrics (validation.so) ──────────────────
# validation.so is loaded at runtime by the benchmark (deviation metrics).
# It was not part of the build chain, so a stale prebuilt copy (built on a
# newer glibc) could linger. Always rebuild it with the local toolchain.
echo ""
echo "--- Step 3: register_metrics.sh (validation.so) ---"
if [ -x "scripts/register_metrics.sh" ]; then
  scripts/register_metrics.sh
else
  echo "WARNING: scripts/register_metrics.sh not found, skipping"
fi

# ── Step 4: daemons (only those selected in daemon_config.jsonc) ─
# The daemon toolchain is chosen SOLELY by the arch the user sets in
# bench_basic_config.jsonc (same convention as build_tests.sh); the user is
# responsible for having that toolchain installed:
#   mpi  → gcc    (daemons never call MPI, so gcc, not mpicc)   [CC_MPI]
#   nccl → nvcc   (nvcc resolves its own CUDA include/lib paths and statically
#                  links libcudart; the CUDA burn kernel is compiled in too) [CC_NCCL]
#   rccl → hipcc  (AMD)                                          [CC_RCCL]
# On GPU arches the CUDA burn kernel source is compiled into every selected
# daemon — only the GPU stress daemon references it; harmless for the rest.
echo ""
echo "--- Step 4: daemons (selected_daemons only) ---"
DAEMON_CONFIG="userconfig/config_in_jsonc/daemon_config.jsonc"
SELECTED_DAEMONS=""
if [ -f "$DAEMON_CONFIG" ]; then
    # `|| true`: absent key must not abort the build under `set -e`.
    SELECTED_DAEMONS=$(grep '"selected_daemons"' "$DAEMON_CONFIG" | head -1 | sed 's/.*: *"\(.*\)".*/\1/') || true
fi
if [ -z "$SELECTED_DAEMONS" ]; then
  echo "  no selected_daemons in $DAEMON_CONFIG — skipping daemon build"
else
  case "$ARCH" in
    mpi)
      DAEMON_CC="${CC_MPI:-gcc}"
      DAEMON_CFLAGS="-O2 -march=x86-64 -Wall -Wextra"
      DAEMON_EXTRA_SRC=""
      ;;
    nccl)
      DAEMON_CC="${CC_NCCL:-nvcc}"
      DAEMON_CFLAGS="-O2"   # nvcc rejects -Wall/-Wextra/-march directly
      DAEMON_EXTRA_SRC="run_codes/daemons/src/daemon_gpu_burn_kernel.cu"
      ;;
    rccl)
      DAEMON_CC="${CC_RCCL:-hipcc}"
      DAEMON_CFLAGS="-O2"
      DAEMON_EXTRA_SRC=""
      ;;
    *)    echo "Unknown arch: $ARCH (valid: mpi, nccl, rccl)"; exit 1 ;;
  esac
  echo "  selected_daemons: $SELECTED_DAEMONS (compiler: $DAEMON_CC, arch=$ARCH)"
  mkdir -p bin/daemons

  for d in $(echo "$SELECTED_DAEMONS" | tr ',' ' '); do
    [ -z "$d" ] && continue
    src=""
    for dir in userconfig/daemon_code_examples userconfig/your_daemons; do
      [ -f "$dir/daemon_${d}.c" ] && src="$dir/daemon_${d}.c"
    done
    if [ -z "$src" ]; then
      echo "  WARNING: daemon_${d}.c not found, skipping"
      continue
    fi
    echo "  compiling daemon_${d} ← $src"
    $DAEMON_CC $DAEMON_CFLAGS -I./run_codes/daemons/include \
        run_codes/daemons/src/daemon_helper.c "$src" $DAEMON_EXTRA_SRC \
        -o "bin/daemons/daemon_${d}" -lm -lpthread
  done
fi

# ── Step 5: job device mapper (libjob_device_mapper.so) ─────────
echo ""
echo "--- Step 5: register_job_device_mapper.sh (libjob_device_mapper.so) ---"
if [ -x "scripts/register_job_device_mapper.sh" ]; then
  scripts/register_job_device_mapper.sh
else
  echo "WARNING: scripts/register_job_device_mapper.sh not found, skipping"
fi

# ── Step 6: benchmark binaries ──────────────────────────────────
echo ""
echo "--- Step 6: build_tests.sh ${ARCH} ---"
if [ -x "scripts/intermediate/build_tests.sh" ]; then
  scripts/intermediate/build_tests.sh "${ARCH}"
else
  echo "WARNING: scripts/intermediate/build_tests.sh not found, skipping"
fi

echo ""
echo "═══ compile_all.sh done ═══"
