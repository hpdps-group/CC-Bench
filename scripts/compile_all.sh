#!/bin/bash
# compile_all.sh — build all CCBench components (checker + wrappers + benchmarks)
#                 for the architecture specified in bench_basic_config.jsonc.
#
# Usage: ./scripts/compile_all.sh [--rebuild-bench]
#   --rebuild-bench   Recompile benchmark binaries (default: skip if already built)

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

# ── Step 3: benchmark binaries (optional) ───────────────────────
REBUILD=false
for arg in "$@"; do [ "$arg" = "--rebuild-bench" ] && REBUILD=true; done
if [ "$REBUILD" = "true" ]; then
  echo ""
  echo "--- Step 3: build_tests.sh ${ARCH} ---"
  if [ -x "scripts/intermediate/build_tests.sh" ]; then
    scripts/intermediate/build_tests.sh "${ARCH}"
  else
    echo "WARNING: scripts/intermediate/build_tests.sh not found, skipping"
  fi
else
  echo ""
  echo "--- Step 3: build_tests.sh ${ARCH} (skipped, use --rebuild-bench to rebuild) ---"
fi

echo ""
echo "═══ compile_all.sh done ═══"
