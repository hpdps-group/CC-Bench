#!/bin/bash
# build_checker.sh — compile findso.c for the given arch
#
# Usage: ./scripts/intermediate/build_checker.sh <arch>
#   arch: mpi | nccl | rccl
#
# Output: bin/<arch>/findso
#   Records the base implementation .so path to bin/libs/base_so_name
#   (used later by test binaries via load_base_impl()).
#
# Compiler selection (override via env):
#   CC_MPI=mpicc    CC_NCCL=nvcc    CC_RCCL=hipcc

set -euo pipefail
cd "$(dirname "$0")/../.."  # project root

ARCH="${1:?"Usage: $0 <arch: mpi|nccl|rccl>"}"

case "$ARCH" in
    mpi)  CC="${CC_MPI:-mpicc}"  ;;
    nccl) CC="${CC_NCCL:-nvcc}"  ;;
    rccl) CC="${CC_RCCL:-hipcc}" ;;
    *)    echo "Unknown arch: $ARCH (valid: mpi, nccl, rccl)"; exit 1 ;;
esac

SRC="run_codes/tools/$ARCH/findso.c"
OUT="bin/$ARCH/findso"

if [ ! -f "$SRC" ]; then
    echo "Error: source not found: $SRC"
    exit 1
fi

mkdir -p "bin/$ARCH"

echo "[build_checker] $CC → $OUT"
$CC "$SRC" -I run_codes/benchmark-core/include -ldl -o "$OUT"
echo "[build_checker] done: $OUT"

# Verify it runs
echo "[build_checker] verifying: $OUT"
"$OUT" && echo "[build_checker] verification passed" || {
    echo "[build_checker] verification FAILED"
    exit 1
}