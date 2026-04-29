#!/bin/bash
# build_tests.sh — compile all benchmark tests for the given arch
#
# Usage: ./scripts/intermediate/build_tests.sh <arch>
#   arch: mpi | nccl | rccl
#
# For each test under run_codes/benchmarks/<arch>/<test_name>/<test_name>.c,
# collects framework sources + test source and compiles together:
#   - Framework base:   run_codes/benchmark-core/src/*.c
#   - Framework arch:    run_codes/benchmark-core/src/<arch>/*.c
#   - Test:              run_codes/benchmarks/<arch>/<test_name>/<test_name>.c
#
# Output: bin/<arch>/<test_name>/<test_name>
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

BENCH_DIR="run_codes/benchmarks"
CORE_SRC="run_codes/benchmark-core/src"
INC_DIR="run_codes/benchmark-core/include"

# ---- Collect all benchmark directories ----
TESTS=()
for d in "$BENCH_DIR/$ARCH"/*/; do
    [ -d "$d" ] || continue
    test_name=$(basename "$d")
    test_src="$d/$test_name.c"
    if [ -f "$test_src" ]; then
        TESTS+=("$test_name")
    else
        echo "[build_tests] warning: no $test_name.c in $d, skipping"
    fi
done

if [ ${#TESTS[@]} -eq 0 ]; then
    echo "[build_tests] no benchmarks found under $BENCH_DIR/$ARCH/"
    exit 1
fi

echo "[build_tests] arch=$ARCH, compiler=$CC"
echo "[build_tests] benchmarks found: ${TESTS[*]}"

# ---- Collect framework sources (base + arch-specific) ----
FRAME_SRCS=()
for f in "$CORE_SRC"/*.c; do
    [ -f "$f" ] && FRAME_SRCS+=("$f")
done
for f in "$CORE_SRC/$ARCH"/*.c; do
    [ -f "$f" ] && FRAME_SRCS+=("$f")
done

echo "[build_tests] framework sources (${#FRAME_SRCS[@]}): ${FRAME_SRCS[*]}"

# ---- Compile each benchmark ----
for test_name in "${TESTS[@]}"; do
    test_src="$BENCH_DIR/$ARCH/$test_name/$test_name.c"
    out_dir="bin/$ARCH/$test_name"
    out_bin="$out_dir/$test_name"

    mkdir -p "$out_dir"

    echo "[build_tests] compiling $test_name → $out_bin"
    $CC \
        "${FRAME_SRCS[@]}" \
        "$test_src" \
        -I "$INC_DIR" \
        -lm -ldl \
        -o "$out_bin"

    echo "[build_tests] done: $out_bin"
done

echo "[build_tests] all benchmarks built successfully for arch=$ARCH"