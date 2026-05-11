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

# ---- Filter environment variables to exclude custom implementations ----
# When multiple NCCL forks (COCCL, ZCCL) are in CPATH/LIBRARY_PATH, the
# compiler may pick up the wrong nccl.h.  Strip any path containing known
# custom-implementation patterns, matching the same logic in findso.c.
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

if [ "$ARCH" = "nccl" ]; then
    filter_env CPATH
    filter_env C_INCLUDE_PATH
    filter_env LIBRARY_PATH
    filter_env LD_LIBRARY_PATH
fi

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

# Add wrapper sources for extended collectives (nccl only)
CFLAGS=""
# Add -g for debug symbols if GDB env var is set (for stack traces)
if [ -n "${GDB:-}" ]; then
    CFLAGS="$CFLAGS -g"
fi
LIBS="-lm -ldl"
if [ "$ARCH" = "nccl" ]; then
    for f in run_codes/wrappers/src/nccl/nccl_extensions.c; do
        [ -f "$f" ] && FRAME_SRCS+=("$f")
    done
    # Link with libnccl for symbol resolution; LD_PRELOAD overrides at runtime.
    LIBS="-lm -ldl -lnccl"
fi

# If MPI is available on the system, enable HAVE_MPI so job_device_mapper
# compiles with MPI support.  Works for both MPI and NCCL arches:
# mpicc --showme:link gives -L/-l flags that nvcc passes through to the linker.
if command -v mpicc &>/dev/null; then
    CFLAGS="-DHAVE_MPI"
    LIBS="$LIBS $(mpicc --showme:link 2>/dev/null | tr ' ' '\n' | grep '^-[Ll]' | tr '\n' ' ')"

fi

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
        -I "run_codes/wrappers/include" \
        $CFLAGS \
        $LIBS \
        -o "$out_bin"

    echo "[build_tests] done: $out_bin"
done

echo "[build_tests] all benchmarks built successfully for arch=$ARCH"