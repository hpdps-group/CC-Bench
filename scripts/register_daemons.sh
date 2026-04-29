#!/bin/bash
# register_daemons.sh — scan daemon .c files and compile executables
#
# Usage:
#   ./scripts/register_daemons.sh               # scan + compile
#   ./scripts/register_daemons.sh --dry-run      # just list what would be built
#
# Scans these directories for daemon_<name>.c files:
#   userconfig/daemon_code_examples/   (reference examples)
#   userconfig/your_daemons/          (user's own daemons)
#
# Each file must be named daemon_<name>.c and contain main().
# daemon_helper.c/.h are excluded (they are the shared framework).
#
# Output: bin/daemons/daemon_<name> (one executable per daemon)
#
# Name collision rules:
#   - Two daemons with same <name> from different dirs → error

set -euo pipefail
cd "$(dirname "$0")/.."  # project root (mybench/)

DAEMON_DIRS=(
    userconfig/daemon_code_examples
    userconfig/your_daemons
)
HELPER_SRC=run_codes/daemons/src/daemon_helper.c
HELPER_INC="-I./run_codes/daemons/include"
OUTPUT_DIR=bin/daemons
DRY_RUN="${1:-}"

mkdir -p "$OUTPUT_DIR"

# ---- Collect & sort daemon source files ----
declare -A daemon_seen  # name -> source file
FILES=""

for dir in "${DAEMON_DIRS[@]}"; do
    [ -d "$dir" ] || continue
    for f in "$dir"/daemon_*.c; do
        [ -f "$f" ] || continue

        # Skip the shared helper module
        bn=$(basename "$f")
        [ "$bn" = "daemon_helper.c" ] && continue

        # Extract daemon name (daemon_<name>.c → <name>)
        name=$(basename "$f" .c)
        name="${name#daemon_}"

        if [ -z "$name" ]; then
            echo "register_daemons.sh: error: malformed filename '$bn' (expected daemon_<name>.c)" >&2
            exit 1
        fi

        # Check intra-daemon collision
        if [ -n "${daemon_seen[$name]:-}" ]; then
            echo "register_daemons.sh: error: daemon name collision: '$name'" >&2
            echo "  first:  ${daemon_seen[$name]}" >&2
            echo "  second: $f" >&2
            exit 1
        fi
        daemon_seen["$name"]="$f"

        FILES="$FILES $name:$f"
    done
done

if [ -z "$FILES" ]; then
    echo "register_daemons.sh: no daemon_*.c files found (excluding daemon_helper.c) in:" >&2
    printf '  %s\n' "${DAEMON_DIRS[@]}" >&2
    exit 1
fi

# Sort alphabetically for deterministic output
FILES=$(echo "$FILES" | tr ' ' '\n' | sort)

echo "register_daemons.sh: found daemons:" >&2
for entry in $FILES; do
    name="${entry%%:*}"
    src="${entry#*:}"
    echo "  $name  ← $src" >&2
done

# ---- Check that the helper module exists ----
if [ ! -f "$HELPER_SRC" ]; then
    echo "register_daemons.sh: error: helper not found: $HELPER_SRC" >&2
    exit 1
fi

# ---- Compile each daemon ----
for entry in $FILES; do
    name="${entry%%:*}"
    src="${entry#*:}"
    out="$OUTPUT_DIR/daemon_$name"

    if [ "$DRY_RUN" = "--dry-run" ]; then
        echo "  would compile: $src + $HELPER_SRC → $out" >&2
        continue
    fi

    echo "register_daemons.sh: compiling $out ..." >&2
    gcc -O3 -Wall -Wextra "$HELPER_INC" \
        "$HELPER_SRC" "$src" -o "$out" -lm
done

if [ "$DRY_RUN" = "--dry-run" ]; then
    exit 0
fi

echo "register_daemons.sh: done — $(echo "$FILES" | wc -l) daemon(s) in $OUTPUT_DIR/" >&2