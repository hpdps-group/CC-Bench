#!/bin/bash
# manage_snapshots.sh — interactive snapshot management for CCBench
#
# Usage: bash scripts/manage_snapshots.sh [command]
#
# Commands (non-interactive):
#   save   <name> [description]
#   load   <name>
#   compare <name>
#   list
#   show   <name>
#   delete <name>
#
# Run without arguments for interactive TUI mode.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Color definitions ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# ── Paths ──
SNAPSHOT_DIR="env_saves"
CONFIG_DIR="userconfig/config_in_jsonc"

# Global: set to true in TUI mode
INTERACTIVE=false

# ── Helper: pause (no-op in non-interactive mode) ──
pause() {
    $INTERACTIVE || return 0
    echo ""
    read -p "Press Enter to continue..."
}

# ── Helper: select a snapshot from the list ──
select_snapshot() {
    local snaps=()
    for d in "${SNAPSHOT_DIR}"/*/; do
        [ -d "$d" ] && snaps+=("$(basename "$d")")
    done
    if [ ${#snaps[@]} -eq 0 ]; then
        echo -e "${YELLOW}No snapshots found.${NC}" >&2
        return 1
    fi

    # All display output goes to stderr so it doesn't pollute the return value
    echo -e "${CYAN}Available snapshots:${NC}" >&2
    for i in "${!snaps[@]}"; do
        local meta="${SNAPSHOT_DIR}/${snaps[$i]}/metadata.json"
        local level="?"
        if [ -f "$meta" ]; then
            level=$(python3 -c "import json; print(json.load(open('$meta')).get('snapshot_level','?'))" 2>/dev/null || echo "?")
        fi
        printf "  %3d) %-45s [%s]\n" $((i+1)) "${snaps[$i]}" "$level" >&2
    done
    echo "" >&2

    while true; do
        read -p "Select (1-${#snaps[@]}, or 0 to cancel): " choice
        if [[ ! "$choice" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Please enter a number.${NC}" >&2
            continue
        fi
        choice=$((choice))
        [ "$choice" -eq 0 ] && return 1
        [ "$choice" -ge 1 ] && [ "$choice" -le ${#snaps[@]} ] && {
            # Only the selected name goes to stdout (the return value)
            echo "${snaps[$((choice-1))]}"
            return 0
        }
        echo -e "${RED}Invalid selection.${NC}" >&2
    done
}

# ── 1. List snapshots ──
cmd_list() {
    if [ ! -d "${SNAPSHOT_DIR}" ]; then
        echo -e "${YELLOW}No snapshots found (${SNAPSHOT_DIR}/ does not exist).${NC}"
        return
    fi

    local snaps=()
    for d in "${SNAPSHOT_DIR}"/*/; do
        [ -d "$d" ] && snaps+=("$(basename "$d")")
    done
    if [ ${#snaps[@]} -eq 0 ]; then
        echo -e "${YELLOW}No snapshots found.${NC}"
        return
    fi

    echo -e "${CYAN}Snapshots (${#snaps[@]} total):${NC}"
    echo ""
    printf "  %-45s %-8s %-22s %s\n" "NAME" "LEVEL" "TIMESTAMP" "DESCRIPTION"
    printf "  %-45s %-8s %-22s %s\n" "----" "-----" "---------" "-----------"

    for snap in "${snaps[@]}"; do
        python3 - "$snap" << 'PYEOF' 2>/dev/null || printf "  %-45s %-8s %-22s %s\n" "$1" "?" "(no metadata)" ""
import json, sys
snap = sys.argv[1]
try:
    m = json.load(open(f'env_saves/{snap}/metadata.json'))
    lvl = m.get('snapshot_level', '?')
    ts = m.get('timestamp', '')[:19]
    desc = m.get('description', '')[:40]
except:
    lvl, ts, desc = '?', '', ''
print(f'  {snap:<45} {lvl:<8} {ts:<22} {desc}')
PYEOF
    done
}

# ── 2. Save snapshot (shell level) ──
cmd_save() {
    local name="${1:-}"
    local desc="${2:-}"
    local interactive=false

    if [ -z "$name" ]; then
        interactive=true
        echo -e "${CYAN}--- Save Snapshot (shell level: configs only) ---${NC}"
        read -p "Snapshot name: " name
        [ -z "$name" ] && { echo -e "${RED}Cancelled.${NC}"; return; }
        read -p "Description (optional): " desc
    fi

    mkdir -p "${SNAPSHOT_DIR}/${name}/configs"
    cp -r "${CONFIG_DIR}/"* "${SNAPSHOT_DIR}/${name}/configs/"

    cat > "${SNAPSHOT_DIR}/${name}/metadata.json" << EOF
{
  "snapshot_level": "shell",
  "timestamp": "$(date -Iseconds)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "description": "${desc}"
}
EOF
    [ -n "$desc" ] && echo "$desc" > "${SNAPSHOT_DIR}/${name}/DESCRIPTION.md"

    echo -e "${GREEN}✅ Snapshot saved: ${SNAPSHOT_DIR}/${name}/${NC}"
    $interactive && pause
}

# ── 3. Load snapshot ──
cmd_load() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        name=$(select_snapshot) || { echo -e "${YELLOW}Cancelled.${NC}"; return; }
    fi

    local dir="${SNAPSHOT_DIR}/${name}"
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: snapshot not found: $name${NC}"
        pause
        return
    fi

    # Restore configs
    if [ -d "${dir}/configs" ]; then
        cp -r "${dir}/configs/"* "${CONFIG_DIR}/"
        echo -e "${GREEN}✅ Configs restored to ${CONFIG_DIR}/${NC}"
    else
        echo -e "${YELLOW}⚠️  No configs/ found in snapshot.${NC}"
    fi

    # If runtime metadata exists, compare environments
    if [ -f "${dir}/metadata.runtime.json" ]; then
        echo ""
        echo -e "${CYAN}Snapshot has runtime metadata — checking environment...${NC}"
        python3 << 'PYEOF' 2>/dev/null || true
import json, os, subprocess

snap_dir = os.path.expanduser("'""${dir}""'")
with open(os.path.join(snap_dir, 'metadata.runtime.json')) as f:
    snap = json.load(f)

cur = {}
try:
    r = subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],
                      capture_output=True, text=True, timeout=10)
    cur['gpu'] = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
except:
    cur['gpu'] = []
cur['hostname'] = os.uname().nodename
cur['kernel'] = os.uname().release
cur['slurm_nnodes'] = os.environ.get('SLURM_NNODES', 'N/A')
cur['slurm_ntasks'] = os.environ.get('SLURM_NTASKS', 'N/A')

issues = []
for key in ['gpu', 'hostname', 'kernel', 'slurm_nnodes', 'slurm_ntasks']:
    s = snap.get(key, 'N/A')
    c = cur.get(key, 'N/A')
    if str(s) != str(c) and s != 'N/A':
        issues.append(f"  ⚠️  {key}: snapshot={s}  current={c}")

if issues:
    print("[snapshot] ⚠️  Environment differs from snapshot:")
    for i in issues: print(i)
else:
    print("[snapshot] ✅ Environment matches snapshot.")
PYEOF
        echo ""

        if [ -f "${dir}/effective_config.json" ]; then
            echo -e "${CYAN}Also available: effective_config.json. Use 'compare' for detailed diff.${NC}"
        fi
    elif [ -f "${dir}/effective_config.json" ]; then
        echo -e "${CYAN}ℹ️  effective_config.json found — use 'compare' for detailed config diff.${NC}"
    fi

    echo -e "${GREEN}✅ Load complete.${NC}"
    pause
}

# ── 4. Compare snapshot with current environment ──
cmd_compare() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        name=$(select_snapshot) || { echo -e "${YELLOW}Cancelled.${NC}"; return; }
    fi

    local dir="${SNAPSHOT_DIR}/${name}"
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: snapshot not found: $name${NC}"
        pause
        return
    fi

    if [ ! -f "${dir}/metadata.runtime.json" ] && [ ! -f "${dir}/effective_config.json" ] && [ ! -d "${dir}/configs" ]; then
        echo -e "${YELLOW}Snapshot '$name' is empty — nothing to compare.${NC}"
        pause
        return
    fi

    # Flag: this is a shell-level snapshot (no effective_config/metadata)
    local SHELL_ONLY=false
    [ ! -f "${dir}/metadata.runtime.json" ] && [ ! -f "${dir}/effective_config.json" ] && SHELL_ONLY=true

    python3 << 'PYEOF'
import json, os, subprocess, sys

snap_dir = os.path.expanduser("'""${dir}""'")
snap_name = "'""${name}""'"
shell_only = "'""${SHELL_ONLY}""'" == "true"

if shell_only:
    print("=" * 58)
    print(f"  Shell-level snapshot — config files only")
    print(f"  Snapshot: {snap_name}")
    print("=" * 58)
else:
    # 1. Runtime metadata comparison
    runtime_file = os.path.join(snap_dir, 'metadata.runtime.json')
    if os.path.exists(runtime_file):
        with open(runtime_file) as f:
            snap = json.load(f)
        cur = {}
        try:
            r = subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap,memory.total','--format=csv,noheader'],
                              capture_output=True, text=True, timeout=10)
            cur['gpu'] = [l.strip() for l in r.stdout.strip().split('\n') if l.strip()]
        except: cur['gpu'] = []
        cur['hostname'] = os.uname().nodename
        cur['kernel'] = os.uname().release
        cur['slurm_nnodes'] = os.environ.get('SLURM_NNODES', '')
        cur['slurm_ntasks'] = os.environ.get('SLURM_NTASKS', '')

        print("=" * 58)
        print(f"  Comparing environment with snapshot: {snap_name}")
        print("=" * 58)
        for key in ['gpu', 'hostname', 'kernel', 'slurm_nnodes', 'slurm_ntasks', 'slurm_cpus_on_node']:
            s = snap.get(key, 'N/A')
            c = cur.get(key, 'N/A')
            marker = "⚠️ " if str(s) != str(c) and s != 'N/A' else "✅"
            print(f"  {marker} {key}:")
            print(f"       snapshot: {s}")
            print(f"       current:  {c}")

        # NCCL version
        if 'nccl_version' in snap:
            print(f"  ℹ️  nccl_version (snapshot): {snap['nccl_version']}")
            try:
                import ctypes
                nccl = ctypes.CDLL('libnccl.so')
                ver = ctypes.c_int()
                nccl.ncclGetVersion(ctypes.byref(ver))
                print(f"  ℹ️  nccl_version (current):  {ver.value}")
                if ver.value != snap['nccl_version']:
                    print(f"  ⚠️  NCCL version mismatch!")
            except: print(f"  ℹ️  nccl_version (current):  N/A")
    else:
        print("No runtime metadata in snapshot.")

# 2. Effective config vs current JSONC
eff_file = os.path.join(snap_dir, 'effective_config.json')
if os.path.exists(eff_file):
    print("")
    print("--- Effective config diff ---")
    with open(eff_file) as f:
        eff_snap = json.load(f)
    import re
    def load_jsonc(path):
        with open(path) as f:
            text = f.read()
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(text)

    base = 'userconfig/config_in_jsonc'
    cur_cfg = {}
    try:
        cur_cfg = {
            'basic': load_jsonc(base + '/bench_basic_config.jsonc'),
            'comm':  load_jsonc(base + '/communication_lib_selection.jsonc'),
            'comp':  load_jsonc(base + '/compression_kernel_selection.jsonc'),
        }
    except: pass

    if cur_cfg:
        def deep_diff(a, b, path=""):
            diffs = []
            for k in set(list(a.keys()) + list(b.keys())):
                if k not in a: diffs.append((path + "." + k, "MISSING", b[k]))
                elif k not in b: diffs.append((path + "." + k, a[k], "MISSING"))
                elif isinstance(a[k], dict) and isinstance(b[k], dict):
                    diffs += deep_diff(a[k], b[k], path + "." + k)
                elif a[k] != b[k]: diffs.append((path + "." + k, a[k], b[k]))
            return diffs
        diffs = []
        for section in ['basic', 'comm', 'comp']:
            if section in eff_snap and section in cur_cfg:
                diffs += deep_diff(eff_snap[section], cur_cfg[section], section)
        if diffs:
            print(f"  {len(diffs)} difference(s) (snapshot vs current):")
            for path, sv, cv in diffs[:25]:
                print(f"    {path}: snap={sv}  cur={cv}")
            if len(diffs) > 25: print(f"    ... and {len(diffs)-25} more")
        else: print("  ✅ Configs match snapshot.")
else: print("--- No effective_config.json in snapshot ---")
print("")
PYEOF

    # 3. Configs directory diff (available for all snapshot levels)
    if [ -d "${dir}/configs" ]; then
        echo ""
        echo -e "${CYAN}--- Config file diff (snapshot configs/ vs current) ---${NC}"
        if diff -rq "${dir}/configs" "${CONFIG_DIR}" &>/dev/null; then
            echo -e "${GREEN}  ✅ Config files match current.${NC}"
        else
            echo -e "${YELLOW}  ⚠️  Config files differ:${NC}"
            diff -rq "${dir}/configs" "${CONFIG_DIR}" 2>/dev/null | head -30
        fi
    fi
    pause
}

# ── 5. Show snapshot details ──
cmd_show() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        name=$(select_snapshot) || { echo -e "${YELLOW}Cancelled.${NC}"; return; }
    fi

    local dir="${SNAPSHOT_DIR}/${name}"
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: snapshot not found: $name${NC}"
        pause
        return
    fi

    echo -e "${CYAN}Snapshot: ${name}${NC}"
    echo "  Path: $dir"
    echo ""

    for meta_file in metadata.json metadata.build.json metadata.runtime.json; do
        if [ -f "${dir}/${meta_file}" ]; then
            echo -e "${BLUE}--- ${meta_file} ---${NC}"
            python3 -c "
import json
d = json.load(open('${dir}/${meta_file}'))
print(json.dumps(d, indent=2))
" 2>/dev/null || cat "${dir}/${meta_file}"
            echo ""
        fi
    done

    if [ -f "${dir}/DESCRIPTION.md" ]; then
        echo -e "${BLUE}--- Description ---${NC}"
        cat "${dir}/DESCRIPTION.md"
        echo ""
    fi

    if [ -f "${dir}/effective_config.json" ]; then
        local eff_size=$(wc -c < "${dir}/effective_config.json")
        echo -e "${BLUE}effective_config.json${NC} (${eff_size} bytes)"
    fi
    if [ -d "${dir}/results" ]; then
        local res_count=$(find "${dir}/results" -type f 2>/dev/null | wc -l)
        echo -e "${BLUE}results/${NC} (${res_count} files)"
    fi
    if [ -d "${dir}/perf_files" ]; then
        local perf_count=$(find "${dir}/perf_files" -type f 2>/dev/null | wc -l)
        echo -e "${BLUE}perf_files/${NC} (${perf_count} files)"
    fi
    if [ -d "${dir}/configs" ]; then
        local cfg_count=$(find "${dir}/configs" -type f 2>/dev/null | wc -l)
        echo -e "${BLUE}configs/${NC} (${cfg_count} files)"
    fi

    pause
}

# ── 6. Delete snapshot ──
cmd_delete() {
    local name="${1:-}"
    if [ -z "$name" ]; then
        echo -e "${YELLOW}--- Delete Snapshot ---${NC}"
        echo -e "${RED}WARNING: This permanently removes the snapshot!${NC}"
        name=$(select_snapshot) || { echo -e "${YELLOW}Cancelled.${NC}"; return; }
    fi

    local dir="${SNAPSHOT_DIR}/${name}"
    if [ ! -d "$dir" ]; then
        echo -e "${RED}Error: snapshot not found: $name${NC}"
        pause
        return
    fi

    echo ""
    echo -e "${YELLOW}About to delete: ${dir}/${NC}"
    local confirm="y"
    if $INTERACTIVE; then
        read -p "Are you sure? (y/N): " confirm
        confirm="${confirm,,}"
    fi
    if [ "$confirm" = "y" ]; then
        rm -rf "$dir"
        echo -e "${GREEN}✅ Deleted: ${name}${NC}"
    else
        echo -e "${YELLOW}Cancelled.${NC}"
    fi
    pause
}

# ── Menu ──
show_menu() {
    clear
    echo "============================================================"
    echo -e "              ${BLUE}Snapshot Management Tool${NC}"
    echo "============================================================"
    echo ""

    # Count snapshots
    local count=0
    [ -d "${SNAPSHOT_DIR}" ] && count=$(find "${SNAPSHOT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    echo -e "  Snapshots: ${CYAN}${count}${NC}"
    echo ""

    echo -e "${BLUE}Options:${NC}"
    echo "  1) List snapshots"
    echo "  2) Save snapshot (shell level — backup config_in_jsonc/)"
    echo "  3) Load snapshot (restore configs, compare if metadata exists)"
    echo "  4) Compare snapshot with current environment"
    echo "  5) Show snapshot details"
    echo "  6) Delete snapshot"
    echo "  7) Exit"
    echo ""
    echo "============================================================"
}

# ── Main ──
main() {
    # If command-line args provided, run non-interactively
    if [ $# -gt 0 ]; then
        case "$1" in
            list)    cmd_list ;;
            save)    cmd_save "${2:-}" "${3:-}" ;;
            load)    cmd_load "${2:-}" ;;
            compare) cmd_compare "${2:-}" ;;
            show)    cmd_show "${2:-}" ;;
            delete)  cmd_delete "${2:-}" ;;
            help|--help|-h)
                echo "Usage: $0 [command]"
                echo "  list              — list snapshots"
                echo "  save <name> [desc]— create shell-level snapshot"
                echo "  load <name>       — restore configs from snapshot"
                echo "  compare <name>    — compare env with snapshot"
                echo "  show <name>       — show snapshot details"
                echo "  delete <name>     — remove a snapshot"
                echo ""
                echo "Run without arguments for interactive TUI mode."
                ;;
            *)
                echo -e "${RED}Unknown command: $1${NC}"
                echo "Usage: $0 [command] or run without args for interactive mode"
                exit 1
                ;;
        esac
        exit 0
    fi

    # Interactive TUI mode
    INTERACTIVE=true
    while true; do
        show_menu
        read -p "Enter choice (1-7): " choice
        case "$choice" in
            1) cmd_list; pause ;;
            2) cmd_save ;;
            3) cmd_load ;;
            4) cmd_compare ;;
            5) cmd_show ;;
            6) cmd_delete ;;
            7) echo "Exiting."; exit 0 ;;
            *) echo -e "${RED}Invalid option.${NC}"; pause ;;
        esac
    done
}

main "$@"
