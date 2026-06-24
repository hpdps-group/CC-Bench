#!/bin/bash

# Check if sourced
if [[ "$0" == "$BASH_SOURCE" ]]; then
    echo "Error: This script must be sourced. Usage: source configure_modules.sh"
    echo "Usage: source configure_modules.sh"
    exit 1
fi

# Color definitions (optional)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
SELECTED_MODULES=()

# Function: get currently loaded modules
get_loaded_modules() {
    # Parse module list output, extract module names
    module list 2>&1 | grep -v "Currently Loaded Modulefiles:" | grep -v "^$" | \
    while read line; do
        # Use regex to extract all "number) modulename" patterns
        # e.g., "  1) compiler/dtk/22.04.2        3) mpi/hpcx/gcc-7.3.1"
        while [[ $line =~ [[:space:]]*([0-9]+)\)[[:space:]]+([^[:space:]]+) ]]; do
            echo "${BASH_REMATCH[2]}"
            # Remove matched part and continue
            line="${line#*${BASH_REMATCH[2]}}"
        done
    done
}

# Function: get available modules
get_available_modules() {
    local pattern="$1"
    # Get all available modules
    module avail 2>&1 | grep -v "^--" | grep -v "^$" | while read line; do
        # Skip header lines
        if [[ $line =~ ^[[:space:]]*$ ]] || [[ $line =~ ^- ]]; then
            continue
        fi

        # Split multiple module names on a line
        for module in $line; do
            # Valid module names contain a slash and are not parenthesized
            if [[ $module == */* ]] && [[ $module != "("* ]] && [[ $module != *")" ]]; then
                echo "$module"
            fi
        done
    done | sort -u
}

# Function: select from a list
select_from_list() {
    local items=("${@}")
    local count=${#items[@]}

    if [[ $count -eq 0 ]]; then
        echo "No items available" >&2
        return 1
    fi

    # Display options (to stderr to avoid being captured by command substitution)
    for i in "${!items[@]}"; do
        printf "  %3d) %s\n" $((i+1)) "${items[$i]}" >&2
    done

    while true; do
        read -p "Select (1-$count, or 0 to cancel): " choice

        # Check if numeric
        if [[ ! $choice =~ ^[0-9]+$ ]]; then
            echo "Please enter a number" >&2
            continue
        fi

        choice=$((choice))

        if [[ $choice -eq 0 ]]; then
            echo "Cancelled" >&2
            return 1
        fi

        if [[ $choice -ge 1 ]] && [[ $choice -le $count ]]; then
            echo "${items[$((choice-1))]}"
            return 0
        fi

        echo "Invalid selection" >&2
    done
}

# Function: search and add module
search_and_add_module() {
    read -p "Enter search pattern (press Enter to show all): " pattern

    echo "Searching modules..."

    # Get all modules
    local all_modules=($(get_available_modules))
    local matched_modules=()

    if [[ -z "$pattern" ]]; then
        matched_modules=("${all_modules[@]}")
    else
        for module in "${all_modules[@]}"; do
            if [[ $module =~ $pattern ]] || [[ ${module,,} =~ ${pattern,,} ]]; then
                matched_modules+=("$module")
            fi
        done
    fi

    if [[ ${#matched_modules[@]} -eq 0 ]]; then
        echo "No matching modules found"
        read -p "Press Enter to continue..."
        return
    fi

    echo "Found ${#matched_modules[@]} modules:"

    # Select module
    local selected_module
    selected_module=$(select_from_list "${matched_modules[@]}")

    if [[ $? -ne 0 ]]; then
        return
    fi

    # Check conflicts
    local conflicts=()
    for existing in "${SELECTED_MODULES[@]}"; do
        # Simple conflict detection: same prefix may conflict
        local prefix1="${selected_module%%/*}"
        local prefix2="${existing%%/*}"
        if [[ "$prefix1" == "$prefix2" ]]; then
            conflicts+=("$existing")
        fi
    done

    if [[ ${#conflicts[@]} -gt 0 ]]; then
        echo -e "${YELLOW}Warning: Module $selected_module may conflict with:${NC}"
        for conflict in "${conflicts[@]}"; do
            echo "  - $conflict"
        done

        read -p "Replace conflicting modules? (y/N): " replace
        if [[ ${replace,,} == "y" ]]; then
            # Remove conflicting modules
            for conflict in "${conflicts[@]}"; do
                SELECTED_MODULES=("${SELECTED_MODULES[@]/$conflict}")
                # Rebuild array, remove empty elements
                local new_array=()
                for item in "${SELECTED_MODULES[@]}"; do
                    if [[ -n "$item" ]]; then
                        new_array+=("$item")
                    fi
                done
                SELECTED_MODULES=("${new_array[@]}")
            done
            SELECTED_MODULES+=("$selected_module")
            echo -e "${GREEN}Added module $selected_module (replaced conflicting modules)${NC}"

            # Ask whether to execute replacement immediately
            read -p "Apply replacement now (unload conflicts, load new module)? (Y/n): " execute_now
            if [[ -z "$execute_now" ]] || [[ ${execute_now,,} == "y" ]]; then
                # Unload conflicting modules first
                for conflict in "${conflicts[@]}"; do
                    echo -n "Unloading $conflict... "
                    if module unload "$conflict" 2>&1; then
                        echo -e "${GREEN}OK${NC}"
                    else
                        echo -e "${RED}Failed${NC}"
                    fi
                done
                # Load new module
                echo -n "Loading $selected_module... "
                if module load "$selected_module" 2>&1; then
                    echo -e "${GREEN}OK${NC}"
                else
                    echo -e "${RED}Failed${NC}"
                fi
            fi
        else
            echo "Module not added"
        fi
    else
        SELECTED_MODULES+=("$selected_module")
        echo -e "${GREEN}Added module $selected_module${NC}"
    fi

    # Ask whether to load immediately
    read -p "Load this module now? (Y/n): " load_now
    if [[ -z "$load_now" ]] || [[ ${load_now,,} == "y" ]]; then
        echo -n "Loading $selected_module... "
        if module load "$selected_module" 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}Failed${NC}"
        fi
    fi

    read -p "Press Enter to continue..."
}

# Function: remove module
remove_selected_module() {
    if [[ ${#SELECTED_MODULES[@]} -eq 0 ]]; then
        echo "No modules currently selected"
        read -p "Press Enter to continue..."
        return
    fi

    echo "Select module to remove:"
    local selected_module
    selected_module=$(select_from_list "${SELECTED_MODULES[@]}")

    if [[ $? -ne 0 ]]; then
        return
    fi

    # Remove from array
    local new_array=()
    for module in "${SELECTED_MODULES[@]}"; do
        if [[ "$module" != "$selected_module" ]]; then
            new_array+=("$module")
        fi
    done
    SELECTED_MODULES=("${new_array[@]}")

    echo -e "${GREEN}Removed module $selected_module from plan${NC}"

    # Ask whether to unload immediately
    read -p "Unload this module from environment now? (Y/n): " unload_now
    if [[ -z "$unload_now" ]] || [[ ${unload_now,,} == "y" ]]; then
        echo -n "Unloading $selected_module... "
        if module unload "$selected_module" 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}Failed${NC}"
        fi
    fi

    read -p "Press Enter to continue..."
}

# Function: show all available modules
show_all_modules() {
    echo "Fetching all available modules (may take a few seconds)..."
    local all_modules=($(get_available_modules))

    if [[ ${#all_modules[@]} -eq 0 ]]; then
        echo "No modules available"
    else
        echo "Available modules (${#all_modules[@]}):"
        for i in "${!all_modules[@]}"; do
            printf "  %3d) %s\n" $((i+1)) "${all_modules[$i]}"
            if [[ $(( (i+1) % 20 )) -eq 0 ]] && [[ $((i+1)) -lt ${#all_modules[@]} ]]; then
                read -p "Press Enter for more, or Ctrl+C to abort..."
            fi
        done
    fi

    read -p "Press Enter to continue..."
}

# Function: display menu
show_menu() {
    clear
    echo "============================================================"
    echo "                  MODULE Configuration Tool"
    echo "============================================================"

    # Show planned modules
    echo -e "${BLUE}Planned modules (select 5 to apply):${NC}"
    if [[ ${#SELECTED_MODULES[@]} -gt 0 ]]; then
        for i in "${!SELECTED_MODULES[@]}"; do
            echo "  $((i+1))) ${SELECTED_MODULES[$i]}"
        done
    else
        echo "  (none)"
    fi

    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  1) Add module"
    echo "  2) Delete module"
    echo "  3) Search modules"
    echo "  4) Show all available modules"
    echo "  5) Apply selected modules"
    echo "  6) Exit"
    echo ""
    echo "============================================================"
}

# Function: apply selected modules
apply_modules() {
    echo "Applying selected modules..."

    # Show current loaded modules
    echo "Currently loaded modules:"
    module list 2>&1

    echo ""
    echo "Will load the following modules:"
    for module in "${SELECTED_MODULES[@]}"; do
        echo "  module load $module"
    done

    read -p "Confirm loading these modules? (y/N): " confirm
    if [[ ${confirm,,} != "y" ]]; then
        echo "Cancelled"
        read -p "Press Enter to continue..."
        return
    fi

    # Actually load modules
    for module in "${SELECTED_MODULES[@]}"; do
        echo -n "Loading $module... "
        if module load "$module" 2>&1; then
            echo -e "${GREEN}OK${NC}"
        else
            echo -e "${RED}Failed${NC}"
        fi
    done

    echo ""
    echo "Module loading complete."
    read -p "Press Enter to continue..."
}

# Main function
main() {
    # Initialize: start with currently loaded modules
    echo "Initializing..."
    echo "Note: This tool maintains a module load plan list."
    echo "      - Add/remove modules with option to execute immediately"
    echo "      - Or select option 5 to batch apply all planned modules"
    echo ""
    SELECTED_MODULES=($(get_loaded_modules))

    while true; do
        show_menu

        read -p "Enter choice (1-6): " choice

        case $choice in
            1)
                search_and_add_module
                ;;
            2)
                remove_selected_module
                ;;
            3)
                search_and_add_module
                ;;
            4)
                show_all_modules
                ;;
            5)
                apply_modules
                ;;
            6)
                echo "Exiting."
                if [[ ${#SELECTED_MODULES[@]} -gt 0 ]]; then
                    echo "Planned modules (not yet loaded):"
                    for module in "${SELECTED_MODULES[@]}"; do
                        echo "  module load $module"
                    done
                    echo ""
                    echo "Note: These modules have not been loaded into the environment."
                    echo "      - Re-run the script and select option 5 to load them"
                    echo "      - Or manually run the module load commands above"
                else
                    echo "No planned modules."
                fi
                break
                ;;
            *)
                echo "Invalid option"
                read -p "Press Enter to continue..."
                ;;
        esac
    done
}

# Run main function
main