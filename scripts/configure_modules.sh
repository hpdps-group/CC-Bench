#!/bin/bash

# 检查是否被 source 执行
if [[ "$0" == "$BASH_SOURCE" ]]; then
    echo "错误: 请用 'source configure_modules.sh' 执行此脚本"
    echo "Usage: source configure_modules.sh"
    exit 1
fi

# 颜色定义（可选）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 全局变量
SELECTED_MODULES=()

# 函数：获取当前加载的模块
get_loaded_modules() {
    # 解析 module list 输出，提取模块名
    module list 2>&1 | grep -v "Currently Loaded Modulefiles:" | grep -v "^$" | \
    while read line; do
        # 使用正则表达式提取所有 "数字) 模块名" 的模式
        # 例如: "  1) compiler/dtk/22.04.2        3) mpi/hpcx/gcc-7.3.1"
        while [[ $line =~ [[:space:]]*([0-9]+)\)[[:space:]]+([^[:space:]]+) ]]; do
            echo "${BASH_REMATCH[2]}"
            # 移除已匹配的部分，继续匹配
            line="${line#*${BASH_REMATCH[2]}}"
        done
    done
}

# 函数：获取可用模块
get_available_modules() {
    local pattern="$1"
    # 获取所有可用模块
    module avail 2>&1 | grep -v "^--" | grep -v "^$" | while read line; do
        # 跳过标题行
        if [[ $line =~ ^[[:space:]]*$ ]] || [[ $line =~ ^- ]]; then
            continue
        fi

        # 分割行中的多个模块名
        for module in $line; do
            # 有效的模块名包含斜杠且不是括号
            if [[ $module == */* ]] && [[ $module != "("* ]] && [[ $module != *")" ]]; then
                echo "$module"
            fi
        done
    done | sort -u
}

# 函数：从列表中选择
select_from_list() {
    local items=("${@}")
    local count=${#items[@]}

    if [[ $count -eq 0 ]]; then
        echo "没有可选项" >&2
        return 1
    fi

    # 显示选项（输出到stderr，避免被命令替换捕获）
    for i in "${!items[@]}"; do
        printf "  %3d) %s\n" $((i+1)) "${items[$i]}" >&2
    done

    while true; do
        read -p "请选择 (1-$count, 或 0 取消): " choice

        # 检查是否为数字
        if [[ ! $choice =~ ^[0-9]+$ ]]; then
            echo "请输入数字" >&2
            continue
        fi

        choice=$((choice))

        if [[ $choice -eq 0 ]]; then
            echo "已取消" >&2
            return 1
        fi

        if [[ $choice -ge 1 ]] && [[ $choice -le $count ]]; then
            echo "${items[$((choice-1))]}"
            return 0
        fi

        echo "无效的选择" >&2
    done
}

# 函数：搜索并添加模块
search_and_add_module() {
    read -p "输入搜索模式 (直接回车显示所有): " pattern

    echo "搜索模块中..."

    # 获取所有模块
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
        echo "未找到匹配的模块"
        read -p "按回车继续..."
        return
    fi

    echo "找到 ${#matched_modules[@]} 个模块:"

    # 选择模块
    local selected_module
    selected_module=$(select_from_list "${matched_modules[@]}")

    if [[ $? -ne 0 ]]; then
        return
    fi

    # 检查冲突
    local conflicts=()
    for existing in "${SELECTED_MODULES[@]}"; do
        # 简单的冲突检测：相同前缀可能冲突
        local prefix1="${selected_module%%/*}"
        local prefix2="${existing%%/*}"
        if [[ "$prefix1" == "$prefix2" ]]; then
            conflicts+=("$existing")
        fi
    done

    if [[ ${#conflicts[@]} -gt 0 ]]; then
        echo -e "${YELLOW}警告: 模块 $selected_module 可能与以下模块冲突:${NC}"
        for conflict in "${conflicts[@]}"; do
            echo "  - $conflict"
        done

        read -p "是否替换冲突模块? (y/N): " replace
        if [[ ${replace,,} == "y" ]]; then
            # 移除冲突模块
            for conflict in "${conflicts[@]}"; do
                SELECTED_MODULES=("${SELECTED_MODULES[@]/$conflict}")
                # 重新构建数组，移除空元素
                local new_array=()
                for item in "${SELECTED_MODULES[@]}"; do
                    if [[ -n "$item" ]]; then
                        new_array+=("$item")
                    fi
                done
                SELECTED_MODULES=("${new_array[@]}")
            done
            SELECTED_MODULES+=("$selected_module")
            echo -e "${GREEN}已添加模块 $selected_module (替换了冲突模块)${NC}"

            # 询问是否立即执行替换
            read -p "是否立即执行替换（卸载冲突模块，加载新模块）? (Y/n): " execute_now
            if [[ -z "$execute_now" ]] || [[ ${execute_now,,} == "y" ]]; then
                # 先卸载冲突模块
                for conflict in "${conflicts[@]}"; do
                    echo -n "卸载 $conflict... "
                    if module unload "$conflict" 2>&1; then
                        echo -e "${GREEN}成功${NC}"
                    else
                        echo -e "${RED}失败${NC}"
                    fi
                done
                # 加载新模块
                echo -n "加载 $selected_module... "
                if module load "$selected_module" 2>&1; then
                    echo -e "${GREEN}成功${NC}"
                else
                    echo -e "${RED}失败${NC}"
                fi
            fi
        else
            echo "未添加模块"
        fi
    else
        SELECTED_MODULES+=("$selected_module")
        echo -e "${GREEN}已添加模块 $selected_module${NC}"
    fi

    # 询问是否立即加载
    read -p "是否立即加载此模块? (Y/n): " load_now
    if [[ -z "$load_now" ]] || [[ ${load_now,,} == "y" ]]; then
        echo -n "加载 $selected_module... "
        if module load "$selected_module" 2>&1; then
            echo -e "${GREEN}成功${NC}"
        else
            echo -e "${RED}失败${NC}"
        fi
    fi

    read -p "按回车继续..."
}

# 函数：删除模块
remove_selected_module() {
    if [[ ${#SELECTED_MODULES[@]} -eq 0 ]]; then
        echo "当前没有已选模块"
        read -p "按回车继续..."
        return
    fi

    echo "选择要删除的模块:"
    local selected_module
    selected_module=$(select_from_list "${SELECTED_MODULES[@]}")

    if [[ $? -ne 0 ]]; then
        return
    fi

    # 从数组中移除
    local new_array=()
    for module in "${SELECTED_MODULES[@]}"; do
        if [[ "$module" != "$selected_module" ]]; then
            new_array+=("$module")
        fi
    done
    SELECTED_MODULES=("${new_array[@]}")

    echo -e "${GREEN}已从计划中移除模块 $selected_module${NC}"

    # 询问是否立即卸载
    read -p "是否立即从环境中卸载此模块? (Y/n): " unload_now
    if [[ -z "$unload_now" ]] || [[ ${unload_now,,} == "y" ]]; then
        echo -n "卸载 $selected_module... "
        if module unload "$selected_module" 2>&1; then
            echo -e "${GREEN}成功${NC}"
        else
            echo -e "${RED}失败${NC}"
        fi
    fi

    read -p "按回车继续..."
}

# 函数：显示所有可用模块
show_all_modules() {
    echo "获取所有可用模块中（可能需要几秒钟）..."
    local all_modules=($(get_available_modules))

    if [[ ${#all_modules[@]} -eq 0 ]]; then
        echo "没有可用模块"
    else
        echo "可用模块 (${#all_modules[@]} 个):"
        for i in "${!all_modules[@]}"; do
            printf "  %3d) %s\n" $((i+1)) "${all_modules[$i]}"
            if [[ $(( (i+1) % 20 )) -eq 0 ]] && [[ $((i+1)) -lt ${#all_modules[@]} ]]; then
                read -p "按回车显示更多，或按 Ctrl+C 中断..."
            fi
        done
    fi

    read -p "按回车继续..."
}

# 函数：显示菜单
show_menu() {
    clear
    echo "============================================================"
    echo "                  MODULE 配置工具"
    echo "============================================================"

    # 显示计划加载的模块
    echo -e "${BLUE}计划加载的模块（选择5应用）:${NC}"
    if [[ ${#SELECTED_MODULES[@]} -gt 0 ]]; then
        for i in "${!SELECTED_MODULES[@]}"; do
            echo "  $((i+1))) ${SELECTED_MODULES[$i]}"
        done
    else
        echo "  (无)"
    fi

    echo ""
    echo -e "${BLUE}选项:${NC}"
    echo "  1) 添加模块"
    echo "  2) 删除模块"
    echo "  3) 搜索模块"
    echo "  4) 显示所有可用模块"
    echo "  5) 应用选中的模块"
    echo "  6) 退出"
    echo ""
    echo "============================================================"
}

# 函数：应用选中的模块
apply_modules() {
    echo "应用选中的模块..."

    # 先显示当前加载的模块
    echo "当前加载的模块:"
    module list 2>&1

    echo ""
    echo "将加载以下模块:"
    for module in "${SELECTED_MODULES[@]}"; do
        echo "  module load $module"
    done

    read -p "确定要加载这些模块吗? (y/N): " confirm
    if [[ ${confirm,,} != "y" ]]; then
        echo "已取消"
        read -p "按回车继续..."
        return
    fi

    # 实际加载模块
    for module in "${SELECTED_MODULES[@]}"; do
        echo -n "加载 $module... "
        if module load "$module" 2>&1; then
            echo -e "${GREEN}成功${NC}"
        else
            echo -e "${RED}失败${NC}"
        fi
    done

    echo ""
    echo "模块加载完成。"
    read -p "按回车继续..."
}

# 主函数
main() {
    # 初始化：从当前加载的模块开始
    echo "正在初始化..."
    echo "提示: 本工具维护一个模块加载计划列表。"
    echo "      - 添加/删除模块时可以选择立即执行"
    echo "      - 或选择选项5批量应用所有计划模块"
    echo ""
    SELECTED_MODULES=($(get_loaded_modules))

    while true; do
        show_menu

        read -p "请输入选项 (1-6): " choice

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
                echo "退出。"
                if [[ ${#SELECTED_MODULES[@]} -gt 0 ]]; then
                    echo "计划加载的模块（尚未实际加载）:"
                    for module in "${SELECTED_MODULES[@]}"; do
                        echo "  module load $module"
                    done
                    echo ""
                    echo "提示: 这些模块尚未加载到环境中。"
                    echo "      - 如需加载，请重新运行脚本并选择选项5"
                    echo "      - 或手动执行上述 module load 命令"
                else
                    echo "没有计划加载的模块。"
                fi
                break
                ;;
            *)
                echo "无效选项"
                read -p "按回车继续..."
                ;;
        esac
    done
}

# 运行主函数
main