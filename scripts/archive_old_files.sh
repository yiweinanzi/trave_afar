#!/bin/bash
#
# GoAfar 项目过期代码归档脚本
# 只归档在 gaozao.md (2025-11-09) 实施之前创建的过期文件
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 项目根目录
PROJECT_ROOT="/root/autodl-tmp/goafar_project"
cd "$PROJECT_ROOT"

# 归档基准时间：gaozao.md 创建时间 2025-11-09
ARCHIVE_CUTOFF="2025-11-09"

# 归档目录
ARCHIVE_DIR="$PROJECT_ROOT/.archive_before_gaozao"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log_info "=========================================="
log_info "GoAfar 过期文件归档脚本"
log_info "归档基准时间: $ARCHIVE_CUTOFF"
log_info "归档目录: $ARCHIVE_DIR"
log_info "时间戳: $TIMESTAMP"
log_info "=========================================="

# 创建归档目录结构
mkdir -p "$ARCHIVE_DIR"/{docs_old,code_old,outputs_old,logs_old,config_old}
log_info "归档目录结构已创建"

# 获取文件修改时间 (YYYY-MM-DD HH:MM:SS)
get_file_date() {
    stat -c "%y %n" "$1" 2>/dev/null || echo "9999-12-31 23:59:59"
}

# 比较文件是否在归档基准之前
is_older_than_cutoff() {
    local file_date=$(get_file_date "$1")
    local file_yyyy=${file_date:0:4}
    local file_mm=${file_date:5:2}
    local file_dd=${file_date:8:2}

    # 拼接为 YYYYMMDD
    local file_num=$(printf "%s%02d%02d" "$file_yyyy" "$file_mm" "$file_dd")
    local cutoff_num="20251109"

    # echo "DEBUG: $1 -> $file_num vs $cutoff_num" >&2
    [ "$file_num" -lt "$cutoff_num" ]
}

# 统计变量
MOVED_COUNT=0
KEPT_COUNT=0
ERROR_COUNT=0

# =====================
# 第一阶段：归档过期文档
# =====================
log_info ""
log_info "=== 第一阶段：归档过期文档 ==="

# 检查 review_docs/ 目录
if [ -d "review_docs" ]; then
    file_count=$(find review_docs -type f -name "*.md" 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        log_info "归档 review_docs/ ($file_count 个文件)"
        mkdir -p "$ARCHIVE_DIR/docs_old/review_docs"

        for file in review_docs/*.md; do
            if [ -f "$file" ]; then
                if mv "$file" "$ARCHIVE_DIR/docs_old/review_docs/"; then
                    ((MOVED_COUNT++))
                else
                    ((ERROR_COUNT++))
                    log_error "移动失败: $file"
                fi
            fi
        done
    else
        log_warn "review_docs/ 目录为空或不存在"
    fi
else
    log_warn "review_docs/ 目录不存在"
fi

# 检查 .specstory/ 目录
if [ -d ".specstory" ]; then
    file_count=$(find .specstory -type f 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        log_info "归档 .specstory/ ($file_count 个文件)"
        mv .specstory "$ARCHIVE_DIR/docs_old/" 2>/dev/null && ((MOVED_COUNT++))
    fi
fi

# 归档根目录下的旧文档 (排除当前使用的)
OLD_DOCS=(
    "GPU优化说明.md"
    "LLM4REC_IMPLEMENTATION.md"
    "项目改进基础文档.md"
    "最终交付报告.md"
    "最终交付清单.md"
    "最终交付报告-更新版.md"
)

for doc in "${OLD_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        file_date=$(get_file_date "$doc")
        file_yyyy=${file_date:0:4}
        if [ "$file_yyyy" -lt "2025" ]; then
            log_info "归档文档: $doc"
            mv "$doc" "$ARCHIVE_DIR/docs_old/" 2>/dev/null && ((MOVED_COUNT++))
        fi
    fi
done

# =====================
# 第二阶段：归档过期代码
# =====================
log_info ""
log_info "=== 第二阶段：归档过期代码 ==="

# 这些文件已被新架构替代
OLD_CODE_FILES=(
    # 如果存在旧版本的 web_ui 实现
    "web_ui.py"
    "demo_ui.py"
    "simple_gradio.py"
)

# 检查 open_resource/ 中的第三方库 (不需要提交到项目本身)
if [ -d "open_resource" ]; then
    # open_resource 包含第三方库，不应该删除
    # 但可以标记为不需要版本控制
    log_info "跳过 open_resource/ (第三方库目录)"
fi

# =====================
# 第三阶段：归档日志和临时文件
# =====================
log_info ""
log_info "=== 第三阶段：归档日志和临时文件 ==="

# 归档 context/logs/ 如果存在
if [ -d "context/logs" ]; then
    file_count=$(find context/logs -type f 2>/dev/null | wc -l)
    if [ "$file_count" -gt 0 ]; then
        log_info "归档 context/logs/ ($file_count 个文件)"
        mv context/logs "$ARCHIVE_DIR/logs_old/" 2>/dev/null && ((MOVED_COUNT++))
    fi
fi

# 归档 *.log 文件
log_count=$(find . -maxdepth 1 -type f -name "*.log" 2>/dev/null | wc -l)
if [ "$log_count" -gt 0 ]; then
    log_info "归档根目录 *.log 文件 ($log_count 个)"
    find . -maxdepth 1 -type f -name "*.log" -exec mv {} "$ARCHIVE_DIR/logs_old/" \; 2>/dev/null
    ((MOVED_COUNT += log_count))
fi

# 归档 __pycache__
if [ -d "src/__pycache__" ]; then
    log_info "归档 src/__pycache__/"
    rm -rf src/__pycache__
fi

# =====================
# 第四阶段：创建归档元数据和恢复脚本
# =====================
log_info ""
log_info "=== 第四阶段：创建归档元数据 ==="

# 创建归档信息
cat > "$ARCHIVE_DIR/archive_info.json" << EOF
{
  "archive_date": "$(date -Iseconds)",
  "archive_timestamp": "$TIMESTAMP",
  "project_name": "GoAfar",
  "cutoff_date": "$ARCHIVE_CUTOFF",
  "description": "Files created before gaozao.md implementation (2025-11-09)",
  "moved_files": $MOVED_COUNT,
  "categories": {
    "docs_old": "Documentation from before gaozao.md",
    "code_old": "Code replaced by new architecture",
    "outputs_old": "Model outputs and training results",
    "logs_old": "Log files and temporary data"
  },
  "recovery_script": "scripts/restore_from_archive.sh",
  "important": [
    "DO NOT delete archive directory",
    "Use restore_from_archive.sh to recover files",
    "Current active code is in src/, configs/, data/"
  ]
}
EOF

log_info "归档元数据已创建: $ARCHIVE_DIR/archive_info.json"

# 创建恢复脚本
cat > scripts/restore_from_archive.sh << 'RESTORESCRIPT'
#!/bin/bash
#
# GoAfar 归档恢复脚本
# 用于从 .archive_before_gaozao 恢复文件
#

set -e

GREEN='\033[0;32m'
NC='\033[0m'

PROJECT_ROOT="/root/autodl-tmp/goafar_project"
ARCHIVE_DIR="$PROJECT_ROOT/.archive_before_gaozao"

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

case "${1:-list}" in
    "docs"|"all")
        log_info "恢复文档..."
        cp -r "$ARCHIVE_DIR/docs_old/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "文档已恢复"
        ;;
    "code"|"all")
        log_info "恢复代码..."
        cp -r "$ARCHIVE_DIR/code_old/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "代码已恢复"
        ;;
    "list")
        log_info "归档内容列表:"
        echo ""
        echo "文档:"
        find "$ARCHIVE_DIR/docs_old" -type f 2>/dev/null || echo "  (空)"
        echo ""
        echo "代码:"
        find "$ARCHIVE_DIR/code_old" -type f 2>/dev/null || echo "  (空)"
        echo ""
        echo "日志:"
        find "$ARCHIVE_DIR/logs_old" -type f 2>/dev/null || echo "  (空)"
        ;;
    *)
        echo "用法: $0 {docs|code|all|list}"
        echo ""
        echo "命令:"
        echo "  docs  - 恢复归档的文档"
        echo "  code  - 恢复归档的代码"
        echo "  all   - 恢复所有内容"
        echo "  list  - 列出归档内容"
        exit 1
        ;;
esac
RESTORESCRIPT

chmod +x scripts/restore_from_archive.sh
log_info "恢复脚本已创建: scripts/restore_from_archive.sh"

# =====================
# 第五阶段：更新 .gitignore
# =====================
log_info ""
log_info "=== 第五阶段：更新 .gitignore ==="

if ! grep -q "^\.archive_before_gaozao/" .gitignore 2>/dev/null; then
    echo "" >> .gitignore
    echo "# 归档目录 (gaozao.md 实施前的过期文件)" >> .gitignore
    echo ".archive_before_gaozao/" >> .gitignore
    log_info ".gitignore 已更新"
else
    log_info ".gitignore 已包含归档目录"
fi

if ! grep -q "^context/logs/\*" .gitignore 2>/dev/null; then
    echo "context/logs/*" >> .gitignore 2>/dev/null
fi

# =====================
# 完成统计
# =====================
log_info ""
log_info "=========================================="
log_info "归档完成统计:"
log_info "  移动文件数: $MOVED_COUNT"
log_info "  错误数: $ERROR_COUNT"
log_info "  归档目录: $ARCHIVE_DIR"
log_info "=========================================="
log_info ""
log_info "下一步操作:"
log_info "  1. 检查归档内容: ls -la $ARCHIVE_DIR"
log_info "  2. 如需恢复: bash scripts/restore_from_archive.sh list"
log_info "  3. 提交到GitHub: git add . && git commit -m 'archive: 归档过期代码和文档'"
log_info "  4. 推送: git push origin main"
