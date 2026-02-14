#!/bin/bash
#
# GoAfar 项目归档脚本
# 只归档在 gaozao.md (2025-11-09) 实施之前的过期文件
#
set -e

# 颜色
GREEN='\033[0;32m'
NC='\033[0m'
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

# 项目目录
PROJECT_ROOT="/root/autodl-tmp/goafar_project"
cd "$PROJECT_ROOT"

# 归档目录
ARCHIVE_DIR=".archive_before_gaozao_20251109"

# 被删除的过期文件（在 gaozao.md 实施前）
declare -A DELETED_DOCS=(
    # review_docs/ 目录的内容（已在 7786c92 被清空）
    "review_docs/00_项目概述.md"
    "review_docs/01_技术架构_BGE-M3.md"
    "review_docs/02_技术架构_RecBole.md"
    "review_docs/03_技术架构_LLM4Rec.md"
    "review_docs/04_技术场景_VRPTW.md"
    "review_docs/05_技术架构_DPO.md"
    "review_docs/06_技术架构_评测系统.md"
    "review_docs/07_技术架构_缓存优化.md"
    "review_docs/08_技术架构_多日规划.md"
    "review_docs/09_后续TODO规划.md"
    "review_docs/README.md"
    "review_docs/升级说明.md"
    "review_docs/最终交付清单.md"
    "review_docs/最终交付报告.md"
    "review_docs/最终交付报告-更新.md"
)

declare -A DELETED_OTHER=(
    # 其他过期文档
    "GPU优化说明.md"
    "LLM4REC_IMPLEMENTATION.md"
    "项目改进基础文档.md"
    "项目完整文档.md"
    "项目完整文档-2024.md"
    "项目完整文档-更新.md"
    "历史-面试问答.md"
    "历史-面试问答-更新.md"
    "最终交付清单.md"
    "最终交付报告.md"
    "最终交付报告-更新.md"
)

log_info "=========================================="
log_info "GoAfar 项目归档 (gaozao.md 实施前的过期文件)"
log_info "归档目录: $ARCHIVE_DIR"
log_info "=========================================="

# 创建归档目录
mkdir -p "$ARCHIVE_DIR/docs_old"
log_info "已创建归档目录"

# 移动文件
MOVED_COUNT=0

# 1. 处理已删除的文档
for file in "${DELETED_DOCS[@]}"; do
    if [ -f "$file" ]; then
        # 使用 git show 恢复文件内容（如果存在）
        if git show "HEAD:$file" >/dev/null 2>&1; then
            git show "HEAD:$file" > "$ARCHIVE_DIR/$file"
            log_info "✓ 归档: $file"
            ((MOVED_COUNT++))
        fi
    fi
done

# 2. 处理其他过期文档
for file in "${DELETED_OTHER[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$ARCHIVE_DIR/"
        log_info "✓ 归档: $file"
        ((MOVED_COUNT++))
    fi
done

# 3. 处理 context/logs（如果存在）
if [ -d "context/logs" ]; then
    mv context/logs "$ARCHIVE_DIR/"
    log_info "✓ 归档: context/logs/"
    ((MOVED_COUNT++))
fi

# 4. 创建归档元数据
cat > "$ARCHIVE_DIR/archive_info.json" <<EOF
{
  "archive_date": "$(date -Iseconds)",
  "archive_timestamp": "$(date +%Y%m%d_%H%M%S)",
  "project_name": "GoAfar",
  "description": "Files created before gaozao.md implementation (2025-11-09)",
  "cutoff_date": "2025-11-09T17:57:20",
  "moved_files": $MOVED_COUNT,
  "categories": {
    "docs_old": "Documentation from gaozao.md era (review_docs/)",
    "other": "Other outdated documents and logs"
  },
  "recovery_script": "scripts/restore_from_archive.sh"
}
EOF

log_info ""
log_info "=========================================="
log_info "归档完成: $MOVED_COUNT 个文件"
log_info "归档位置: $ARCHIVE_DIR/"
log_info "恢复方式: bash scripts/restore_from_archive.sh list"
log_info "=========================================="

# 5. 创建恢复脚本
cat > scripts/restore_from_archive.sh <<'RESTORE_SCRIPT'
#!/bin/bash
#
# GoAfar 归档恢复脚本
#

set -e

GREEN='\033[0;32m'
NC='\033[0m'

ARCHIVE_DIR=".archive_before_gaozao_20251109"

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

case "${1:-list}" in
    "docs")
        log_info "恢复文档..."
        cp -r "$ARCHIVE_DIR/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "文档已恢复"
        ;;
    "list")
        log_info "归档内容列表:"
        find "$ARCHIVE_DIR" -type f | sort
        ;;
    *)
        echo "用法: $0 {docs|list}"
        echo ""
        echo "命令:"
        echo "  docs  - 恢复归档的文档到项目根目录"
        echo "  list  - 列出归档内容"
        exit 1
        ;;
esac
'RESTORE_SCRIPT'

chmod +x scripts/restore_from_archive.sh
log_info "恢复脚本已创建: scripts/restore_from_archive.sh"
