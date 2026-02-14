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
