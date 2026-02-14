#!/bin/bash
#
# GoAfar 项目备份与归档脚本
# 用于将过期代码和文档归档，并支持一键恢复
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/root/autodl-tmp/goafar_project"
cd "$PROJECT_ROOT"

# 归档目录
ARCHIVE_DIR="archive"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 创建归档目录结构
init_archive_dirs() {
    log_info "创建归档目录结构..."
    mkdir -p "$ARCHIVE_DIR/docs_old/project_docs"
    mkdir -p "$ARCHIVE_DIR/docs_old/review_docs"
    mkdir -p "$ARCHIVE_DIR/docs_old/web_ui_docs"
    mkdir -p "$ARCHIVE_DIR/code_old/web_ui"
    mkdir -p "$ARCHIVE_DIR/code_old/legacy"
    mkdir -p "$ARCHIVE_DIR/outputs_legacy"
    mkdir -p "$ARCHIVE_DIR/logs_archive"
    mkdir -p "$ARCHIVE_DIR/meta"
    log_info "归档目录创建完成"
}

# 归档文档
archive_docs() {
    log_info "归档过期文档..."

    # 项目文档 (在 gaozao.md 实施前)
    local moved=0
    for file in PROJECT_OVERVIEW.txt PROJECT_SUMMARY.md README_项目指南.md DOWNLOAD_LIST.md REFERENCE_CODES.md; do
        if [ -f "$file" ]; then
            git mv "$file" "$ARCHIVE_DIR/docs_old/project_docs/" 2>/dev/null || mv "$file" "$ARCHIVE_DIR/docs_old/project_docs/"
            ((moved++))
        fi
    done

    # 项目完整文档 (多个版本)
    for file in 项目完整文档.md 项目完整文档-2024.md 项目完整文档-更新.md; do
        if [ -f "$file" ]; then
            git mv "$file" "$ARCHIVE_DIR/docs_old/project_docs/" 2>/dev/null || mv "$file" "$ARCHIVE_DIR/docs_old/project_docs/"
            ((moved++))
        fi
    done

    # 简历相关文档
    for file in 简历-面试问答.md 简历-项目描述.md 简历-面试问答-更新.md; do
        if [ -f "$file" ]; then
            git mv "$file" "$ARCHIVE_DIR/docs_old/" 2>/dev/null || mv "$file" "$ARCHIVE_DIR/docs_old/"
            ((moved++))
        fi
    done

    # 最终交付报告 (多个版本)
    for file in 最终交付报告.md 最终交付报告-2024.md 最终交付报告-更新.md; do
        if [ -f "$file" ]; then
            git mv "$file" "$ARCHIVE_DIR/docs_old/" 2>/dev/null || mv "$file" "$ARCHIVE_DIR/docs_old/"
            ((moved++))
        fi
    done

    log_info "文档归档完成: $moved 个文件"
}

# 归档 review_docs (已在 7786c92 清空，但可能有历史版本)
archive_review_docs() {
    log_info "归档 review_docs..."

    if [ -d "review_docs" ]; then
        # 检查是否为空
        if [ -z "$(ls -A review_docs)" ]; then
            log_warn "review_docs 目录已为空"
        else
            git mv review_docs/* "$ARCHIVE_DIR/docs_old/review_docs/" 2>/dev/null || mv review_docs/* "$ARCHIVE_DIR/docs_old/review_docs/"
            log_info "review_docs 归档完成"
        fi
    fi
}

# 归档过期代码
archive_old_code() {
    log_info "归档过期代码..."

    # Web UI 相关 (已被 app.py 替代)
    for file in web_ui/demo_ui.py web_ui/web_ui.py web_ui/simple_gradio.py; do
        if [ -f "$file" ]; then
            git mv "$file" "$ARCHIVE_DIR/code_old/web_ui/" 2>/dev/null || mv "$file" "$ARCHIVE_DIR/code_old/web_ui/"
        fi
    done

    log_info "过期代码归档完成"
}

# 归档日志文件
archive_logs() {
    log_info "归档日志文件..."

    local moved=0
    for log_file in *.log; do
        if [ -f "$log_file" ]; then
            mv "$log_file" "$ARCHIVE_DIR/logs_archive/"
            ((moved++))
        fi
    done

    # 检查 context/logs 如果存在
    if [ -d "context/logs" ]; then
        for log_file in context/logs/*.log; do
            if [ -f "$log_file" ]; then
                mv "$log_file" "$ARCHIVE_DIR/logs_archive/"
                ((moved++))
            fi
        done
    fi

    log_info "日志归档完成: $moved 个文件"
}

# 创建归档元数据
create_archive_metadata() {
    log_info "创建归档元数据..."

    cat > "$ARCHIVE_DIR/meta/archive_info.json" << EOF
{
  "archive_date": "$(date -Iseconds)",
  "archive_timestamp": "$TIMESTAMP",
  "project_name": "GoAfar",
  "description": "过期代码和文档归档",
  "gaozao_date": "2025-11-09",
  "categories": {
    "docs_old": "gaozao.md 实施前的项目文档",
    "code_old": "被新架构替代的旧代码",
    "outputs_legacy": "gaozao.md 实施前的模型输出",
    "logs_archive": "历史日志文件"
  },
  "recovery_script": "scripts/restore_from_archive.sh"
}
EOF
    log_info "归档元数据已创建"
}

# 创建恢复脚本
create_restore_script() {
    log_info "创建恢复脚本..."

    cat > scripts/restore_from_archive.sh << 'RESTORESCRIPT'
#!/bin/bash
#
# GoAfar 项目归档恢复脚本
# 用于从归档中恢复文件到项目根目录
#

set -e

PROJECT_ROOT="/root/autodl-tmp/goafar_project"
ARCHIVE_DIR="$PROJECT_ROOT/archive"

GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }

case "${1:-help}" in
    "docs")
        log_info "恢复文档..."
        cp -r "$ARCHIVE_DIR/docs_old/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "文档已恢复"
        ;;
    "code")
        log_info "恢复代码..."
        cp -r "$ARCHIVE_DIR/code_old/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "代码已恢复"
        ;;
    "all")
        log_info "恢复所有内容..."
        cp -r "$ARCHIVE_DIR/"* "$PROJECT_ROOT/" 2>/dev/null
        log_info "所有内容已恢复"
        ;;
    "list")
        log_info "归档内容列表:"
        find "$ARCHIVE_DIR" -type f | sort
        ;;
    *)
        echo "用法: $0 {docs|code|all|list}"
        echo ""
        echo "命令:"
        echo "  docs  - 恢复归档的文档"
        echo "  code  - 恢复归档的代码"
        echo "  all   - 恢复所有内容"
        echo "  list   - 列出归档内容"
        exit 1
        ;;
esac
RESTORESCRIPT

    chmod +x scripts/restore_from_archive.sh
    log_info "恢复脚本已创建: scripts/restore_from_archive.sh"
}

# 创建 .gitignore 更新
update_gitignore() {
    log_info "更新 .gitignore..."

    if ! grep -q "^archive/" .gitignore 2>/dev/null; then
        echo "" >> .gitignore
        echo "# 归档目录 (不需要提交到Git)" >> .gitignore
        echo "archive/" >> .gitignore
    fi

    if ! grep -q "^*.log" .gitignore 2>/dev/null; then
        echo "*.log" >> .gitignore
    fi

    log_info ".gitignore 已更新"
}

# 创建备份提交
create_backup_commit() {
    log_info "创建备份提交..."

    # 添加归档目录
    git add archive/ || true
    git add scripts/restore_from_archive.sh || true

    # 提交
    git commit -m "chore: 归档过期代码和文档

- 归档 gaozao.md 实施前的文档
- 归档旧的 web_ui 代码
- 归档历史日志文件
- 添加恢复脚本 scripts/restore_from_archive.sh

归档内容可通过 scripts/restore_from_archive.sh 恢复" || true

    log_info "备份提交已完成"
}

# 主函数
main() {
    log_info "=== GoAfar 项目归档脚本 ==="
    log_info "项目目录: $PROJECT_ROOT"
    log_info "归档目录: $ARCHIVE_DIR"
    log_info "时间戳: $TIMESTAMP"

    # 检查是否在正确的目录
    if [ ! -f "gaozao.md" ]; then
        log_error "未找到 gaozao.md，请确保在项目根目录执行"
        exit 1
    fi

    # 执行归档
    init_archive_dirs
    archive_docs
    archive_review_docs
    archive_old_code
    archive_logs
    create_archive_metadata
    create_restore_script
    update_gitignore

    # 询问是否创建提交
    echo ""
    read -p "$(echo -e ${YELLOW}是否创建 Git 提交? [y/N]: ${NC})" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_backup_commit
    fi

    log_info "=== 归档完成 ===="
    echo ""
    log_info "下一步操作:"
    echo "  1. 检查归档: ls -la archive/"
    echo "  2. 提交到 GitHub: git push origin main"
    echo "  3. 恢复文件: bash scripts/restore_from_archive.sh list"
}

# 运行主函数
main "$@"
