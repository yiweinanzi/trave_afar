#!/bin/bash
# 模型下载脚本
# 用法: bash scripts/download_models.sh

set -e

PROJECT_ROOT="/root/autodl-tmp/goafar_project"
MODELS_DIR="$PROJECT_ROOT/models"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$MODELS_DIR"
mkdir -p "$LOG_DIR"


pip show huggingface_hub >/dev/null 2>&1 || {
    echo "安装huggingface_hub..."
    pip install huggingface_hub -q
}

echo ""
echo "=========================================="
echo "  GoAfar 模型下载"
echo "=========================================="
echo "模型目录: $MODELS_DIR"
echo "日志目录: $LOG_DIR"
echo ""

# 检查Qwen3-8B状态
echo "1. 检查 Qwen3-8B..."
if [ -f "$MODELS_DIR/Qwen3-8B/config.json" ]; then
    size=$(du -sh "$MODELS_DIR/Qwen3-8B" | cut -f1)
    echo "  ✓ Qwen3-8B 已存在 ($size)"
else
    echo "  ✗ Qwen3-8B 不完整"
fi

# 下载 Embedding 模型
echo ""
echo "2. 下载 Qwen3-Embedding-4B (~8GB)..."
python -c "
from huggingface_hub import snapshot_download
import os

models_dir = '/root/autodl-tmp/goafar_project/models'
os.makedirs(models_dir, exist_ok=True)

print('开始下载 Qwen3-Embedding-4B...')
snapshot_download(
    repo_id='Qwen/Qwen3-Embedding-4B',
    local_dir=os.path.join(models_dir, 'Qwen3-Embedding-4B'),
    local_dir_use_symlinks=False,
    resume_download=True
)
print('下载完成')
" 2>&1 | tee "$LOG_DIR/download_embedding_bg.log"

# 下载 Reranker 模型
echo ""
echo "3. 下载 Qwen3-Reranker-4B (~8GB)..."
python -c "
from huggingface_hub import snapshot_download
import os

models_dir = '/root/autodl-tmp/goafar_project/models'
os.makedirs(models_dir, exist_ok=True)

print('开始下载 Qwen3-Reranker-4B...')
snapshot_download(
    repo_id='Qwen/Qwen3-Reranker-4B',
    local_dir=os.path.join(models_dir, 'Qwen3-Reranker-4B'),
    local_dir_use_symlinks=False,
    resume_download=True
)
print('下载完成')
" 2>&1 | tee "$LOG_DIR/download_reranker_bg.log"

echo ""
echo "=========================================="
echo "  下载完成！"
echo "=========================================="
echo ""
echo "查看日志:"
echo "  Embedding: tail -f $LOG_DIR/download_embedding_bg.log"
echo "  Reranker: tail -f $LOG_DIR/download_reranker_bg.log"
echo ""
echo "检查状态:"
echo "  bash scripts/check_models.sh"
