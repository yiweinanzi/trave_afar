#!/bin/bash
# 后台下载Qwen3系列模型
# 使用HF Mirror加速下载

set -e

export HF_ENDPOINT=https://hf-mirror.com
MODELS_DIR="./models"
LOG_DIR="./logs"

mkdir -p $MODELS_DIR
mkdir -p $LOG_DIR

echo "======================================"
echo "Qwen3系列模型后台下载脚本"
echo "======================================"
echo "模型目录: $MODELS_DIR"
echo "日志目录: $LOG_DIR"
echo "HF镜像: $HF_ENDPOINT"
echo "======================================"

# 检查huggingface-cli是否安装
CLI_CMD="huggingface-cli"
if ! command -v huggingface-cli &> /dev/null; then
    if python -c "import huggingface_hub" 2>/dev/null; then
        CLI_CMD="python -m huggingface_cli"
        echo "使用 python -m huggingface_cli"
    else
        echo "huggingface-cli未安装，正在安装..."
        pip install -U "huggingface_hub" -q
        CLI_CMD="python -m huggingface_cli"
    fi
fi


# 下载Qwen3-Embedding-4B（编码模型，优先级P1）
echo ""
echo "[$(date)] 启动下载: Qwen3-Embedding-4B (~8GB)..."
nohup $CLI_CMD download Qwen/Qwen3-Embedding-4B \
    --local-dir $MODELS_DIR/Qwen3-Embedding-4B \
    --local-dir-use-symlinks False \
    --resume-download > $LOG_DIR/download_embedding.log 2>&1 &
echo "Qwen3-Embedding-4B 下载已在后台启动，PID: $!"
echo "查看进度: tail -f $LOG_DIR/download_embedding.log"

# 等待2秒
sleep 2

# 下载Qwen3-Reranker-4B（重排序模型，优先级P2）
echo ""
echo "[$(date)] 启动下载: Qwen3-Reranker-4B (~8GB)..."
nohup $CLI_CMD download Qwen/Qwen3-Reranker-4B \
    --local-dir $MODELS_DIR/Qwen3-Reranker-4B \
    --local-dir-use-symlinks False \
    --resume-download > $LOG_DIR/download_reranker.log 2>&1 &
echo "Qwen3-Reranker-4B 下载已在后台启动，PID: $!"
echo "查看进度: tail -f $LOG_DIR/download_reranker.log"

echo ""
echo "======================================"
echo "所有模型下载任务已启动！"
echo "======================================"
echo ""
echo "查看下载进度："
echo "  tail -f $LOG_DIR/download_qwen3_8b.log"
echo "  tail -f $LOG_DIR/download_embedding.log"
echo "  tail -f $LOG_DIR/download_reranker.log"
echo ""
echo "查看下载进度汇总："
echo "  bash $0/status"
echo ""

# 如果参数是status，显示下载状态
if [ "$1" = "status" ]; then
    echo "======================================"
    echo "下载状态检查"
    echo "======================================"
    for model in Qwen3-8B Qwen3-Embedding-4B Qwen3-Reranker-4B; do
        dir="$MODELS_DIR/$model"
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            files=$(find "$dir" -type f | wc -l)
            echo "$model: $size ($files files)"
        else
            echo "$model: 尚未开始"
        fi
    done
fi
