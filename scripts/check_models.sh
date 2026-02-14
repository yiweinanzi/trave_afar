#!/bin/bash
# 检查模型下载状态
# 用法: bash scripts/check_models.sh

set -e

MODELS_DIR="/root/autodl-tmp/goafar_project/models"

echo "=========================================="
echo "  GoAfar 模型状态检查"
echo "=========================================="
echo ""

for model in "Qwen3-8B" "Qwen3-Embedding-4B" "Qwen3-Reranker-4B"; do
    model_dir="$MODELS_DIR/$model"

    echo "模型: $model"

    if [ -d "$model_dir" ]; then
        # 计算大小
        if command -v du >/dev/null 2>&1; then
            size=$(du -sh "$model_dir" | cut -f1)
            echo "  目录存在: $size"
        else
            echo "  目录存在"
        fi

        # 检查关键文件
        if [ -f "$model_dir/config.json" ]; then
            echo "  ✓ config.json 存在"
        else
            echo "  ✗ config.json 不存在"
        fi

        # 计算文件数
        file_count=$(find "$model_dir" -type f | wc -l)
        echo "  文件数: $file_count"

        # 检查safetensors
        safetensors=$(find "$model_dir" -name "*.safetensors" 2>/dev/null | wc -l)
        echo "  模型文件: $safetensors 个"
    else
        echo "  ✗ 目录不存在"
    fi

    echo ""
done

echo "=========================================="
echo "  使用方法"
echo "=========================================="
echo ""
echo "下载模型:"
echo "  bash scripts/download_models.sh"
echo ""
echo "查看下载进度:"
echo "  tail -f logs/download_embedding_bg.log"
echo "  tail -f logs/download_reranker_bg.log"
