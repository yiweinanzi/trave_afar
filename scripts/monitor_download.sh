#!/bin/bash
# 模型下载监控脚本

while true; do
    clear
    echo "======================================"
    echo "Qwen3模型下载进度监控"
    echo "======================================"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    for model in "Qwen3-8B" "Qwen3-Embedding-4B" "Qwen3-Reranker-4B"; do
        dir="models/$model"
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            files=$(find "$dir" -type f 2>/dev/null | wc -l)

            # 检查是否在下载中
            if ls "$dir"/*.tmp 2>/dev/null | head -1 > /dev/null; then
                status="下载中..."
            elif [ -f "$dir/model.safetensors" ] || [ -f "$dir/pytorch_model.bin" ]; then
                status="✓ 完成"
            else
                status="进行中..."
            fi

            printf "%-25s %8s  %4d files  %s\n" "$model" "$size" "$files" "$status"
        else
            printf "%-25s %8s  %4s files  %s\n" "$model" "-" "-" "未开始"
        fi
    done

    echo ""
    echo "按 Ctrl+C 退出，等待5秒后刷新..."
    sleep 5
done
