#!/bin/bash
# GoAfar 全模型评测脚本
# 用法: bash scripts/evaluate_all.sh [options]

set -e

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate goafar

# 默认参数
SFT_MODEL="outputs/sft/qwen3-8b-tourism"
DPO_MODEL="outputs/dpo/qwen3-8b-dpo"
GRPO_MODEL=""  # GRPO模型是可选的
MAX_SAMPLES=100
OUTPUT_DIR="outputs/evaluation"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft-model)
            SFT_MODEL="$2"
            shift 2
            ;;
        --dpo-model)
            DPO_MODEL="$2"
            shift 2
            ;;
        --grpo-model)
            GRPO_MODEL="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  GoAfar 模型评测"
echo "=========================================="
echo ""
echo "SFT模型: $SFT_MODEL"
echo "DPO模型: $DPO_MODEL"
echo "GRPO模型: ${GRPO_MODEL:-未指定}"
echo "最大样本: $MAX_SAMPLES"
echo "输出目录: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# 1. SFT评测
echo "=========================================="
echo "1. SFT模型评测"
echo "=========================================="

if [ -d "$SFT_MODEL" ] || [ "$SFT_MODEL" = "skip" ]; then
    if [ "$SFT_MODEL" != "skip" ]; then
        python src/evaluation/evaluate_sft.py \
            --model "$SFT_MODEL" \
            --test-data outputs/datasets/sft_data.jsonl \
            --max-samples "$MAX_SAMPLES" \
            --output "$OUTPUT_DIR/sft_eval.json" \
            2>&1 | tee "$OUTPUT_DIR/sft_eval.log"
    else
        echo "跳过SFT评测"
    fi
else
    echo "SFT模型不存在: $SFT_MODEL"
    echo "跳过SFT评测"
fi

# 2. DPO评测
echo ""
echo "=========================================="
echo "2. DPO模型评测"
echo "=========================================="

if [ -d "$DPO_MODEL" ] || [ "$DPO_MODEL" = "skip" ]; then
    if [ "$DPO_MODEL" != "skip" ]; then
        python src/evaluation/evaluate_dpo.py \
            --model "$DPO_MODEL" \
            --test-data outputs/datasets/dpo_prefs.csv \
            --max-samples "$MAX_SAMPLES" \
            --output "$OUTPUT_DIR/dpo_eval.json" \
            2>&1 | tee "$OUTPUT_DIR/dpo_eval.log"
    else
        echo "跳过DPO评测"
    fi
else
    echo "DPO模型不存在: $DPO_MODEL"
    echo "跳过DPO评测"
fi

# 3. GRPO评测
echo ""
echo "=========================================="
echo "3. GRPO模型评测"
echo "=========================================="

if [ -n "$GRPO_MODEL" ] && [ -d "$GRPO_MODEL" ]; then
    python src/evaluation/evaluate_grpo.py \
        --model "$GRPO_MODEL" \
        --test-data outputs/datasets/grpo_planner_prompts.jsonl \
        --poi-csv data/all/poi_expanded.csv \
        --max-samples "$MAX_SAMPLES" \
        --output "$OUTPUT_DIR/grpo_eval.json" \
        2>&1 | tee "$OUTPUT_DIR/grpo_eval.log"
elif [ -n "$GRPO_MODEL" ]; then
    echo "GRPO模型不存在: $GRPO_MODEL"
    echo "使用基准策略评测..."
    python src/evaluation/evaluate_grpo.py \
        --model "" \
        --test-data outputs/datasets/grpo_planner_prompts.jsonl \
        --poi-csv data/all/poi_expanded.csv \
        --max-samples "$MAX_SAMPLES" \
        --output "$OUTPUT_DIR/grpo_baseline.json" \
        2>&1 | tee "$OUTPUT_DIR/grpo_baseline.log"
else
    echo "未指定GRPO模型，跳过评测"
fi

# 4. 端到端评测
echo ""
echo "=========================================="
echo "4. 端到端评测"
echo "=========================================="

if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
    echo "已存在端到端评测结果，跳过。"
else
    python src/evaluation/evaluate_pipeline.py \
        --use-llm \
        --output "$OUTPUT_DIR/pipeline_eval.json" \
        2>&1 | tee "$OUTPUT_DIR/pipeline_eval.log"
fi

# 5. 召回贡献分析
echo ""
echo "=========================================="
echo "5. 召回贡献分析"
echo "=========================================="

if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
    echo "分析各路召回贡献..."
    python -c "
import json
import sys

with open('$OUTPUT_DIR/pipeline_eval.json', 'r') as f:
    data = json.load(f)

metrics = data.get('metrics', {})
recall_contrib = metrics.get('recall_contributions', {})

if recall_contrib and any(recall_contrib.values()):
    print('\n各路召回平均贡献:')
    total = sum([v for v in recall_contrib.values() if isinstance(v, (int, float))])
    if total > 0:
        for recall_type, count in recall_contrib.items():
            if isinstance(count, (int, float)):
                pct = count / total * 100
                print(f'  {recall_type}: {count:.1f} ({pct:.1f}%)')
else:
    print('未找到召回贡献数据或数据为空')
"
fi

# 6. 转化率和延迟分析
echo ""
echo "=========================================="
echo "6. 转化率和延迟分析"
echo "=========================================="

if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
    python -c "
import json

with open('$OUTPUT_DIR/pipeline_eval.json', 'r') as f:
    data = json.load(f)

metrics = data.get('metrics', {})

# 转化率
conv = metrics.get('conversion_metrics', {})
if conv:
    print('\n转化率指标:')
    print(f'  召回→重排序: {conv.get(\"recall_to_rerank\", 0):.2%}')
    print(f'  重排序→路线: {conv.get(\"rerank_to_route\", 0):.2%}')
    print(f'  整体成功率: {conv.get(\"overall_success\", 0):.2%}')

# 延迟分解
latency = metrics.get('latency_breakdown', {})
if latency and isinstance(latency, dict):
    print('\n延迟分解:')
    for stage, time_val in latency.items():
        if isinstance(time_val, (int, float)):
            print(f'  {stage}: {time_val:.2f}s')
"
fi

# 7. 汇总报告
echo ""
echo "=========================================="
echo "  评测完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  SFT:   $OUTPUT_DIR/sft_eval.json"
echo "  DPO:   $OUTPUT_DIR/dpo_eval.json"
echo "  GRPO:  $OUTPUT_DIR/grpo_eval.json"
echo "  端到端: $OUTPUT_DIR/pipeline_eval.json"
echo ""

# 生成简要汇总
python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
summary = []

for eval_file in ['sft_eval.json', 'dpo_eval.json', 'grpo_eval.json', 'pipeline_eval.json']:
    eval_path = output_dir / eval_file
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            data = json.load(f)
        metrics = data.get('metrics', {})
        summary.append({
            'name': eval_file.replace('_eval.json', '').upper(),
            'total': metrics.get('total_queries', metrics.get('total_samples', 0)),
            'success': metrics.get('successful_queries', metrics.get('successful_samples', 0))
        })

if summary:
    print('【评测汇总】')
    for item in summary:
        total = item['total']
        success = item['success']
        rate = success / total * 100 if total > 0 else 0
        print(f\"  {item['name']}: {success}/{total} ({rate:.1f}%)\")
"
echo ""

