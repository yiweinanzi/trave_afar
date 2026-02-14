#!/bin/bash
# GoAfar 完整评测脚本
# 包含推荐质量、召回贡献、转化率、延迟分解等全面指标
#
# 用法: bash scripts/evaluate_complete.sh [options]

set -e

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate goafar

# 默认参数
USE_LLM=false
TRACK_RECALL=true
QUERIES_FILE=""
MAX_SAMPLES=50
OUTPUT_DIR="outputs/evaluation"
K_VALUES="5,10,20,50"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --use-llm)
            USE_LLM=true
            shift
            ;;
        --no-recall-tracking)
            TRACK_RECALL=false
            shift
            ;;
        --queries)
            QUERIES_FILE="$2"
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
        --k-values)
            K_VALUES="$2"
            shift 2
            ;;
        --help)
            echo "用法: bash scripts/evaluate_complete.sh [options]"
            echo ""
            echo "选项:"
            echo "  --use-llm                使用LLM模式（需要模型）"
            echo "  --no-recall-tracking     不跟踪召回贡献"
            echo "  --queries FILE           测试查询CSV文件"
            echo "  --max-samples N          最大样本数 (默认: 50)"
            echo "  --output-dir DIR         输出目录 (默认: outputs/evaluation)"
            echo "  --k-values K1,K2,...     K值列表 (默认: 5,10,20,50)"
            echo "  --help                   显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "  GoAfar 完整评测"
echo "=========================================="
echo ""
echo "配置:"
echo "  LLM模式: $USE_LLM"
echo "  召回跟踪: $TRACK_RECALL"
echo "  查询文件: ${QUERIES_FILE:-使用默认测试查询}"
echo "  最大样本: $MAX_SAMPLES"
echo "  输出目录: $OUTPUT_DIR"
echo "  K值: $K_VALUES"
echo ""

mkdir -p "$OUTPUT_DIR"

# 1. 端到端流水线评测
echo "=========================================="
echo "1. 端到端流水线评测"
echo "=========================================="

PIPELINE_ARGS="--output $OUTPUT_DIR/pipeline_eval.json"
if [ "$USE_LLM" = true ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --use-llm"
fi
if [ -n "$QUERIES_FILE" ]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --queries $QUERIES_FILE"
fi

python src/evaluation/evaluate_pipeline.py $PIPELINE_ARGS \
    2>&1 | tee "$OUTPUT_DIR/pipeline_eval.log"

# 2. 推荐质量指标评测
echo ""
echo "=========================================="
echo "2. 推荐质量指标评测"
echo "=========================================="

# 这里可以添加更多推荐质量评测
# 例如：离线评测、A/B测试等

# 3. 召回贡献分析
echo ""
echo "=========================================="
echo "3. 召回贡献分析"
echo "=========================================="

if [ "$TRACK_RECALL" = true ]; then
    echo "分析各路召回贡献..."
    # 可以从pipeline_eval.json中提取召回贡献数据
    if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
        python -c "
import json
import sys

with open('$OUTPUT_DIR/pipeline_eval.json', 'r') as f:
    data = json.load(f)

metrics = data.get('metrics', {})
recall_contrib = metrics.get('recall_contributions', {})

if recall_contrib:
    print('\n各路召回平均贡献:')
    total = sum(recall_contrib.values())
    if total > 0:
        for recall_type, count in recall_contrib.items():
            pct = count / total * 100
            print(f'  {recall_type}: {count:.1f} ({pct:.1f}%)')
else:
    print('未找到召回贡献数据')
"
    fi
fi

# 4. 转化率分析
echo ""
echo "=========================================="
echo "4. 转化率分析"
echo "=========================================="

if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
    python -c "
import json

with open('$OUTPUT_DIR/pipeline_eval.json', 'r') as f:
    data = json.load(f)

metrics = data.get('metrics', {})
conv = metrics.get('conversion_metrics', {})

if conv:
    print('\n转化率指标:')
    print(f'  召回→重排序: {conv.get(\"recall_to_rerank\", 0):.2%}')
    print(f'  重排序→路线: {conv.get(\"rerank_to_route\", 0):.2%}')
    print(f'  整体成功率: {conv.get(\"overall_success\", 0):.2%}')
"
fi

# 5. 延迟分解分析
echo ""
echo "=========================================="
echo "5. 延迟分解分析"
echo "=========================================="

if [ -f "$OUTPUT_DIR/pipeline_eval.json" ]; then
    python -c "
import json

with open('$OUTPUT_DIR/pipeline_eval.json', 'r') as f:
    data = json.load(f)

metrics = data.get('metrics', {})
latency = metrics.get('latency_breakdown', {})
avg_latency = metrics.get('avg_latency', 0)

print('\n平均延迟分解:')
if isinstance(latency, dict) and latency:
    total = latency.get('total', avg_latency)
    if total > 0:
        for stage, time_val in latency.items():
            if stage != 'total' and isinstance(time_val, (int, float)):
                pct = time_val / total * 100
                print(f'  {stage}: {time_val:.2f}s ({pct:.1f}%)')
    print(f'  总计: {total:.2f}s')
"
fi

# 6. 生成汇总报告
echo ""
echo "=========================================="
echo "6. 生成汇总报告"
echo "=========================================="

python -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
report_path = output_dir / 'evaluation_summary.txt'

# 收集所有评测结果
results = {}
for eval_file in ['pipeline_eval.json']:
    eval_path = output_dir / eval_file
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            results[eval_file] = json.load(f)

# 生成汇总报告
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('GoAfar 评测汇总报告\n')
    f.write('=' * 80 + '\n\n')

    for eval_name, eval_data in results.items():
        f.write(f'\n【{eval_name}】\n')
        metrics = eval_data.get('metrics', {})
        f.write(f'  成功查询: {metrics.get(\"successful_queries\", 0)}/{metrics.get(\"total_queries\", 0)}\n')
        f.write(f'  可行路线: {metrics.get(\"feasible_routes\", 0)}\n')
        f.write(f'  平均延迟: {metrics.get(\"avg_latency\", 0):.2f}s\n')
        f.write(f'  平均候选数: {metrics.get(\"avg_candidates\", 0):.0f}\n')
        f.write(f'  平均最终POI: {metrics.get(\"avg_final_pois\", 0):.0f}\n')

print(f'\n汇总报告已保存: {report_path}')
"

# 7. 完成
echo ""
echo "=========================================="
echo "  评测完成！"
echo "=========================================="
echo ""
echo "结果文件:"
echo "  端到端评测: $OUTPUT_DIR/pipeline_eval.json"
echo "  端到端日志: $OUTPUT_DIR/pipeline_eval.log"
echo "  汇总报告: $OUTPUT_DIR/evaluation_summary.txt"
echo ""
echo "使用以下命令查看详细结果:"
echo "  cat $OUTPUT_DIR/evaluation_summary.txt"
echo ""
