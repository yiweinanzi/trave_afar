#!/bin/bash
# 重建POI向量索引脚本
# 用于为127,978个POI生成Qwen3-Embedding-4B语义向量

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GoAfar POI 向量索引重建工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 默认参数
POI_CSV="data/all/poi_expanded.csv"
OUTPUT_DIR="outputs/emb"
MODEL_PATH="models/Xorbits/bge-m3"
MODEL_TYPE="bge_m3"
BATCH_SIZE=128
USE_GPU=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --poi-csv)
            POI_CSV="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --no-gpu)
            USE_GPU=false
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --poi-csv PATH         POI数据文件路径 (默认: data/all/poi_expanded.csv)"
            echo "  --output-dir PATH      输出目录 (默认: outputs/emb)"
            echo "  --model-path PATH      模型路径 (默认: models/Qwen3-Embedding-4B)"
            echo "  --model-type TYPE      模型类型: qwen3 或 bge_m3 (默认: qwen3)"
            echo "  --batch-size SIZE      批处理大小 (默认: 128)"
            echo "  --no-gpu               不使用GPU"
            echo "  --help                 显示此帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 显示配置信息
echo -e "${YELLOW}配置信息:${NC}"
echo "  POI数据文件: $POI_CSV"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型路径: $MODEL_PATH"
echo "  模型类型: $MODEL_TYPE"
echo "  批处理大小: $BATCH_SIZE"
echo "  使用GPU: $USE_GPU"
echo ""

# 检查输入文件是否存在
if [ ! -f "$POI_CSV" ]; then
    echo -e "${RED}错误: POI数据文件不存在: $POI_CSV${NC}"
    exit 1
fi

# 检查模型目录是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型目录不存在: $MODEL_PATH${NC}"
    exit 1
fi

# 统计POI数量
POI_COUNT=$(wc -l < "$POI_CSV")
POI_COUNT=$((POI_COUNT - 1))  # 减去表头
echo -e "${GREEN}✓ 发现 $POI_COUNT 个POI${NC}"
echo ""

# 构建GPU参数
GPU_ARGS=""
if [ "$USE_GPU" = false ]; then
    GPU_ARGS="--no-gpu"
fi

# 开始计时
START_TIME=$(date +%s)

echo -e "${GREEN}开始生成向量...${NC}"
echo ""

# 运行Python脚本
python src/embedding/build_embeddings_gpu.py \
    --poi-csv "$POI_CSV" \
    --output-dir "$OUTPUT_DIR" \
    --model-path "$MODEL_PATH" \
    --model-type "$MODEL_TYPE" \
    --batch-size "$BATCH_SIZE" \
    $GPU_ARGS

# 计算耗时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  向量生成完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}总耗时: ${MINUTES}分${SECONDS}秒${NC}"
echo ""

# 显示输出文件
MODEL_SUFFIX=$(echo "$MODEL_TYPE" | sed 's/_//g')
EMB_FILE="${OUTPUT_DIR}/poi_emb_${MODEL_SUFFIX}.npy"
META_FILE="${OUTPUT_DIR}/poi_meta_${MODEL_SUFFIX}.csv"

if [ -f "$EMB_FILE" ]; then
    EMB_SIZE=$(du -h "$EMB_FILE" | cut -f1)
    echo -e "${GREEN}✓ 向量文件: $EMB_FILE ($EMB_SIZE)${NC}"
fi

if [ -f "$META_FILE" ]; then
    META_SIZE=$(du -h "$META_FILE" | cut -f1)
    echo -e "${GREEN}✓ 元数据文件: $META_FILE ($META_SIZE)${NC}"
fi

echo ""
echo -e "${YELLOW}下一步:${NC}"
echo "  1. 验证向量数量: python scripts/validate_embeddings.py"
echo "  2. 测试语义检索: python -c \"from src.embedding.vector_builder import search_similar_pois; search_similar_pois('想去喀纳斯', topk=10)\""
echo "  3. 构建FAISS索引: python -c \"from src.embedding.vector_builder import build_faiss_index; import numpy as np; build_faiss_index(np.load('$EMB_FILE'))\""
