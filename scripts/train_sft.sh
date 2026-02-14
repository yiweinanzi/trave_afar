#!/bin/bash
# SFT训练脚本 - 使用QLoRA对Qwen3-8B进行监督微调
#
# 功能：
# - 使用4-bit量化 (NF4) + LoRA进行高效微调
# - 支持旅游推荐任务：意图理解、文案生成、POI推荐
# - 数据集: outputs/datasets/sft_data.jsonl
# - 输出: outputs/sft/qwen3-8b-tourism
#
# 使用方法:
#   bash scripts/train_sft.sh                    # 使用默认参数
#   bash scripts/train_sft.sh --epochs 5         # 自定义训练轮数
#   bash scripts/train_sft.sh --quick            # 快速测试（1个epoch）

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/root/autodl-tmp/goafar_project_broken"
cd "$PROJECT_ROOT"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  GoAfar SFT Training - QLoRA微调${NC}"
echo -e "${BLUE}================================================${NC}"

# 检查环境
echo -e "\n${YELLOW}[1/5] 检查环境...${NC}"

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python --version)${NC}"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA: $(nvidia-smi | head -n 1)${NC}"
else
    echo -e "${YELLOW}⚠ 警告: 未检测到CUDA，将使用CPU训练（速度较慢）${NC}"
fi

# 检查必要的包
echo -e "\n${YELLOW}检查依赖包...${NC}"
python -c "import torch; import transformers; import peft; import trl; import bitsandbytes" 2>/dev/null || {
    echo -e "${RED}错误: 缺少必要的依赖包${NC}"
    echo -e "请运行: pip install torch transformers peft trl bitsandbytes datasets"
    exit 1
}
echo -e "${GREEN}✓ 所有依赖包已安装${NC}"

# 检查数据文件
echo -e "\n${YELLOW}[2/5] 检查数据文件...${NC}"
DATA_FILE="outputs/datasets/sft_data.jsonl"

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}错误: 数据文件不存在: $DATA_FILE${NC}"
    echo -e "请先运行数据准备脚本生成训练数据"
    exit 1
fi

# 统计数据行数
NUM_SAMPLES=$(wc -l < "$DATA_FILE")
echo -e "${GREEN}✓ 数据文件: $DATA_FILE${NC}"
echo -e "${GREEN}  样本数: $NUM_SAMPLES 条${NC}"

# 显示前3条数据
echo -e "\n${YELLOW}数据示例（前3条）:${NC}"
head -n 3 "$DATA_FILE" | python -m json.tool 2>/dev/null || head -n 3 "$DATA_FILE"

# 解析命令行参数
QUICK_MODE=false
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            echo -e "${YELLOW}启用快速测试模式（1 epoch）${NC}"
            shift
            ;;
        --data)
            DATA_FILE="$2"
            EXTRA_ARGS="$EXTRA_ARGS --data $2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            EXTRA_ARGS="$EXTRA_ARGS --output $2"
            shift 2
            ;;
        --epochs)
            EXTRA_ARGS="$EXTRA_ARGS --epochs $2"
            shift 2
            ;;
        --batch-size)
            EXTRA_ARGS="$EXTRA_ARGS --batch-size $2"
            shift 2
            ;;
        --max-length)
            EXTRA_ARGS="$EXTRA_ARGS --max-length $2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# 快速模式配置
if [ "$QUICK_MODE" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --epochs 1 --batch-size 4 --max-length 256"
fi

# 创建输出目录
echo -e "\n${YELLOW}[3/5] 准备输出目录...${NC}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft/qwen3-8b-tourism}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ 输出目录: $OUTPUT_DIR${NC}"

# 设置模型缓存目录
export GOAFAR_MODEL_CACHE="${GOAFAR_MODEL_CACHE:-$PROJECT_ROOT/models}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
mkdir -p "$GOAFAR_MODEL_CACHE" "$HF_HOME"

echo -e "${GREEN}✓ 模型缓存: $GOAFAR_MODEL_CACHE${NC}"
echo -e "${GREEN}✓ HF缓存: $HF_HOME${NC}"

# 显示训练配置
echo -e "\n${YELLOW}[4/5] 训练配置${NC}"
echo -e "  模型: Qwen3-8B (4-bit量化)"
echo -e "  方法: QLoRA (r=16, alpha=16)"
echo -e "  数据: $DATA_FILE ($NUM_SAMPLES 样本)"
echo -e "  输出: $OUTPUT_DIR"
echo -e "  额外参数: $EXTRA_ARGS"

# 执行训练
echo -e "\n${YELLOW}[5/5] 开始训练...${NC}"
echo -e "${BLUE}================================================${NC}"

python src/content_generation/train_sft.py \
    --data "$DATA_FILE" \
    --output "$OUTPUT_DIR" \
    --use-qlora \
    --lora-r 16 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lr 2e-4 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --max-length 512 \
    $EXTRA_ARGS

# 检查训练结果
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}  训练完成！${NC}"
    echo -e "${GREEN}================================================${NC}"

    echo -e "\n${YELLOW}输出文件:${NC}"
    ls -lh "$OUTPUT_DIR" | tail -n +2

    # 显示模型大小
    if [ -d "$OUTPUT_DIR" ]; then
        MODEL_SIZE=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo -e "\n${GREEN}模型大小: $MODEL_SIZE${NC}"
    fi

    echo -e "\n${YELLOW}下一步:${NC}"
    echo -e "  1. 测试模型: python src/content_generation/test_sft.py --model $OUTPUT_DIR"
    echo -e "  2. 部署模型: 更新config/runtime.yaml中的model_path"
    echo -e "  3. 继续DPO训练: bash scripts/train_dpo.sh"

else
    echo -e "\n${RED}================================================${NC}"
    echo -e "${RED}  训练失败！${NC}"
    echo -e "${RED}================================================${NC}"
    exit 1
fi
