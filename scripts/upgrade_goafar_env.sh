#!/bin/bash
# GoAfar 环境升级脚本
# 目标: PyTorch 2.7+ 和 CUDA 12.8
# GPU: RTX 5090 (24GB)

set -e

echo "========================================"
echo "GoAfar 环境升级"
echo "========================================"
echo "目标: PyTorch 2.7+, CUDA 12.8"
echo "GPU: RTX 5090"
echo ""

CONDA_ENV="goafar"
PYTHON_VERSION="3.10"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}错误: conda未安装${NC}"
    exit 1
fi

# 备份当前环境
echo -e "${YELLOW}[1/7] 备份当前环境...${NC}"
CURRENT_ENV=$(conda info --envs | grep "*" | grep -v "base" | head -1)
if [ -n "$CURRENT_ENV" ]; then
    BACKUP_NAME="goafar_backup_$(date +%Y%m%d_%H%M%S)"
    conda create -n $BACKUP_NAME --clone $CONDA_ENV 2>/dev/null || echo "  已有备份，跳过"
    echo -e "  ${GREEN}✓${NC} 备份完成: $BACKUP_NAME"
else
    echo -e "  ${YELLOW}未找到goafar环境${NC}"
fi

# 删除旧环境
echo ""
echo -e "${YELLOW}[2/7] 删除旧环境...${NC}"
conda env remove -n $CONDA_ENV -y 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} 旧环境已删除"

# 创建新环境
echo ""
echo -e "${YELLOW}[3/7] 创建新环境 (Python ${PYTHON_VERSION})...${NC}"
conda create -n $CONDA_ENV python=$PYTHON_VERSION -y
echo -e "  ${GREEN}✓${NC} 环境创建完成"

# 激活环境
echo ""
echo -e "${YELLOW}[4/7] 激活环境...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 检查CUDA
echo ""
echo -e "${YELLOW}[5/7] 检查CUDA...${NC}"
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//g')
if [ -z "$CUDA_VERSION" ]; then
    # 尝试从nvidia-smi获取
    NVIDIA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo -e "  ${GREEN}✓${NC} NVIDIA CUDA: $NVIDIA_VERSION"
else
    echo -e "  ${GREEN}✓${NC} NVCC CUDA: $CUDA_VERSION"
fi

# 安装PyTorch (支持CUDA 12.x)
echo ""
echo -e "${YELLOW}[6/7] 安装PyTorch 2.7+...${NC}"

# PyTorch 2.7+ with CUDA 12.x
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu124
echo -e "  ${GREEN}✓${NC} PyTorch 2.7.0 安装完成"

# 验证安装
echo ""
echo -e "${YELLOW}[7/7] 验证安装...${NC}"

python -c "
import torch
import sys

print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA版本: {torch.version.cuda}')
    print(f'  GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}')
        print(f'    显存: {props.total_memory / 1024**3:.1f} GB')
    print(f'  cuDNN: {torch.backends.cudnn.version()}')
else:
    print(f'  ${YELLOW}⚠ CUDA不可用${NC}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}GoAfar 环境升级完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "使用方式:"
    echo "  conda activate $CONDA_ENV"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}升级失败！${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
