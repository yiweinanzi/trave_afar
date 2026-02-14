#!/bin/bash
# 环境准备脚本
# 用于重建GoAfar项目的conda环境

set -e

echo "======================================"
echo "GoAfar 环境准备脚本"
echo "======================================"
echo ""

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: conda未安装，请先安装Miniconda或Anaconda"
    exit 1
fi

ENV_NAME=${1:-goafar}

echo "当前conda环境: $CONDA_DEFAULT_ENV"
echo "目标环境名称: $ENV_NAME"
echo ""

read -p "是否删除并重建环境 '$ENV_NAME'? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "取消操作"
    exit 0
fi

# 备份当前环境（可选）
read -p "是否备份��前环境? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    BACKUP_FILE="goafar_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "备份当前环境到: $BACKUP_FILE"
    conda pack -n $ENV_NAME -o $BACKUP_FILE 2>/dev/null || echo "备份跳过（可能环境不存在）"
fi

# 删除旧环境
echo ""
echo "删除旧环境..."
conda deactivate 2>/dev/null || true
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# 创建新环境
echo "创建新环境 (Python 3.12)..."
conda create -n $ENV_NAME python=3.12 -y

# 激活环境
echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# 验证Python版本
echo ""
echo "Python版本: $(python --version)"

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装PyTorch 2.9 + CUDA 12.8
echo ""
echo "======================================"
echo "安装PyTorch 2.9 + CUDA 12.8"
echo "======================================"
pip install torch==2.9.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装核心依赖
echo ""
echo "======================================"
echo "安装核心依赖"
echo "======================================"
pip install transformers accelerate bitsandbytes peft
pip install trl datasets
pip install fastapi uvicorn pydantic
pip install pandas numpy
pip install geopandas fiona
pip install FlagEmbedding faiss-cpu
pip install recbole
pip install ortools
pip install huggingface_hub[cli]

# 安装项目依赖
echo ""
echo "安装项目依赖..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo ""
echo "======================================"
echo "环境准备完成！"
echo "======================================"
echo ""
echo "使用方法:"
echo "  conda activate $ENV_NAME"
echo ""
echo "下一步:"
echo "  1. 下载Qwen3模型: bash scripts/download_models_bg.sh"
echo "  2. 运行测试: python test_full_pipeline.py"
echo ""
