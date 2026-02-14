#!/bin/bash
# SFT快速训练测试脚本 - 验证QLoRA配置
#
# 用于快速测试训练流程是否正常
# 使用1个epoch，较小的序列长度

set -e

PROJECT_ROOT="/root/autodl-tmp/goafar_project_broken"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  SFT快速训练测试"
echo "=========================================="
echo ""
echo "配置:"
echo "  - Epochs: 1"
echo "  - Max Length: 256"
echo "  - Batch Size: 4"
echo ""

bash scripts/train_sft.sh --quick

echo ""
echo "快速测试完成！如果一切正常，可以运行完整训练："
echo "  bash scripts/train_sft.sh"
