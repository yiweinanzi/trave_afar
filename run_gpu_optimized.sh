#!/bin/bash
# GoAfar GPU优化运行脚本

echo "=========================================="
echo "GoAfar - GPU加速完整流程"
echo "=========================================="

# 激活环境
source /root/miniconda3/bin/activate goafar

# 切换到项目目录
cd /root/autodl-tmp/goafar_project

# 检查GPU
echo ""
echo "检查GPU状态..."
python -c "import torch; print(f'GPU可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}') if torch.cuda.is_available() else None; print(f'GPU名称: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"

# Step 1: 生成POI向量（GPU加速）
echo ""
echo "=========================================="
echo "[1/4] 构建POI向量（GPU加速）"
echo "=========================================="
python src/embedding/build_embeddings_gpu.py --batch-size 256

# Step 2: 导出RecBole数据
echo ""
echo "=========================================="
echo "[2/4] 导出RecBole数据"
echo "=========================================="
python -c "from src.recommendation.recbole_trainer import export_recbole_data; export_recbole_data()"

# Step 3: 训练RecBole模型（GPU）
echo ""
echo "=========================================="
echo "[3/4] 训练RecBole模型（GPU）"
echo "=========================================="
python train_recbole_gpu.py --gpu 0

# Step 4: 运行LLM增强推荐
echo ""
echo "=========================================="
echo "[4/4] LLM增强推荐（Qwen3-8B）"
echo "=========================================="
python run_with_llm.py

echo ""
echo "=========================================="
echo "✓ GPU优化流程完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - POI向量: outputs/emb/poi_emb.npy"
echo "  - RecBole模型: outputs/recbole/saved/"
echo "  - 推荐结果: outputs/results/"
echo ""
echo "性能对比:"
echo "  - CPU向量生成: ~20分钟（1333个POI）"
echo "  - GPU向量生成: ~2-5分钟（1333个POI）"
echo "  - RecBole CPU: ~30分钟"
echo "  - RecBole GPU: ~5-10分钟"
echo ""

