#!/bin/bash
# GoAfar 快速开始脚本

echo "=========================================="
echo "GoAfar 智能旅行路线推荐系统 - 快速开始"
echo "=========================================="

# 激活环境
source /root/miniconda3/bin/activate goafar

# 切换到项目目录
cd /root/autodl-tmp/goafar_project

# Step 1: 数据准备
echo ""
echo "[1/4] 数据准备..."
python -c "from src.data_processing.sql_extractor import parse_go_address_sql; parse_go_address_sql()"
python -c "from src.data_processing.event_generator import generate_user_events; import pandas as pd; poi_df = pd.read_csv('data/poi.csv'); generate_user_events(poi_df)"

# Step 2: 构建POI向量
echo ""
echo "[2/4] 构建 POI 向量（BGE-M3）..."
python src/embedding/vector_builder.py

# Step 3: 导出RecBole数据
echo ""
echo "[3/4] 导出 RecBole 数据..."
python -c "from src.recommendation.recbole_trainer import export_recbole_data; export_recbole_data()"

# Step 4: 运行演示
echo ""
echo "[4/4] 运行端到端演示..."
python main.py

echo ""
echo "=========================================="
echo "✓ 完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  - POI向量: outputs/emb/poi_emb.npy"
echo "  - 推荐结果: outputs/results/"
echo ""
echo "下一步:"
echo "  - 训练RecBole模型: python -m recbole -c configs/recbole.yaml"
echo "  - 自定义查询: python main.py"
echo ""

