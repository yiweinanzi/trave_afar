
训练范式      脚本                          状态    用途
────────────────────────────────────────────────────────────────
SFT           src/content_generation/     ✓       基础能力微调
              train_sft.py

DPO           src/content_generation/     ✓       用户偏好对齐  
              train_dpo.py

GRPO          src/rl/grpo_trainer.py    ✓       强化学习策略
              优化

训练流程:
1. SFT: 学习基本任务格式 → 输出合格的基础模型
2. DPO: 使用用户反馈对齐 → 优化用户满意度
3. GRPO: 多目标奖励优化 → 优化行程规划策略

数据准备:
- POI描述数据: ✓ (data/poi.csv)
- 用户行为数据: ✓ (data/user_events.csv)  
- 偏好对数据: ✓ (可从user_events构造)
- 奖励配置: ✓ (grpo_trainer.py)

训练命令:
# SFT训练
python src/content_generation/train_sft.py --model models/Qwen3-8B --data data/poi.csv

# DPO训练
python src/content_generation/train_dpo.py --model models/Qwen3-8B --prefs data/preferences.json

# GRPO训练
python src/rl/grpo_trainer.py --config configs/grpo_planner.yaml
