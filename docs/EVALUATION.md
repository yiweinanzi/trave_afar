# GoAfar 评测指南

本文档描述GoAfar项目训练后的模型评测方���。

## 评测类型

### 1. SFT模型评测

评测监督微调后模型的任务执行能力。

**评测维度：**
- 意图理解准确率（省份、天数、兴趣等）
- 路线生成质量（POI覆盖率、天数匹配）
- 文案生成质量（标题、描述完整性）
- 推理速度

**运行方式：**
```bash
conda activate goafar

python src/evaluation/evaluate_sft.py \
    --model outputs/sft/qwen3-8b-tourism \
    --test-data outputs/datasets/sft_data.jsonl \
    --max-samples 100 \
    --output outputs/evaluation/sft_eval.json
```

### 2. DPO模型评测

评测偏好对齐训练后模型的质量偏好能力。

**评测维度：**
- 偏好对齐准确率（chosen vs rejected）
- 奖励分数对比
- 按数据来源的准确率

**运行方式：**
```bash
python src/evaluation/evaluate_dpo.py \
    --model outputs/dpo/qwen3-8b-dpo \
    --test-data outputs/datasets/dpo_prefs.csv \
    --max-samples 100 \
    --output outputs/evaluation/dpo_eval.json
```

### 3. GRPO模型评测

评测强化学习训练后模型的路线规划能力。

**评测维度：**
- 下一步POI预测准确率
- 路线可行性
- 奖励分数
- 与基准策略对比

**运行方式：**
```bash
python src/evaluation/evaluate_grpo.py \
    --model outputs/grpo/qwen3-8b-grpo \
    --test-data outputs/datasets/grpo_planner_prompts.jsonl \
    --max-samples 100 \
    --output outputs/evaluation/grpo_eval.json
```

### 4. 端到端评测

评测整个推荐系统的综合性能。

**评测维度：**
- 成功率（查询处理、路线可行）
- 延迟分解（意图理解、召回、排序、规划）
- 候选数量

**运行方式：**
```bash
python src/evaluation/evaluate_pipeline.py \
    --use-llm \
    --output outputs/evaluation/pipeline_eval.json
```

## 一键评测

运行所有评测：

```bash
bash scripts/evaluate_all.sh \
    --sft-model outputs/sft/qwen3-8b-tourism \
    --dpo-model outputs/dpo/qwen3-8b-dpo \
    --max-samples 100
```

## 评测指标说明

| 指标 | 说明 | 良好阈值 |
|------|------|----------|
| 省份准确率 | 意图理解中省份识别准确率 | >80% |
| 天数MAE | 预测天数与真实天数平均误差 | <1天 |
| POI重叠率 | 生成路线与真实路线POI重叠比例 | >60% |
| 偏好对齐准确率 | DPO模型选择chosen的比例 | >70% |
| 下一POI准确率 | GRPO模型预测下一POI的准确率 | >30% |
| 可行率 | 端到端路线规划可行率 | >80% |
| 平均延迟 | 端到端平均响应时间 | <3秒 |

## 评测结果示例

### SFT评测报告
```
【意图理解】
  省份准确率: 85.00%
  天数MAE: 0.45 天
  兴趣F1: 0.6234
  完全匹配率: 35.00%
  有效JSON率: 92.00%

【路线生成】
  有效路线率: 88.00%
  平均POI数: 12.5
  POI重叠率: 65.00%
  天数匹配率: 82.00%

【文案生成】
  标题生成率: 95.00%
  描述生成率: 88.00%
  平均标题长度: 18.5
  平均描述长度: 52.3
  有效JSON率: 90.00%
```

### DPO评测报告
```
【整体指标】
  偏好对齐准确率: 75.00%
  奖励分数准确率: 68.00%
  平均奖励差: 0.1234

【按来源统计】
  route_template:
    偏好准确率: 78.00%
    奖励准确率: 70.00%
    奖励差: 0.1456
```

### GRPO评测报告
```
【路线规划指标】
  下一POI准确率: 42.00%
  平均奖励: 1.2345
  平均路线重叠度: 55.00%
  平均路线相似度: 48.00%
  有效预测率: 95/100
```

### 端到端评测报告
```
【成功率】
  成功查询: 5/5 (100.0%)
  可行路线: 4/5 (80.0%)

【性能指标】
  平均延迟: 2.35s
  平均候选数: 45
  平均最终POI: 18
  平均路线时长: 7.2h

【延迟分解】
  意图理解: 0.12s
  候选召回: 0.85s
  重排序: 0.65s
  路线规划: 0.73s
```
