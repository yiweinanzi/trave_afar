# 评测指标文档

GoAfar 推荐系统评测指标完整指���。

## 目录

- [基础排序指标](#基础排序指标)
- [多样性与新颖性](#多样性与新颖性)
- [流水线评测](#流水线评测)
- [使用示例](#使用示例)

---

## 基础排序指标

### Recall@K (召回率)

**定义**: 在前K个推荐中，相关物品占所有相关物品的比例。

**公式**:
```
Recall@K = |推荐_K ∩ 相关| / |相关|
```

**使用场景**:
- 评估召回能力
- K值通常选择 [5, 10, 20, 50]

**示例**:
```python
from src.evaluation.metrics import recall_at_k

predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5']
ground_truth = ['poi2', 'poi4', 'poi6']

recall = recall_at_k(predictions, ground_truth, k=5)
# 返回: 0.667 (2/3)
```

### NDCG@K (归一化折损累积增益)

**定义**: 考虑位置权重的排序质量指标，高排名的相关物品权重更高。

**公式**:
```
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@K = DCG@K / IDCG@K
```

**使用场景**:
- 评估排序质量
- 支持多级相关性

**示例**:
```python
from src.evaluation.metrics import ndcg_at_k

predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5']
ground_truth = ['poi2', 'poi4', 'poi6']

ndcg = ndcg_at_k(predictions, ground_truth, k=5)
# 返回: 0.498
```

### HitRate@K (命中率)

**定义**: 是否在前K个推荐中包含任意相关物品。

**公式**:
```
HitRate@K = 1 if |推荐_K ∩ 相关| > 0 else 0
```

**使用场景**:
- 评估是否命中用户需求
- 适合二值评估

**示例**:
```python
from src.evaluation.metrics import hit_rate_at_k

predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5']
ground_truth = ['poi2', 'poi4', 'poi6']

hit_rate = hit_rate_at_k(predictions, ground_truth, k=5)
# 返回: 1.0 (命中)
```

### AUC (ROC曲线下面积)

**定义**: 排序能力的综合指标，衡量正样本得分高于负样本的概率。

**使用场景**:
- 评估整体排序能力
- 不依赖于截断位置K

**示例**:
```python
from src.evaluation.metrics import auc_score

labels = [0, 1, 1, 0, 1]
scores = [0.2, 0.8, 0.6, 0.3, 0.9]

auc = auc_score(labels, scores)
# 返回: 1.0 (完美排序)
```

---

## 多样性与新颖性

### Diversity Score (多样性)

**定义**: 推荐列表内物品的不相似度（基于类别或其他属性）。

**公式**:
```
ILD(L) = (2 / (|L| * (|L|-1))) * Σ Σ dist(item_i, item_j)
```

**使用场景**:
- 避免推荐过于相似的物品
- 提升用户体验

**示例**:
```python
from src.evaluation.metrics import diversity_score

recommendations = [
    ['poi1', 'poi2', 'poi3'],
    ['poi4', 'poi5', 'poi6']
]
item_attributes = {
    'poi1': {'category': 'nature'},
    'poi2': {'category': 'history'},
    'poi3': {'category': 'food'},
    'poi4': {'category': 'nature'},
    'poi5': {'category': 'nature'},
    'poi6': {'category': 'history'}
}

diversity = diversity_score(recommendations, item_attributes)
# 返回: 0.833
```

### Novelty Score (新颖性)

**定义**: 基于流行度的自信息，越不流行的物品新颖性越高。

**公式**:
```
Novelty = -log2(popularity(item))
```

**使用场景**:
- 促进长尾物品发现
- 避免热门物品偏差

**示例**:
```python
from src.evaluation.metrics import novelty_score

recommendations = [['poi1', 'poi2', 'poi3']]
item_popularity = {
    'poi1': 0.1,  # 低流行度 -> 高新颖性
    'poi2': 0.5,
    'poi3': 0.3
}

novelty = novelty_score(recommendations, item_popularity, k=3)
# 返回: 1.740
```

---

## 流水线评测

### PipelineEvaluator 类

端到端流水线评测器，提供完整的推荐系统评测功能。

**功能**:
- 各路召回贡献分析
- 转化率分析（召回→重排序→路线规划）
- 延迟分解（各模块耗时）
- 推荐质量指标

**使用示例**:
```python
from src.evaluation.pipeline_evaluator import PipelineEvaluator

# 初始化
evaluator = PipelineEvaluator()
evaluator.initialize(use_llm=False)

# 评测单个查询
query = {
    "query": "想去新疆看雪山和草原",
    "province": "新疆",
    "days": 3,
    "interests": ["雪山", "草原"]
}
result = evaluator.evaluate_query(query, track_recall_contributions=True)

# 批量评测
queries = [query1, query2, query3]
evaluation = evaluator.evaluate_batch(queries)

# 生成报告
from src.evaluation.pipeline_evaluator import generate_evaluation_report
report = generate_evaluation_report(evaluation)
print(report)
```

### 召回贡献分析

跟踪各路召回（语义、行为、地理）的贡献度。

**输出示例**:
```
【召回贡献分析】
  语义召回: 44.0 (55.0%)
  行为召回: 24.0 (30.0%)
  地理召回: 12.0 (15.0%)
```

### 转化率分析

分析流水线各阶段的转化率。

**指标**:
- **recall_to_rerank**: 召回→重排序转化率
- **rerank_to_route**: 重排序→路线规划转化率
- **overall_success**: 整体成功率

**输出示例**:
```
【转化率】
  召��→重排序: 37.50%
  重排序→路线: 80.00%
  整体成功率: 30.00%
```

### 延迟分解

分析各模块的耗时占比。

**输出示例**:
```
【延迟分解】
  意图理解: 0.15s (17.6%)
  候选召回: 0.45s (52.9%)
  重排序: 0.20s (23.5%)
  路线规划: 0.05s (5.9%)
  总计: 0.85s
```

---

## 使用示例

### 基础指标评测

```python
from src.evaluation.metrics import (
    recall_at_k, ndcg_at_k, hit_rate_at_k,
    diversity_score, novelty_score
)

# 准备数据
predictions = [['poi1', 'poi2', 'poi3', 'poi4', 'poi5']]
ground_truth = [['poi2', 'poi4', 'poi6']]
item_attributes = {
    'poi1': {'category': 'nature'},
    'poi2': {'category': 'history'},
    'poi3': {'category': 'food'},
    'poi4': {'category': 'nature'},
    'poi5': {'category': 'nature'},
    'poi6': {'category': 'history'}
}
item_popularity = {
    'poi1': 0.1, 'poi2': 0.5, 'poi3': 0.3,
    'poi4': 0.2, 'poi5': 0.4, 'poi6': 0.6
}

# 计算指标
metrics = {}
for k in [5, 10, 20]:
    metrics[f'recall@{k}'] = recall_at_k(predictions[0], ground_truth[0], k)
    metrics[f'ndcg@{k}'] = ndcg_at_k(predictions[0], ground_truth[0], k)
    metrics[f'hitrate@{k}'] = hit_rate_at_k(predictions[0], ground_truth[0], k)

metrics['diversity'] = diversity_score(predictions, item_attributes)
metrics['novelty'] = novelty_score(predictions, item_popularity, k=5)

# 输出结果
for metric, value in metrics.items():
    print(f'{metric}: {value:.4f}')
```

### 使用评测脚本

#### 端到端评测

```bash
# 使用默认测试查询
python src/evaluation/evaluate_pipeline.py --output outputs/evaluation/pipeline_eval.json

# 使用自定义查询文件
python src/evaluation/evaluate_pipeline.py \
    --queries data/test_queries.csv \
    --output outputs/evaluation/pipeline_eval.json

# 使用LLM模式
python src/evaluation/evaluate_pipeline.py \
    --use-llm \
    --output outputs/evaluation/pipeline_eval.json
```

#### 完整评测

```bash
# 运行完整评测脚本
bash scripts/evaluate_complete.sh

# 自定义参数
bash scripts/evaluate_complete.sh \
    --queries data/test_queries.csv \
    --max-samples 100 \
    --output-dir outputs/evaluation \
    --k-values 5,10,20,50
```

#### 全模型评测

```bash
# 评测所有模型（SFT、DPO、GRPO、端到端）
bash scripts/evaluate_all.sh \
    --sft-model outputs/sft/qwen3-8b-tourism \
    --dpo-model outputs/dpo/qwen3-8b-dpo \
    --output-dir outputs/evaluation
```

---

## 指标选择指南

| 目标 | 推荐指标 | 说明 |
|------|----------|------|
| **召回能力** | Recall@K | 评估是否能找到相关物品 |
| **排序质量** | NDCG@K | 评估排序位置是否合理 |
| **命中概率** | HitRate@K | 评估是否至少命中一个相关物品 |
| **整体排序** | AUC | 不依赖K的综合排序能力 |
| **推荐多样性** | Diversity | 避免推荐过于相似 |
| **发现长尾** | Novelty | 促进新颖物品推荐 |
| **性能分析** | Latency Breakdown | 各模块耗时分析 |
| **转化分析** | Conversion Rate | 流水线各阶段转化 |

---

## 常见问题

### Q1: 如何选择K值？

**A**: 根据业务场景选择：
- POI推荐通常选择 K=5,10,20,50
- 如果用户通常浏览较少，使用较小的K（5-10）
- 如果需要更多选择，使用较大的K（20-50）

### Q2: 多样性和准确性的平衡？

**A**: 可以通过调整融合权重来平衡：
- 增加多样性权重 -> 提升diversity，可能降低recall
- 增加相关性权重 -> 提升recall，可能降低diversity

### Q3: 如何解读AUC？

**A**:
- AUC = 1.0: 完美排序
- AUC = 0.5: 随机排序
- AUC < 0.5: 比随机还差（可能需要反转预测）

### Q4: 转化率低怎么办？

**A**: 分析瓶颈：
- `recall_to_rerank`低 -> 召回质量差，需要优化召回策略
- `rerank_to_route`低 -> 排序后的候选不适合路线规划，需要调整约束条件

---

## 参考文献

1. [Shani, G., & Gunawardana, A. (2011). Evaluating recommendation systems.](https://doi.org/10.1007/978-1-4899-7637-6_8)
2. [Kantor, P. B., et al. (2011). Recommender systems handbook.](https://doi.org/10.1007/978-0-387-85820-3)
3. [RecSys Wiki - Offline Evaluation](https://recsyswiki.com/wiki/Offline_Evaluation)
