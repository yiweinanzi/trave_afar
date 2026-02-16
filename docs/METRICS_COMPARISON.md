# GoAfar 评测指标对比报告

## 概述

本文档对比 GoAfar 推荐系统中不同评测指标的特性、适用场景和实现方法。

---

## 1. 召回指标 (Recall Metrics)

### Recall@K
- **定义**: Top-K 推荐中相关项目的覆盖率
- **公式**: `|recalled_items| / |relevant_items|`
- **范围**: 0-1, 越高越好
- **用途**: 衡量系统覆盖用户兴趣的能力

### HitRate@K
- **定义**: Top-K 推荐中是否包含至少一个相关项目
- **范围**: 0 或 1
- **用途**: 二元化的召回指标

### Precision@K
- **定义**: Top-K 推荐中相关项目的比例
- **公式**: `|relevant_items_in_top_k| / k`
- **范围**: 0-1, 越高越好
- **用途**: 衡量推荐列表的精确度

**对比**:
| 指标 | 关注点 | 使用场景 |
|-------|--------|----------|
| Recall@K | 覆盖率 | 内容池大、用户兴趣广泛 |
| Precision@K | 精确度 | 展示位有限、用户体验敏感 |
| HitRate@K | 是否命中 | 简化评估、快速迭代 |

---

## 2. 排序指标 (Ranking Metrics)

### NDCG@K (Normalized Discounted Cumulative Gain)
- **定义**: 考虑位置权重的排序质量
- **特点**: 位置越靠前，权重越高
- **用途**: 评估排序质量，适用于有相关性分数的场景

### MRR (Mean Reciprocal Rank)
- **定义**: 第一个相关项目的倒数排名
- **公式**: `1 / position_of_first_relevant`
- **用途**: 关注"首个命中"的场景，如搜索

### MAP (Mean Average Precision)
- **定义**: 各召回位置精确率的平均值
- **用途**: 综合评估所有相关项目的排序质量

**对比**:
| 指标 | 优点 | 缺点 |
|-------|------|------|
| NDCG@K | 考虑位置权重、支持多级相关性 | 计算复杂 |
| MRR | 简单直观、关注首位 | 仅考虑第一个相关项 |
| MAP | 综合考虑所有相关项 | 对噪声敏感 |

---

## 3. 多样性指标 (Diversity Metrics)

### Diversity Score (ILD - Intra-List Diversity)
- **定义**: 推荐列表内项目的平均差异度
- **计算**: 成对距离的平均值
- **用途**: 确保推荐不单调

### Shannon Entropy Diversity
- **定义**: 基于类别的熵值
- **用途**: 衡量类别分布的均衡性

**对比**:
| 指标 | 适用场景 |
|-------|----------|
| ILD | 需要项目间相似度计算 |
| Entropy | 基于类别标签、计算更快 |

---

## 4. 业务指标 (Business Metrics)

### CTR AUC
- **定义**: 点击率预测的AUC
- **用途**: 评估CTR模型区分点击/非点击的能力

### Visit AUC
- **定义**: 到访率预测的AUC
- **用途**: 评估到访预测能力

### ECE (Expected Calibration Error)
- **定义**: 预测概率与实际频率的加权误差
- **用途**: 评估概率校准质量，越低越好

### Brier Score
- **定义**: 概率预测的均方误差
- **用途**: 概率预测精度评估

**对比**:
| 指标 | 关注点 | 理想值 |
|-------|--------|--------|
| CTR AUC | 排序能力 | 1.0 |
| Visit AUC | 到访预测 | 1.0 |
| ECE | 校准误差 | 0.0 |
| Brier Score | 概率精度 | 0.0 |

---

## 5. 公平性指标 (Fairness Metrics)

### Demographic Parity
- **定义**: 不同群体接收推荐的平均数量
- **用途**: 检测推荐偏差

### Equalized Odds
- **定义**: 不同群体的真阳性率
- **用途**: 评估分类公平性

### Disparate Impact
- **定义**: 群体间推荐比例
- **用途**: 法律合规性检查

---

## 6. 指标使用建议

| 场景 | 推荐指标 |
|------|----------|
| 旅游POI推荐 | Recall@10, NDCG@10, Diversity |
| 个性化排序 | NDCG@10, MAP, MRR |
| 冷启动评估 | Precision@5, HitRate@5 |
| CTR预测 | CTR AUC, ECE, Brier Score |
| 多样性优化 | Diversity Entropy, Coverage |
| 公平性审计 | Demographic Parity Diff |

---

## 7. 代码示例

```python
from src.evaluation.metrics_advanced import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mrr,
    map_score,
    diversity_score_entropy,
    ctr_auc,
    visit_auc,
    expected_calibration_error,
    MetricsComparison,
    RecommendationEvaluator,
)

# 单指标计算
predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5']
ground_truth = ['poi2', 'poi4', 'poi6']

recall = recall_at_k(predictions, ground_truth, k=5)  # 0.667
ndcg = ndcg_at_k(predictions, ground_truth, k=5)      # ~0.83
mrr = mrr(predictions, ground_truth)                   # 0.5

# 综合评估
evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
results = evaluator.evaluate(
    predictions=[predictions],
    ground_truth=[ground_truth],
    item_attributes=item_attrs,
    item_popularity=popularity,
    catalog_size=len(all_items)
)

# 模型对比
comparison = MetricsComparison()
comparison.add_result("baseline", baseline_metrics)
comparison.add_result("model_v1", v1_metrics)
comparison.add_result("model_v2", v2_metrics)
report = comparison.generate_comparison_report(baseline="baseline")
```

---

## 8. 指标 vs 现有指标

### metrics.py 中的基础指标:
- `recall_at_k` - 基础召回率
- `ndcg_at_k` - 基础NDCG
- `hit_rate_at_k` - 基础命中率
- `auc_score` - 基础AUC
- `diversity_score` - 基础多样性
- `novelty_score` - 基础新颖性

### metrics_advanced.py 中的新增指标:
- **扩展排序指标**: `precision_at_k`, `f1_score_at_k`, `mrr`, `map_score`
- **业务指标**: `ctr_auc`, `visit_auc`, `ece`, `brier_score`
- **多样性扩展**: `diversity_score_entropy`, `serendipity`
- **公平性指标**: `demographic_parity`, `equalized_odds`, `disparate_impact`
- **对比工具**: `MetricsComparison`, `BusinessMetricsEvaluator`
- **综合评估器**: `RecommendationEvaluator`

---

## 9. 总结

| 类别 | 指标数量 | 核心指标 |
|------|----------|----------|
| 召回类 | 3 | Recall@K, Precision@K, HitRate@K |
| 排序类 | 3 | NDCG@K, MRR, MAP |
| 多样性 | 4 | ILD, Entropy, Serendipity, Coverage |
| 业务类 | 4 | CTR AUC, Visit AUC, ECE, Brier |
| 公平性 | 3 | Demographic Parity, Equalized Odds, Disparate Impact |

**总计**: 17+ 种推荐系统标准指标
