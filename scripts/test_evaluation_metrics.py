#!/usr/bin/env python3
"""
测试评测指标功能

演示GoAfar推荐系统评测指标的完整功能
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    auc_score,
    diversity_score,
    novelty_score
)


def test_basic_metrics():
    """测试基础排序指标"""
    print("=" * 60)
    print("测试基础排序指标")
    print("=" * 60)

    # 测试数据
    predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5', 'poi6', 'poi7']
    ground_truth = ['poi2', 'poi4', 'poi6', 'poi8']

    # Recall@K
    print("\n1. Recall@K")
    for k in [3, 5, 7]:
        recall = recall_at_k(predictions, ground_truth, k)
        print(f"   Recall@{k}: {recall:.3f}")

    # NDCG@K
    print("\n2. NDCG@K")
    for k in [3, 5, 7]:
        ndcg = ndcg_at_k(predictions, ground_truth, k)
        print(f"   NDCG@{k}: {ndcg:.3f}")

    # HitRate@K
    print("\n3. HitRate@K")
    for k in [3, 5, 7]:
        hit_rate = hit_rate_at_k(predictions, ground_truth, k)
        print(f"   HitRate@{k}: {hit_rate:.3f}")

    # AUC
    print("\n4. AUC")
    labels = [0, 1, 1, 0, 1, 0, 1]
    scores = [0.2, 0.8, 0.6, 0.3, 0.9, 0.4, 0.7]
    auc = auc_score(labels, scores)
    print(f"   AUC: {auc:.3f}")


def test_diversity_novelty():
    """测试多样性与新颖性"""
    print("\n" + "=" * 60)
    print("测试多样性与新颖性")
    print("=" * 60)

    # 测试数据
    recommendations = [
        ['poi1', 'poi2', 'poi3', 'poi4', 'poi5'],
        ['poi6', 'poi7', 'poi8', 'poi9', 'poi10'],
        ['poi11', 'poi12', 'poi13', 'poi14', 'poi15']
    ]

    # 物品属性
    item_attributes = {
        'poi1': {'category': 'nature', 'province': 'Xinjiang'},
        'poi2': {'category': 'history', 'province': 'Beijing'},
        'poi3': {'category': 'food', 'province': 'Sichuan'},
        'poi4': {'category': 'nature', 'province': 'Tibet'},
        'poi5': {'category': 'history', 'province': 'Shaanxi'},
        'poi6': {'category': 'nature', 'province': 'Yunnan'},
        'poi7': {'category': 'nature', 'province': 'Guizhou'},
        'poi8': {'category': 'history', 'province': 'Henan'},
        'poi9': {'category': 'food', 'province': 'Guangdong'},
        'poi10': {'category': 'nature', 'province': 'Sichuan'},
        'poi11': {'category': 'history', 'province': 'Nanjing'},
        'poi12': {'category': 'history', 'province': 'Xi\'an'},
        'poi13': {'category': 'food', 'province': 'Hunan'},
        'poi14': {'category': 'nature', 'province': 'Qinghai'},
        'poi15': {'category': 'history', 'province': 'Gansu'}
    }

    # 物品流行度
    item_popularity = {
        'poi1': 0.05, 'poi2': 0.30, 'poi3': 0.15, 'poi4': 0.08, 'poi5': 0.25,
        'poi6': 0.10, 'poi7': 0.12, 'poi8': 0.28, 'poi9': 0.18, 'poi10': 0.07,
        'poi11': 0.20, 'poi12': 0.35, 'poi13': 0.22, 'poi14': 0.06, 'poi15': 0.29
    }

    # 多样性
    print("\n1. Diversity (基于类别)")
    diversity_cat = diversity_score(recommendations, item_attributes, attribute_key='category')
    print(f"   Diversity: {diversity_cat:.3f}")

    print("\n2. Diversity (基于省份)")
    diversity_prov = diversity_score(recommendations, item_attributes, attribute_key='province')
    print(f"   Diversity: {diversity_prov:.3f}")

    # 新颖性
    print("\n3. Novelty (K=5)")
    novelty = novelty_score(recommendations, item_popularity, k=5)
    print(f"   Novelty: {novelty:.3f}")

    print("\n4. Novelty (K=10)")
    novelty_k10 = novelty_score(recommendations, item_popularity, k=10)
    print(f"   Novelty: {novelty_k10:.3f}")


def test_batch_evaluation():
    """测试批量评测"""
    print("\n" + "=" * 60)
    print("测试批量评测")
    print("=" * 60)

    # 模拟多个查询的预测结果
    predictions_list = [
        ['poi1', 'poi2', 'poi3', 'poi4', 'poi5'],
        ['poi6', 'poi7', 'poi8', 'poi9', 'poi10'],
        ['poi11', 'poi12', 'poi13', 'poi14', 'poi15']
    ]

    ground_truth_list = [
        ['poi2', 'poi4', 'poi6'],
        ['poi7', 'poi9', 'poi11'],
        ['poi12', 'poi14', 'poi16']
    ]

    # 计算平均指标
    print("\n1. 平均指标 (K=5)")

    recalls = []
    ndcgs = []
    hit_rates = []

    for pred, truth in zip(predictions_list, ground_truth_list):
        recalls.append(recall_at_k(pred, truth, 5))
        ndcgs.append(ndcg_at_k(pred, truth, 5))
        hit_rates.append(hit_rate_at_k(pred, truth, 5))

    import numpy as np

    print(f"   Avg Recall@5: {np.mean(recalls):.3f}")
    print(f"   Avg NDCG@5: {np.mean(ndcgs):.3f}")
    print(f"   Avg HitRate@5: {np.mean(hit_rates):.3f}")

    print(f"\n2. 各查询详细指标")
    for i, (pred, truth) in enumerate(zip(predictions_list, ground_truth_list), 1):
        r = recall_at_k(pred, truth, 5)
        n = ndcg_at_k(pred, truth, 5)
        h = hit_rate_at_k(pred, truth, 5)
        print(f"   Query {i}: Recall={r:.3f}, NDCG={n:.3f}, HitRate={h:.3f}")


def test_pipeline_evaluator():
    """测试流水线评测器"""
    print("\n" + "=" * 60)
    print("测试流水线评测器")
    print("=" * 60)

    from src.evaluation.pipeline_evaluator import PipelineEvaluator

    # 创建评测器
    print("\n1. 创建PipelineEvaluator")
    evaluator = PipelineEvaluator()
    print("   ✓ 评测器创建成功")

    # 测试推荐质量指标计算
    print("\n2. 计算推荐质量指标")

    predictions = [
        ['poi1', 'poi2', 'poi3', 'poi4', 'poi5'],
        ['poi6', 'poi7', 'poi8', 'poi9', 'poi10']
    ]

    ground_truth = [
        ['poi2', 'poi4', 'poi6'],
        ['poi7', 'poi9', 'poi11']
    ]

    item_attributes = {
        'poi1': {'category': 'nature'}, 'poi2': {'category': 'history'},
        'poi3': {'category': 'food'}, 'poi4': {'category': 'nature'},
        'poi5': {'category': 'history'}, 'poi6': {'category': 'nature'},
        'poi7': {'category': 'history'}, 'poi8': {'category': 'food'},
        'poi9': {'category': 'nature'}, 'poi10': {'category': 'history'}
    }

    item_popularity = {
        'poi1': 0.1, 'poi2': 0.3, 'poi3': 0.2, 'poi4': 0.15, 'poi5': 0.25,
        'poi6': 0.12, 'poi7': 0.28, 'poi8': 0.18, 'poi9': 0.14, 'poi10': 0.22
    }

    metrics = evaluator.calculate_recommendation_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        item_attributes=item_attributes,
        item_popularity=item_popularity,
        k_values=[5]
    )

    print("\n   指标结果:")
    for metric, value in sorted(metrics.items()):
        print(f"   {metric}: {value:.4f}")


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("GoAfar 评测指标功能测试")
    print("=" * 80)

    try:
        # 1. 基础指标测试
        test_basic_metrics()

        # 2. 多样性与新颖性测试
        test_diversity_novelty()

        # 3. 批量评测测试
        test_batch_evaluation()

        # 4. 流水线评测器测试
        test_pipeline_evaluator()

        # 总结
        print("\n" + "=" * 80)
        print("✓ 所有测试通过！")
        print("=" * 80)
        print("\n可用的评测指标:")
        print("  • recall_at_k      - 召回率@K")
        print("  • ndcg_at_k        - NDCG@K")
        print("  • hit_rate_at_k    - 命中率@K")
        print("  • auc_score        - AUC分数")
        print("  • diversity_score  - 多样性分数")
        print("  • novelty_score    - 新颖性分数")
        print("\n可用的评测工具:")
        print("  • PipelineEvaluator           - 流水线评测器")
        print("  • generate_evaluation_report  - 生成评测报告")
        print("  • save_evaluation_report      - 保存评测结果")
        print("\n可用的评测脚本:")
        print("  • scripts/evaluate_complete.sh - 完整评测脚本")
        print("  • scripts/evaluate_all.sh      - 全模型评测脚本")
        print("\n文档:")
        print("  • docs/evaluation_metrics.md           - 评测指标文档")
        print("  • docs/evaluation_implementation_summary.md - 实现总结")
        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
