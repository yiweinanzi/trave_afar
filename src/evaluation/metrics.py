"""
评测指标计算
包含召回率、准确率、路线质量等指标
"""
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from sklearn.metrics import ndcg_score, roc_auc_score

# ============================================================================
# 基础排序指标
# ============================================================================

def recall_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int
) -> float:
    """
    计算Recall@K

    Args:
        predictions: 预测的POI ID列表
        ground_truth: 真实相关的POI ID列表
        k: 截断位置

    Returns:
        召回率分数(0-1)
    """
    if not ground_truth:
        return 0.0

    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return len(pred_k & true_set) / len(true_set)


def ndcg_at_k(
    predictions: List[str],
    ground_truth: Union[List[str], Dict[str, float]],
    k: int
) -> float:
    """
    计算NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        predictions: 预测的POI ID列表
        ground_truth: 相关POI列表 或 {poi_id: relevance}字典
        k: 截断位置

    Returns:
        NDCG分数(0-1)
    """
    # 构建相关性字典
    if isinstance(ground_truth, dict):
        relevance = ground_truth
    else:
        relevance = {item: 1.0 for item in ground_truth}

    # 计算DCG
    dcg = 0.0
    for i, item in enumerate(predictions[:k]):
        rel = relevance.get(item, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    # 计算IDCG (理想排序)
    sorted_relevance = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted_relevance))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int
) -> float:
    """
    计算HitRate@K (命中是否有任意相关项)

    Args:
        predictions: 预测的POI ID列表
        ground_truth: 真实相关的POI ID列表
        k: 截断位置

    Returns:
        命中率(0或1)
    """
    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return 1.0 if len(pred_k & true_set) > 0 else 0.0


def auc_score(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray]
) -> float:
    """
    计算AUC (Area Under ROC Curve)

    Args:
        labels: 二元标签(0或1)
        scores: 预测分数

    Returns:
        AUC分数(0-1)
    """
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        # 处理只有一个类别的情况
        return 0.5


# ============================================================================
# 多样性、新颖性指标
# ============================================================================

def diversity_score(
    recommendations: List[List[str]],
    item_attributes: Optional[Dict[str, Dict[str, any]]] = None,
    attribute_key: str = "category"
) -> float:
    """
    计算平均列表内多样性 (Intra-List Diversity)

    衡量推荐列表内物品的多样性程度

    Args:
        recommendations: 推荐列表的列表
        item_attributes: {item_id: {attr: value}} 字典
        attribute_key: 用于计算多样性的属性键

    Returns:
        平均多样性分数(0-1)
    """
    if item_attributes is None:
        # 使用简单的Jaccard距离
        def distance(a, b):
            return 1.0 if a != b else 0.0
    else:
        # 使用属性距离
        def distance(a, b):
            attr_a = item_attributes.get(a, {}).get(attribute_key, "")
            attr_b = item_attributes.get(b, {}).get(attribute_key, "")
            return 0.0 if attr_a == attr_b else 1.0

    total_diversity = 0.0
    total_lists = 0

    for rec_list in recommendations:
        if len(rec_list) < 2:
            continue

        # 计算成对距离
        pair_count = 0
        pair_distance = 0.0
        for i in range(len(rec_list)):
            for j in range(i + 1, len(rec_list)):
                pair_distance += distance(rec_list[i], rec_list[j])
                pair_count += 1

        if pair_count > 0:
            total_diversity += pair_distance / pair_count
            total_lists += 1

    return total_diversity / total_lists if total_lists > 0 else 0.0


def novelty_score(
    recommendations: List[List[str]],
    item_popularity: Dict[str, float],
    k: int = 10
) -> float:
    """
    计算平均新颖性 (基于流行度的自信息)

    Args:
        recommendations: 推荐列表的列表
        item_popularity: {item_id: popularity_score} 字典
        k: 考虑前K个物品

    Returns:
        平均新颖性分数(越高越新颖)
    """
    total_novelty = 0.0
    total_items = 0

    for rec_list in recommendations:
        for item in rec_list[:k]:
            if item in item_popularity:
                # -log2(popularity) 计算自信息
                pop = item_popularity[item]
                novelty = -math.log2(pop + 1e-10)
                total_novelty += novelty
                total_items += 1

    return total_novelty / total_items if total_items > 0 else 0.0


# ============================================================================
# Legacy API - 保持向后兼容
# ============================================================================

def evaluate_recall(predictions, ground_truth, k_list=[10, 20, 50]):
    """
    评测召回率 Recall@K
    
    Args:
        predictions: 预测的POI ID列表
        ground_truth: 真实相关的POI ID列表
        k_list: K值列表
    
    Returns:
        dict: 各K值的召回率
    """
    results = {}
    
    for k in k_list:
        pred_k = set(predictions[:k])
        true_set = set(ground_truth)
        
        if len(true_set) == 0:
            recall = 0.0
        else:
            recall = len(pred_k & true_set) / len(true_set)
        
        results[f'Recall@{k}'] = recall
    
    return results

def evaluate_ndcg(predictions_with_scores, ground_truth_with_scores, k_list=[10, 20]):
    """
    评测NDCG@K (Normalized Discounted Cumulative Gain)

    Args:
        predictions_with_scores: [(poi_id, score), ...] 或简单的poi_id列表
        ground_truth_with_scores: [(poi_id, relevance), ...] 或简单的poi_id列表
        k_list: K值列表

    Returns:
        dict: 各K值的NDCG
    """
    results = {}

    # 支持简单的poi_id列表格式
    if predictions_with_scores and isinstance(predictions_with_scores[0], str):
        predictions = predictions_with_scores
    else:
        predictions = [poi_id for poi_id, _ in predictions_with_scores]

    # 构建相关性字典
    if ground_truth_with_scores and isinstance(ground_truth_with_scores[0], str):
        relevance = {poi_id: 1.0 for poi_id in ground_truth_with_scores}
    else:
        relevance = {poi_id: score for poi_id, score in ground_truth_with_scores}

    # 使用新的ndcg_at_k函数
    for k in k_list:
        results[f'NDCG@{k}'] = ndcg_at_k(predictions, relevance, k)

    return results

def evaluate_route_quality(route_solution, max_duration):
    """
    评测路线质量
    
    Args:
        route_solution: VRPTW求解结果
        max_duration: 最大允许时长（小时）
    
    Returns:
        dict: 路线质量指标
    """
    metrics = {}
    
    # 可行性
    metrics['feasible'] = route_solution is not None
    
    if route_solution:
        # 时长利用率
        metrics['duration_utilization'] = route_solution['total_hours'] / max_duration
        
        # 访问景点数
        metrics['num_visited'] = route_solution['visited_pois']
        
        # 平均每景点时长
        if metrics['num_visited'] > 0:
            metrics['avg_time_per_poi'] = route_solution['total_hours'] / metrics['num_visited']
        else:
            metrics['avg_time_per_poi'] = 0
        
        # 目标函数值（越小越好）
        metrics['objective_value'] = route_solution['objective_value']
    else:
        metrics['duration_utilization'] = 0
        metrics['num_visited'] = 0
        metrics['avg_time_per_poi'] = 0
        metrics['objective_value'] = float('inf')
    
    return metrics

def evaluate_overall(test_queries, results):
    """
    综合评测
    
    Args:
        test_queries: 测试查询列表
        results: 推荐结果列表
    
    Returns:
        dict: 综合评测指标
    """
    metrics = {
        'total_queries': len(test_queries),
        'successful_recommendations': 0,
        'avg_response_time': 0,
        'avg_num_pois': 0,
        'avg_route_hours': 0,
        'feasibility_rate': 0
    }
    
    successful_results = [r for r in results if 'error' not in r]
    metrics['successful_recommendations'] = len(successful_results)
    metrics['feasibility_rate'] = len(successful_results) / len(test_queries) if test_queries else 0
    
    if successful_results:
        metrics['avg_num_pois'] = np.mean([r['num_pois'] for r in successful_results])
        metrics['avg_route_hours'] = np.mean([r['total_hours'] for r in successful_results])
    
    return metrics

def generate_performance_report(system_metrics):
    """
    生成性能报告（用于简历）
    
    Args:
        system_metrics: 系统性能指标字典
    
    Returns:
        str: 格式化的性能报告
    """
    report = []
    
    report.append("="*80)
    report.append("GoAfar 系统性能评测报告")
    report.append("="*80)
    
    report.append("\n【召回性能】")
    if 'Recall@50' in system_metrics:
        report.append(f"  Recall@50: {system_metrics['Recall@50']:.2%}")
    if 'NDCG@10' in system_metrics:
        report.append(f"  NDCG@10: {system_metrics['NDCG@10']:.4f}")
    
    report.append("\n【路线质量】")
    report.append(f"  可行率: {system_metrics.get('feasibility_rate', 0):.2%}")
    report.append(f"  平均景点数: {system_metrics.get('avg_num_pois', 0):.1f}")
    report.append(f"  平均时长: {system_metrics.get('avg_route_hours', 0):.1f}小时")
    
    report.append("\n【系统性能】")
    if 'gpu_speedup' in system_metrics:
        report.append(f"  GPU加速比: {system_metrics['gpu_speedup']:.0f}x")
    if 'vector_generation_speed' in system_metrics:
        report.append(f"  向量生成速度: {system_metrics['vector_generation_speed']:.1f} POI/秒")
    if 'query_latency' in system_metrics:
        report.append(f"  查询延迟: {system_metrics['query_latency']:.2f}秒")
    
    report.append("\n" + "="*80)
    
    return '\n'.join(report)

if __name__ == "__main__":
    # 测试评测指标
    print("="*60)
    print("测试评测指标")
    print("="*60)
    
    # 测试召回率
    predictions = ['POI_001', 'POI_002', 'POI_003', 'POI_004', 'POI_005']
    ground_truth = ['POI_002', 'POI_004', 'POI_006', 'POI_008']
    
    recall_metrics = evaluate_recall(predictions, ground_truth, k_list=[3, 5])
    print(f"\n召回率: {recall_metrics}")
    
    # 测试路线质量
    mock_solution = {
        'total_hours': 7.5,
        'visited_pois': 5,
        'objective_value': 1000
    }
    
    quality_metrics = evaluate_route_quality(mock_solution, max_duration=8)
    print(f"\n路线质量: {quality_metrics}")
    
    # 生成报告
    system_metrics = {
        'Recall@50': 0.75,
        'NDCG@10': 0.82,
        'feasibility_rate': 0.92,
        'avg_num_pois': 5.3,
        'avg_route_hours': 7.8,
        'gpu_speedup': 600,
        'vector_generation_speed': 669.7,
        'query_latency': 0.85
    }
    
    report = generate_performance_report(system_metrics)
    print(f"\n{report}")

