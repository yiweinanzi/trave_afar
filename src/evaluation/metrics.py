"""
评测指标计算
包含召回率、准确率、路线质量等指标
"""
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

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
        predictions_with_scores: [(poi_id, score), ...]
        ground_truth_with_scores: [(poi_id, relevance), ...]
        k_list: K值列表
    
    Returns:
        dict: 各K值的NDCG
    """
    results = {}
    
    # 构建相关性字典
    relevance_dict = {poi_id: score for poi_id, score in ground_truth_with_scores}
    
    # 为预测结果分配相关性分数
    y_true = []
    y_pred = []
    
    for poi_id, score in predictions_with_scores:
        y_pred.append(score)
        y_true.append(relevance_dict.get(poi_id, 0.0))
    
    for k in k_list:
        if len(y_true) >= k and len(y_pred) >= k:
            try:
                ndcg = ndcg_score([y_true[:k]], [y_pred[:k]])
                results[f'NDCG@{k}'] = ndcg
            except:
                results[f'NDCG@{k}'] = 0.0
        else:
            results[f'NDCG@{k}'] = 0.0
    
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

