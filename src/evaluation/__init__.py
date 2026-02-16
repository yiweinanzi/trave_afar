"""
GoAfar评测模块

包含SFT、DPO、GRPO和端到端评测
"""
from .metrics import (
    # 新的标准指标
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    auc_score,
    diversity_score,
    novelty_score,
    # Legacy API
    evaluate_recall,
    evaluate_ndcg,
    evaluate_route_quality,
    evaluate_overall,
    generate_performance_report
)
from .metrics_advanced import (
    # Basic ranking metrics
    precision_at_k,
    f1_score_at_k,
    mean_reciprocal_rank,
    mrr,
    mean_average_precision,
    map_score,
    # Classification metrics
    log_loss_score,
    # Diversity/novelty/coverage
    diversity_score_entropy,
    novelty_score_history,
    serendipity,
    coverage,
    coverage_by_category,
    # Fairness metrics
    demographic_parity,
    demographic_parity_diff,
    equalized_odds,
    equalized_odds_diff,
    disparate_impact,
    # Batch functions
    batch_recall_at_k,
    batch_precision_at_k,
    batch_ndcg_at_k,
    batch_mrr,
    # Evaluation runner
    RecommendationEvaluator,
    # Business metrics
    ctr_auc,
    visit_auc,
    expected_calibration_error,
    brier_score,
    ctr_metrics,
    # Comparison and reporting
    MetricsComparison,
    BusinessMetricsEvaluator,
    # Legacy advanced API
    evaluate_recall as evaluate_recall_advanced,
    evaluate_ndcg as evaluate_ndcg_advanced,
)
from .resume_generator import generate_resume_content

__all__ = [
    # Basic metrics
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "auc_score",
    "diversity_score",
    "novelty_score",
    # Advanced metrics
    "precision_at_k",
    "f1_score_at_k",
    "mean_reciprocal_rank",
    "mrr",
    "mean_average_precision",
    "map_score",
    "log_loss_score",
    "diversity_score_entropy",
    "novelty_score_history",
    "serendipity",
    "coverage",
    "coverage_by_category",
    # Fairness
    "demographic_parity",
    "demographic_parity_diff",
    "equalized_odds",
    "equalized_odds_diff",
    "disparate_impact",
    # Batch functions
    "batch_recall_at_k",
    "batch_precision_at_k",
    "batch_ndcg_at_k",
    "batch_mrr",
    # Evaluator
    "RecommendationEvaluator",
    # Business metrics
    "ctr_auc",
    "visit_auc",
    "expected_calibration_error",
    "brier_score",
    "ctr_metrics",
    # Comparison and reporting
    "MetricsComparison",
    "BusinessMetricsEvaluator",
    # Legacy API
    "evaluate_recall",
    "evaluate_ndcg",
    "evaluate_route_quality",
    "evaluate_overall",
    "generate_performance_report"
]
