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
from .resume_generator import generate_resume_content

__all__ = [
    # 新的标准指标
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "auc_score",
    "diversity_score",
    "novelty_score",
    # Legacy API
    "evaluate_recall",
    "evaluate_ndcg",
    "evaluate_route_quality",
    "evaluate_overall",
    "generate_performance_report"
]
