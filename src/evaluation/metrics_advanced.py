"""
Advanced Evaluation Metrics for Recommendation Systems.

Implements standard recommendation metrics:
- Recall@K, Precision@K, NDCG@K, MRR
- HitRate, AUC
- Diversity, Novelty, Serendipity
- Fairness metrics

Compatible with GoAfar POI recommendation system.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Basic Ranking Metrics
# ============================================================================

def recall_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Recall score (0-1)
    """
    if not ground_truth:
        return 0.0

    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return len(pred_k & true_set) / len(true_set)


def precision_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Precision score (0-1)
    """
    if k == 0:
        return 0.0

    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return len(pred_k & true_set) / k


def ndcg_at_k(
    predictions: List[str],
    ground_truth: List[str] | Dict[str, float],
    k: int,
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Relevant items (list) or {item: relevance} dict
        k: Cutoff position

    Returns:
        NDCG score (0-1)
    """
    # Build relevance dictionary
    if isinstance(ground_truth, dict):
        relevance = ground_truth
    else:
        relevance = {item: 1.0 for item in ground_truth}

    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(predictions[:k]):
        rel = relevance.get(item, 0.0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    # Compute IDCG (ideal ranking)
    sorted_relevance = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(sorted_relevance))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mean_reciprocal_rank(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items

    Returns:
        MRR score (0-1)
    """
    true_set = set(ground_truth)
    for i, item in enumerate(predictions):
        if item in true_set:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute HitRate@K (whether any relevant item appears in top-K).

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Hit rate (0 or 1)
    """
    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return 1.0 if len(pred_k & true_set) > 0 else 0.0


def auc_score(
    labels: List[int] | np.ndarray,
    scores: List[float] | np.ndarray,
) -> float:
    """
    Compute Area Under ROC Curve (AUC).

    Args:
        labels: Binary labels (0 or 1)
        scores: Prediction scores

    Returns:
        AUC score (0-1)
    """
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels, scores)
    except ImportError:
        # Simple implementation
        labels = np.array(labels)
        scores = np.array(scores)

        # Sort by score
        order = np.argsort(scores)[::-1]
        labels = labels[order]

        # Compute AUC
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Count ranks of positive items
        rank_sum = 0.0
        pos_count = 0
        for i, label in enumerate(labels):
            if label == 1:
                rank_sum += i + 1
                pos_count += 1

        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return max(0.0, min(1.0, auc))


# ============================================================================
# Beyond Accuracy: Diversity, Novelty, Serendipity
# ============================================================================

def diversity_score(
    recommendations: List[List[str]],
    item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    attribute_key: str = "category",
) -> float:
    """
    Compute average intra-list diversity (ILD).

    Measures how diverse items are within each recommendation list.

    Args:
        recommendations: List of recommendation lists
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to use for diversity

    Returns:
        Average diversity score (0-1)
    """
    if item_attributes is None:
        # Use simple Jaccard distance
        def distance(a, b): return a != b
    else:
        # Use attribute distance
        def distance(a, b):
            attr_a = item_attributes.get(a, {}).get(attribute_key, "")
            attr_b = item_attributes.get(b, {}).get(attribute_key, "")
            return 0 if attr_a == attr_b else 1

    total_diversity = 0.0
    total_lists = 0

    for rec_list in recommendations:
        if len(rec_list) < 2:
            continue

        # Compute pairwise distance
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
    k: int = 10,
) -> float:
    """
    Compute average novelty (inverse popularity).

    Args:
        recommendations: List of recommendation lists
        item_popularity: {item: popularity_score} dictionary
        k: Consider top-K items

    Returns:
        Average novelty score (higher = more novel)
    """
    total_novelty = 0.0
    total_items = 0

    for rec_list in recommendations:
        for item in rec_list[:k]:
            if item in item_popularity:
                # -log2(popularity) for self-information
                pop = item_popularity[item]
                novelty = -math.log2(pop + 1e-10)
                total_novelty += novelty
                total_items += 1

    return total_novelty / total_items if total_items > 0 else 0.0


def serendipity(
    recommendations: List[str],
    user_history: List[str],
    item_similarities: Dict[Tuple[str, str], float],
    k: int = 10,
) -> float:
    """
    Compute serendipity (unexpected but relevant recommendations).

    Args:
        recommendations: Ordered list of recommended items
        user_history: User's past interactions
        item_similarities: {(item1, item2): similarity} dictionary
        k: Consider top-K items

    Returns:
        Serendipity score (0-1)
    """
    unexpectedness = 0.0
    relevant_count = 0

    for item in recommendations[:k]:
        # Unexpectedness: 1 - max similarity to history
        max_sim = 0.0
        for hist_item in user_history:
            sim = item_similarities.get((item, hist_item), 0.0)
            max_sim = max(max_sim, sim)

        # For serendipity, we'd need relevance feedback
        # Here we assume all recs are potentially relevant
        unexpectedness += 1 - max_sim
        relevant_count += 1

    return unexpectedness / relevant_count if relevant_count > 0 else 0.0


def coverage(
    all_recommendations: List[List[str]],
    catalog_size: int,
) -> float:
    """
    Compute catalog coverage (what % of items are ever recommended).

    Args:
        all_recommendations: List of recommendation lists
        catalog_size: Total number of items in catalog

    Returns:
        Coverage ratio (0-1)
    """
    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)

    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


# ============================================================================
# Fairness Metrics
# ============================================================================

def demographic_parity(
    group_recommendations: Dict[str, List[str]],
    group_attributes: Dict[str, str],
    attribute_values: List[str],
) -> Dict[str, float]:
    """
    Compute demographic parity across groups.

    Args:
        group_recommendations: {group_id: [recommended_items]}
        group_attributes: {item: attribute_value}
        attribute_values: List of attribute values to check

    Returns:
        {attribute_value: avg_recommendation_count}
    """
    group_counts = defaultdict(list)

    for group_id, recs in group_recommendations.items():
        for attr_val in attribute_values:
            count = sum(1 for item in recs if group_attributes.get(item) == attr_val)
            group_counts[attr_val].append(count)

    return {
        attr_val: np.mean(counts) if counts else 0.0
        for attr_val, counts in group_counts.items()
    }


def equalized_odds(
    predictions: List[str],
    ground_truth: List[str],
    group_ids: List[str],
) -> Dict[str, float]:
    """
    Compute equalized odds (TPR parity across groups).

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        group_ids: Group membership for each predicted item

    Returns:
        {group_id: true_positive_rate}
    """
    true_set = set(ground_truth)
    tpr_by_group = defaultdict(list)

    for item, group_id in zip(predictions, group_ids):
        if item in true_set:
            tpr_by_group[group_id].append(1.0)
        else:
            tpr_by_group[group_id].append(0.0)

    return {
        group_id: np.mean(rates) if rates else 0.0
        for group_id, rates in tpr_by_group.items()
    }


# ============================================================================
# Multi-Task Metrics
# ============================================================================

def ctr_metrics(
    predictions: List[str],
    clicks: List[int],
    impressions: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Click-Through Rate metrics.

    Args:
        predictions: Predicted item IDs
        clicks: Binary click indicators (0 or 1)
        impressions: Number of impressions (default all 1)

    Returns:
        {metric: value} dictionary
    """
    clicks = np.array(clicks)
    if impressions is None:
        impressions = np.ones_like(clicks)
    else:
        impressions = np.array(impressions)

    return {
        "ctr": np.sum(clicks) / np.sum(impressions),
        "avg_ctr": np.mean(clicks / (impressions + 1e-10)),
        "clicks": int(np.sum(clicks)),
        "impressions": int(np.sum(impressions)),
    }


# ============================================================================
# Evaluation Runner
# ============================================================================

class RecommendationEvaluator:
    """
    Comprehensive evaluation runner for recommendation systems.

    Usage:
        evaluator = RecommendationEvaluator()
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            metrics=["recall@10", "ndcg@10", "diversity", "coverage"]
        )
    """

    def __init__(self, k_values: List[int] = [5, 10, 20, 50]):
        self.k_values = k_values

    def evaluate(
        self,
        predictions: List[List[str]] | List[str],
        ground_truth: List[List[str]] | List[str],
        item_attributes: Optional[Dict[str, Dict]] = None,
        item_popularity: Optional[Dict[str, float]] = None,
        catalog_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation.

        Args:
            predictions: List of recommendation lists or single list
            ground_truth: Corresponding relevant items
            item_attributes: For diversity calculation
            item_popularity: For novelty calculation
            catalog_size: For coverage calculation

        Returns:
            Dictionary of all computed metrics
        """
        # Normalize inputs
        single_query = isinstance(predictions[0], str)
        if single_query:
            predictions = [predictions]
            ground_truth = [ground_truth]

        results = {}

        # Basic ranking metrics
        for k in self.k_values:
            recalls = [recall_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"recall@{k}"] = np.mean(recalls)

            precisions = [precision_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"precision@{k}"] = np.mean(precisions)

            ndcgs = [ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"ndcg@{k}"] = np.mean(ndcgs)

            hit_rates = [hit_rate_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"hitrate@{k}"] = np.mean(hit_rates)

        # MRR
        mrrs = [mean_reciprocal_rank(pred, truth) for pred, truth in zip(predictions, ground_truth)]
        results["mrr"] = np.mean(mrrs)

        # Diversity
        if item_attributes is not None:
            results["diversity"] = diversity_score(predictions, item_attributes)

        # Novelty
        if item_popularity is not None:
            results["novelty"] = novelty_score(predictions, item_popularity, k=10)

        # Coverage
        if catalog_size is not None:
            results["coverage"] = coverage(predictions, catalog_size)

        return results

    def format_report(self, results: Dict[str, float]) -> str:
        """Format evaluation results as a readable report."""
        lines = ["=" * 60, "Recommendation Evaluation Report", "=" * 60]

        # Group by metric type
        ranking_metrics = [k for k in results if "@" in k or k in ["mrr", "auc"]]
        quality_metrics = [k for k in results if k in ["diversity", "novelty", "coverage", "serendipity"]]
        other_metrics = [k for k in results if k not in ranking_metrics + quality_metrics]

        if ranking_metrics:
            lines.append("\n[Ranking Metrics]")
            for k in sorted(ranking_metrics):
                lines.append(f"  {k}: {results[k]:.4f}")

        if quality_metrics:
            lines.append("\n[Quality Metrics]")
            for k in sorted(quality_metrics):
                lines.append(f"  {k}: {results[k]:.4f}")

        if other_metrics:
            lines.append("\n[Other Metrics]")
            for k in sorted(other_metrics):
                lines.append(f"  {k}: {results[k]}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Legacy API Compatibility
# ============================================================================

def evaluate_recall(predictions, ground_truth, k_list=[10, 20, 50]):
    """Legacy wrapper for recall evaluation."""
    results = {}
    for k in k_list:
        results[f"Recall@{k}"] = recall_at_k(predictions, ground_truth, k)
    return results


def evaluate_ndcg(predictions_with_scores, ground_truth_with_scores, k_list=[10, 20]):
    """Legacy wrapper for NDCG evaluation."""
    results = {}
    predictions = [item for item, _ in predictions_with_scores]
    relevance = dict(ground_truth_with_scores)

    for k in k_list:
        results[f"NDCG@{k}"] = ndcg_at_k(predictions, relevance, k)
    return results


def evaluate_route_quality(route_solution, max_duration):
    """Legacy wrapper for route quality evaluation."""
    if route_solution is None:
        return {"feasible": 0.0, "duration_utilization": 0.0, "num_visited": 0}

    return {
        "feasible": 1.0,
        "duration_utilization": route_solution.get("total_hours", 0) / max_duration,
        "num_visited": route_solution.get("visited_pois", 0),
    }


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced recommendation metrics")
    parser.add_argument("--predictions", required=True, help="JSON file with predictions")
    parser.add_argument("--ground-truth", required=True, help="JSON file with ground truth")
    parser.add_argument("--output", help="Output JSON for results")
    args = parser.parse_args()

    # Load data
    import json

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Evaluate
    evaluator = RecommendationEvaluator()
    results = evaluator.evaluate(predictions, ground_truth)

    # Print report
    print(evaluator.format_report(results))

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
