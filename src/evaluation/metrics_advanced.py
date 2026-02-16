# -*- coding: utf-8 -*-
"""
Advanced Evaluation Metrics for Recommendation Systems.

Implements standard recommendation metrics:
- Recall@K, Precision@K, NDCG@K, MRR
- HitRate, AUC, Log Loss
- Diversity, Novelty, Serendipity, Coverage
- Fairness metrics (demographic parity, equalized odds, disparate impact)

Compatible with GoAfar POI recommendation system.

Author: GoAfar Team
Date: 2025

Usage:
    from src.evaluation.metrics_advanced import (
        recall_at_k, precision_at_k, ndcg_at_k, mrr,
        diversity_score, novelty_score, coverage,
        RecommendationEvaluator
    )

    # Single query evaluation
    predictions = ['poi1', 'poi2', 'poi3', 'poi4', 'poi5']
    ground_truth = ['poi2', 'poi4', 'poi6']
    recall = recall_at_k(predictions, ground_truth, k=5)

    # Batch evaluation
    evaluator = RecommendationEvaluator(k_values=[5, 10, 20])
    results = evaluator.evaluate(
        predictions=[predictions],
        ground_truth=[ground_truth]
    )
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

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

    Recall@K measures the fraction of relevant items that are retrieved
    in the top-K recommendations.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Recall score (0-1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3', 'item4', 'item5']
        >>> ground_truth = ['item2', 'item4', 'item6']
        >>> recall_at_k(predictions, ground_truth, k=5)
        0.666...  # 2 out of 3 relevant items found
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

    Precision@K measures the fraction of recommended items that are relevant
    in the top-K recommendations.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Precision score (0-1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3', 'item4', 'item5']
        >>> ground_truth = ['item2', 'item4', 'item6']
        >>> precision_at_k(predictions, ground_truth, k=5)
        0.4  # 2 out of 5 items are relevant
    """
    if k == 0:
        return 0.0

    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return len(pred_k & true_set) / k


def f1_score_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute F1 Score@K.

    F1 is the harmonic mean of precision and recall.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        F1 score (0-1)
    """
    precision = precision_at_k(predictions, ground_truth, k)
    recall = recall_at_k(predictions, ground_truth, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(
    predictions: List[str],
    ground_truth: Union[List[str], Dict[str, float]],
    k: int,
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG measures the quality of ranking by considering the position of
    relevant items. Items higher in the list contribute more to the score.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Relevant items (list) or {item: relevance} dict
        k: Cutoff position

    Returns:
        NDCG score (0-1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3', 'item4']
        >>> ground_truth = ['item2', 'item4']
        >>> ndcg_at_k(predictions, ground_truth, k=4)
        # Returns normalized score based on position
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

    MRR measures the rank of the first relevant item.
    Higher is better (1.0 = first item is relevant).

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items

    Returns:
        MRR score (0-1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3']
        >>> ground_truth = ['item3', 'item5']
        >>> mean_reciprocal_rank(predictions, ground_truth)
        0.333...  # First relevant item is at position 3
    """
    true_set = set(ground_truth)
    for i, item in enumerate(predictions):
        if item in true_set:
            return 1.0 / (i + 1)
    return 0.0


def mrr(predictions: List[str], ground_truth: List[str]) -> float:
    """Alias for mean_reciprocal_rank."""
    return mean_reciprocal_rank(predictions, ground_truth)


def hit_rate_at_k(
    predictions: List[str],
    ground_truth: List[str],
    k: int,
) -> float:
    """
    Compute HitRate@K (whether any relevant item appears in top-K).

    HitRate is a binary metric that checks if at least one relevant item
    appears in the top-K recommendations.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        k: Cutoff position

    Returns:
        Hit rate (0 or 1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3']
        >>> ground_truth = ['item4', 'item5']
        >>> hit_rate_at_k(predictions, ground_truth, k=5)
        0.0  # No relevant items found
    """
    pred_k = set(predictions[:k])
    true_set = set(ground_truth)
    return 1.0 if len(pred_k & true_set) > 0 else 0.0


def mean_average_precision(
    predictions: List[str],
    ground_truth: List[str],
) -> float:
    """
    Compute Mean Average Precision (MAP).

    AP is the average of precision scores at each position
    where a relevant item is found.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items

    Returns:
        MAP score (0-1)

    Example:
        >>> predictions = ['a', 'b', 'c', 'd', 'e']
        >>> ground_truth = ['b', 'd', 'f']
        >>> # Precision@2 = 0.5 (b is relevant), Precision@4 = 0.5 (b,d relevant)
        >>> # AP = (0.5 + 0.5) / 2 = 0.5
    """
    true_set = set(ground_truth)
    if not true_set:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for i, item in enumerate(predictions):
        if item in true_set:
            hit_count += 1
            precision_sum += hit_count / (i + 1)

    return precision_sum / len(true_set)


def map_score(predictions: List[str], ground_truth: List[str]) -> float:
    """Alias for mean_average_precision."""
    return mean_average_precision(predictions, ground_truth)


# ============================================================================
# Classification Metrics
# ============================================================================

def auc_score(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
) -> float:
    """
    Compute Area Under ROC Curve (AUC).

    AUC measures the ability of the model to distinguish between
    positive and negative examples.

    Args:
        labels: Binary labels (0 or 1)
        scores: Prediction scores

    Returns:
        AUC score (0-1)

    Example:
        >>> labels = [0, 1, 0, 1]
        >>> scores = [0.1, 0.9, 0.2, 0.8]
        >>> auc_score(labels, scores)
        1.0  # Perfect separation
    """
    labels = np.array(labels)
    scores = np.array(scores)

    # Handle single class case
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    try:
        from sklearn.metrics import roc_auc_score
        result = roc_auc_score(labels, scores)
        # Handle sklearn returning NaN for edge cases
        if np.isnan(result):
            return 0.5
        return result
    except ImportError:
        # Simple implementation
        # Sort by score
        order = np.argsort(scores)[::-1]
        labels = labels[order]

        # Count ranks of positive items
        rank_sum = 0.0
        pos_count = 0
        for i, label in enumerate(labels):
            if label == 1:
                rank_sum += i + 1
                pos_count += 1

        auc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return max(0.0, min(1.0, auc))


def log_loss_score(
    labels: Union[List[int], np.ndarray],
    scores: Union[List[float], np.ndarray],
    epsilon: float = 1e-15,
) -> float:
    """
    Compute Log Loss (Cross-entropy loss).

    Log loss measures the performance of a classification model
    where the prediction is a probability between 0 and 1.

    Args:
        labels: Binary labels (0 or 1)
        scores: Prediction probabilities
        epsilon: Small value to prevent log(0)

    Returns:
        Log loss (lower is better)

    Example:
        >>> labels = [0, 1, 0, 1]
        >>> scores = [0.1, 0.9, 0.2, 0.8]
        >>> log_loss_score(labels, scores)
        # Returns log loss value
    """
    labels = np.array(labels)
    scores = np.clip(np.array(scores), epsilon, 1 - epsilon)

    return -np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))


# ============================================================================
# Diversity, Novelty, Serendipity, Coverage
# ============================================================================

def diversity_score(
    recommendations: List[List[str]],
    item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    attribute_key: str = "category",
) -> float:
    """
    Compute average intra-list diversity (ILD).

    Diversity measures how diverse items are within each recommendation list.
    Higher values indicate more diverse recommendations.

    Args:
        recommendations: List of recommendation lists
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to use for diversity

    Returns:
        Average diversity score (0-1)

    Example:
        >>> recommendations = [['item1', 'item2', 'item3']]
        >>> item_attributes = {'item1': {'category': 'A'},
        ...                    'item2': {'category': 'B'},
        ...                    'item3': {'category': 'A'}}
        >>> diversity_score(recommendations, item_attributes)
        # Returns average pairwise dissimilarity
    """
    if item_attributes is None:
        # Use simple Jaccard distance
        def distance(a: str, b: str) -> float:
            return 1.0 if a != b else 0.0
    else:
        # Use attribute distance
        def distance(a: str, b: str) -> float:
            attr_a = item_attributes.get(a, {}).get(attribute_key, "")
            attr_b = item_attributes.get(b, {}).get(attribute_key, "")
            return 0.0 if attr_a == attr_b else 1.0

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


def diversity_score_entropy(
    recommendations: List[List[str]],
    item_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    attribute_key: str = "category",
) -> float:
    """
    Compute Shannon entropy-based diversity.

    Uses entropy to measure the diversity of attributes in recommendations.
    Higher entropy = more diverse.

    Args:
        recommendations: List of recommendation lists
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to use for diversity

    Returns:
        Average entropy-based diversity score (0-1, normalized)
    """
    if item_attributes is None:
        # Fall back to item-based diversity
        return diversity_score(recommendations, item_attributes, attribute_key)

    total_entropy = 0.0
    total_lists = 0

    for rec_list in recommendations:
        if not rec_list:
            continue

        # Count attribute occurrences
        attr_counts = defaultdict(int)
        for item in rec_list:
            attr_val = item_attributes.get(item, {}).get(attribute_key, "unknown")
            attr_counts[attr_val] += 1

        # Calculate entropy
        total = len(rec_list)
        entropy = 0.0
        for count in attr_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize by log2(number of unique attributes)
        max_entropy = math.log2(len(attr_counts)) if len(attr_counts) > 1 else 1
        if max_entropy > 0:
            total_entropy += entropy / max_entropy
        total_lists += 1

    return total_entropy / total_lists if total_lists > 0 else 0.0


def novelty_score(
    recommendations: List[List[str]],
    item_popularity: Dict[str, float],
    k: int = 10,
) -> float:
    """
    Compute average novelty (inverse popularity).

    Novelty measures how unexpected recommendations are.
    Less popular items = higher novelty.

    Args:
        recommendations: List of recommendation lists
        item_popularity: {item: popularity_score} dictionary (0-1)
        k: Consider top-K items

    Returns:
        Average novelty score (higher = more novel)

    Example:
        >>> recommendations = [['item1', 'item2']]
        >>> item_popularity = {'item1': 0.8, 'item2': 0.1}
        >>> novelty_score(recommendations, item_popularity, k=2)
        # item2 contributes more to novelty
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


def novelty_score_history(
    recommendations: List[List[str]],
    user_history: List[List[str]],
    all_items: Optional[List[str]] = None,
) -> float:
    """
    Compute novelty based on user history.

    Measures how many recommended items are new to each user.

    Args:
        recommendations: List of recommendation lists per user
        user_history: List of interaction history per user
        all_items: Optional list of all catalog items

    Returns:
        Average novelty score (0-1)
    """
    if len(recommendations) != len(user_history):
        raise ValueError("Recommendations and user_history must have same length")

    total_novelty = 0.0
    total_users = 0

    for recs, history in zip(recommendations, user_history):
        history_set = set(history)
        new_items = sum(1 for item in recs if item not in history_set)
        if len(recs) > 0:
            total_novelty += new_items / len(recs)
        total_users += 1

    return total_novelty / total_users if total_users > 0 else 0.0


def serendipity(
    recommendations: List[List[str]],
    user_history: List[List[str]],
    item_similarities: Dict[Tuple[str, str], float],
    ground_truth: Optional[List[List[str]]] = None,
    k: int = 10,
) -> float:
    """
    Compute serendipity (unexpected but relevant recommendations).

    Serendipity = unexpectedness x relevance

    Args:
        recommendations: List of recommendation lists per user
        user_history: User's past interactions per user
        item_similarities: {(item1, item2): similarity} dictionary (0-1)
        ground_truth: Optional relevant items for relevance calculation
        k: Consider top-K items

    Returns:
        Serendipity score (0-1)

    Example:
        >>> recommendations = [['new_item', 'similar_item']]
        >>> user_history = [['old_item']]
        >>> item_similarities = {('new_item', 'old_item'): 0.1,
        ...                      ('similar_item', 'old_item'): 0.9}
        >>> serendipity(recommendations, user_history, item_similarities)
        # new_item has higher serendipity
    """
    if len(recommendations) != len(user_history):
        raise ValueError("Recommendations and user_history must have same length")

    total_serendipity = 0.0
    total_users = 0

    for idx, (recs, history) in enumerate(zip(recommendations, user_history)):
        history_set = set(history)
        ground_set = set(ground_truth[idx]) if ground_truth and idx < len(ground_truth) else set()

        for item in recs[:k]:
            # Unexpectedness: 1 - max similarity to history
            max_sim = 0.0
            for hist_item in history:
                sim = item_similarities.get((item, hist_item),
                      item_similarities.get((hist_item, item), 0.0))
                max_sim = max(max_sim, sim)

            unexpectedness = 1.0 - max_sim

            # Relevance: is item in ground truth?
            relevance = 1.0 if not ground_set or item in ground_set else 0.0

            total_serendipity += unexpectedness * relevance

        total_users += 1

    return total_serendipity / total_users if total_users > 0 else 0.0


def coverage(
    all_recommendations: List[List[str]],
    catalog_size: Optional[int] = None,
    all_items: Optional[List[str]] = None,
) -> float:
    """
    Compute catalog coverage (what % of items are ever recommended).

    Args:
        all_recommendations: List of recommendation lists
        catalog_size: Total number of items in catalog (optional if all_items provided)
        all_items: Optional list of all catalog items

    Returns:
        Coverage ratio (0-1)

    Example:
        >>> all_recommendations = [['a', 'b'], ['b', 'c'], ['a', 'c']]
        >>> coverage(all_recommendations, catalog_size=5)
        0.6  # 3 out of 5 items recommended
    """
    if catalog_size is None and all_items is not None:
        catalog_size = len(all_items)

    if catalog_size is None:
        raise ValueError("Must provide either catalog_size or all_items")

    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)

    return len(recommended_items) / catalog_size if catalog_size > 0 else 0.0


def coverage_by_category(
    all_recommendations: List[List[str]],
    item_attributes: Dict[str, Dict[str, Any]],
    attribute_key: str = "category",
) -> Dict[str, float]:
    """
    Compute coverage within each category.

    Args:
        all_recommendations: List of recommendation lists
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to use

    Returns:
        {category_value: coverage_ratio} dictionary
    """
    # Build catalog items by category
    catalog_by_category = defaultdict(set)
    for item, attrs in item_attributes.items():
        attr_val = attrs.get(attribute_key, "unknown")
        catalog_by_category[attr_val].add(item)

    # Build recommended items by category
    recommended_by_category = defaultdict(set)
    for rec_list in all_recommendations:
        for item in rec_list:
            if item in item_attributes:
                attr_val = item_attributes[item].get(attribute_key, "unknown")
                recommended_by_category[attr_val].add(item)

    # Calculate coverage per category
    result = {}
    for category, catalog_items in catalog_by_category.items():
        recommended_items = recommended_by_category[category]
        total = len(catalog_items)
        result[category] = len(recommended_items) / total if total > 0 else 0.0

    return result


# ============================================================================
# Fairness Metrics
# ============================================================================

def demographic_parity(
    recommendations: Dict[str, List[str]],
    item_attributes: Dict[str, Dict[str, Any]],
    attribute_key: str = "category",
    attribute_values: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute demographic parity across attribute values.

    Demographic parity checks if different groups receive
    similar recommendation rates.

    Args:
        recommendations: {group_id: [recommended_items]}
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to check
        attribute_values: Specific values to check (default: all unique values)

    Returns:
        {attribute_value: avg_recommendation_count}

    Example:
        >>> recommendations = {'user1': ['a', 'b'], 'user2': ['b', 'c']}
        >>> item_attributes = {'a': {'category': 'A'}, 'b': {'category': 'B'},
        ...                     'c': {'category': 'A'}}
        >>> demographic_parity(recommendations, item_attributes)
        {'A': 1.0, 'B': 1.0}  # Average count per category per user
    """
    # Get all attribute values if not specified
    if attribute_values is None:
        attribute_values = set()
        for item, attrs in item_attributes.items():
            if attribute_key in attrs:
                attribute_values.add(attrs[attribute_key])
        attribute_values = sorted(attribute_values)

    # Count recommendations per attribute value
    group_counts = defaultdict(list)

    for _user_id, recs in recommendations.items():
        for attr_val in attribute_values:
            count = sum(1 for item in recs
                       if item in item_attributes and
                       item_attributes[item].get(attribute_key) == attr_val)
            group_counts[attr_val].append(count)

    return {
        attr_val: np.mean(counts) if counts else 0.0
        for attr_val, counts in group_counts.items()
    }


def demographic_parity_diff(
    recommendations: Dict[str, List[str]],
    item_attributes: Dict[str, Dict[str, Any]],
    attribute_key: str = "category",
) -> float:
    """
    Compute demographic parity difference.

    Returns the maximum difference in recommendation rates
    across groups. Lower is better (0 = perfect parity).

    Args:
        recommendations: {group_id: [recommended_items]}
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to check

    Returns:
        Maximum difference across groups (0-1)
    """
    parity = demographic_parity(recommendations, item_attributes, attribute_key)
    values = list(parity.values())

    if len(values) < 2:
        return 0.0

    return max(values) - min(values)


def equalized_odds(
    predictions: List[str],
    ground_truth: List[str],
    group_ids: List[str],
) -> Dict[str, float]:
    """
    Compute equalized odds (TPR parity across groups).

    Equalized odds measures if true positive rates are
    similar across different demographic groups.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        group_ids: Group membership for each predicted item

    Returns:
        {group_id: true_positive_rate}

    Example:
        >>> predictions = ['a', 'b', 'c']
        >>> ground_truth = ['a', 'b']
        >>> group_ids = ['group1', 'group1', 'group2']
        >>> equalized_odds(predictions, ground_truth, group_ids)
        {'group1': 1.0, 'group2': 0.0}
    """
    true_set = set(ground_truth)
    tpr_by_group = defaultdict(lambda: {"hits": 0, "total": 0})

    for item, group_id in zip(predictions, group_ids):
        tpr_by_group[group_id]["total"] += 1
        if item in true_set:
            tpr_by_group[group_id]["hits"] += 1

    return {
        group_id: (stats["hits"] / stats["total"] if stats["total"] > 0 else 0.0)
        for group_id, stats in tpr_by_group.items()
    }


def equalized_odds_diff(
    predictions: List[str],
    ground_truth: List[str],
    group_ids: List[str],
) -> float:
    """
    Compute equalized odds difference.

    Returns the maximum difference in TPR across groups.

    Args:
        predictions: Ordered list of predicted items
        ground_truth: Set/list of relevant items
        group_ids: Group membership for each predicted item

    Returns:
        Maximum TPR difference across groups (0-1)
    """
    odds = equalized_odds(predictions, ground_truth, group_ids)
    values = list(odds.values())

    if len(values) < 2:
        return 0.0

    return max(values) - min(values)


def disparate_impact(
    recommendations: Dict[str, List[str]],
    item_attributes: Dict[str, Dict[str, Any]],
    attribute_key: str = "category",
    privileged_value: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute disparate impact ratio for each group.

    Disparate impact = (rate for unprivileged) / (rate for privileged).
    Values near 1.0 indicate fairness.

    Args:
        recommendations: {group_id: [recommended_items]}
        item_attributes: {item: {attr: value}} dictionary
        attribute_key: Which attribute to check
        privileged_value: The privileged attribute value

    Returns:
        {attribute_value: impact_ratio}

    Example:
        >>> recommendations = {'user1': ['a', 'b'], 'user2': ['a', 'a']}
        >>> item_attributes = {'a': {'category': 'A'}, 'b': {'category': 'B'}}
        >>> disparate_impact(recommendations, item_attributes, privileged_value='A')
        {'A': 1.0, 'B': 0.5}  # B gets half the recommendations of A
    """
    parity = demographic_parity(recommendations, item_attributes, attribute_key)

    if privileged_value is None or privileged_value not in parity:
        # Use max as privileged
        if not parity:
            return {}
        privileged_value = max(parity, key=parity.get)

    privileged_rate = parity[privileged_value]

    if privileged_rate == 0:
        return {k: 0.0 for k in parity}

    return {
        k: v / privileged_rate if privileged_value != k else 1.0
        for k, v in parity.items()
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
    clicks_arr = np.array(clicks)
    if impressions is None:
        impressions_arr = np.ones_like(clicks_arr)
    else:
        impressions_arr = np.array(impressions)

    return {
        "ctr": np.sum(clicks_arr) / np.sum(impressions_arr),
        "avg_ctr": np.mean(clicks_arr / (impressions_arr + 1e-10)),
        "clicks": int(np.sum(clicks_arr)),
        "impressions": int(np.sum(impressions_arr)),
    }


def ctr_auc(
    predictions: List[str],
    click_labels: List[int],
    scores: Optional[List[float]] = None,
) -> float:
    """
    Compute CTR AUC (Click-Through Rate Area Under Curve).

    This metric measures the model's ability to rank items by click probability.

    Args:
        predictions: Predicted item IDs (used for ranking if scores not provided)
        click_labels: Binary click labels (0 or 1)
        scores: Optional prediction scores (if None, uses position as proxy)

    Returns:
        CTR AUC score (0-1)

    Example:
        >>> predictions = ['item1', 'item2', 'item3', 'item4']
        >>> click_labels = [1, 0, 1, 0]
        >>> ctr_auc(predictions, click_labels)
        0.75  # Ability to distinguish clicked items
    """
    if scores is None:
        # Use position-based scores (higher position = higher score)
        scores = [len(predictions) - i for i in range(len(predictions))]

    # Normalize scores
    max_score = max(scores) if scores else 1.0
    scores = [s / max_score for s in scores]

    return auc_score(click_labels, scores)


def visit_auc(
    predictions: List[str],
    visit_labels: List[int],
    scores: Optional[List[float]] = None,
) -> float:
    """
    Compute Visit AUC (Visit/Check-in Rate Area Under Curve).

    This metric measures the model's ability to predict which POIs
    users will actually visit.

    Args:
        predictions: Predicted POI IDs
        visit_labels: Binary visit labels (0 or 1)
        scores: Optional prediction scores

    Returns:
        Visit AUC score (0-1)

    Example:
        >>> predictions = ['poi1', 'poi2', 'poi3', 'poi4']
        >>> visit_labels = [1, 1, 0, 0]
        >>> visit_auc(predictions, visit_labels)
        1.0  # Perfect ranking
    """
    if scores is None:
        # Use position-based scores
        scores = [len(predictions) - i for i in range(len(predictions))]

    max_score = max(scores) if scores else 1.0
    scores = [s / max_score for s in scores]

    return auc_score(visit_labels, scores)


def expected_calibration_error(
    predicted_probs: Union[List[float], np.ndarray],
    true_labels: Union[List[int], np.ndarray],
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well the predicted probabilities are calibrated.
    Lower ECE = better calibration.

    Args:
        predicted_probs: Predicted probabilities (0-1)
        true_labels: True binary labels (0 or 1)
        n_bins: Number of bins for ECE calculation

    Returns:
        Dictionary with ECE and per-bin statistics

    Example:
        >>> probs = [0.1, 0.9, 0.8, 0.2]
        >>> labels = [0, 1, 1, 0]
        >>> ece_result = expected_calibration_error(probs, labels)
        >>> ece_result['ece']
        0.05  # Well calibrated
    """
    probs = np.array(predicted_probs)
    labels = np.array(true_labels)

    # Ensure valid range
    probs = np.clip(probs, 0.0, 1.0)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Assign predictions to bins
    bin_indices = np.digitize(probs, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute calibration metrics per bin
    ece = 0.0
    bin_stats = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.sum(mask) == 0:
            continue

        bin_conf = np.mean(probs[mask])
        bin_acc = np.mean(labels[mask])
        bin_weight = np.sum(mask) / len(probs)

        ece += bin_weight * abs(bin_conf - bin_acc)

        bin_stats.append({
            "bin": bin_idx,
            "lower": float(bin_lowers[bin_idx]),
            "upper": float(bin_uppers[bin_idx]),
            "count": int(np.sum(mask)),
            "confidence": float(bin_conf),
            "accuracy": float(bin_acc),
            "weight": float(bin_weight)
        })

    return {
        "ece": ece,
        "n_bins": n_bins,
        "bin_stats": bin_stats
    }


def brier_score(
    predicted_probs: Union[List[float], np.ndarray],
    true_labels: Union[List[int], np.ndarray],
) -> float:
    """
    Compute Brier Score (mean squared error of probabilities).

    Brier score measures the accuracy of probabilistic predictions.
    Lower is better (0 = perfect, 0.25 = random for binary).

    Args:
        predicted_probs: Predicted probabilities (0-1)
        true_labels: True binary labels (0 or 1)

    Returns:
        Brier score (lower is better)

    Example:
        >>> probs = [0.1, 0.9, 0.8, 0.2]
        >>> labels = [0, 1, 1, 0]
        >>> brier_score(probs, labels)
        0.015  # Very accurate probabilities
    """
    probs = np.array(predicted_probs)
    labels = np.array(true_labels)

    return np.mean((probs - labels) ** 2)


# ============================================================================
# Batch Computation Functions
# ============================================================================

def batch_recall_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> List[float]:
    """
    Compute Recall@K for multiple queries.

    Args:
        predictions: List of prediction lists
        ground_truth: List of ground truth lists
        k: Cutoff position

    Returns:
        List of recall scores
    """
    return [recall_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]


def batch_precision_at_k(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
    k: int,
) -> List[float]:
    """Compute Precision@K for multiple queries."""
    return [precision_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]


def batch_ndcg_at_k(
    predictions: List[List[str]],
    ground_truth: List[Union[List[str], Dict[str, float]]],
    k: int,
) -> List[float]:
    """Compute NDCG@K for multiple queries."""
    return [ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]


def batch_mrr(
    predictions: List[List[str]],
    ground_truth: List[List[str]],
) -> List[float]:
    """Compute MRR for multiple queries."""
    return [mean_reciprocal_rank(pred, truth) for pred, truth in zip(predictions, ground_truth)]


# ============================================================================
# Evaluation Runner
# ============================================================================

class RecommendationEvaluator:
    """
    Comprehensive evaluation runner for recommendation systems.

    This class provides a unified interface for computing all
    recommendation metrics.

    Usage:
        evaluator = RecommendationEvaluator(k_values=[5, 10, 20, 50])
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            item_attributes=item_attrs,
            item_popularity=popularity,
            catalog_size=len(all_items)
        )
        print(evaluator.format_report(results))
    """

    def __init__(
        self,
        k_values: List[int] = None,
        compute_diversity: bool = True,
        compute_novelty: bool = True,
        compute_coverage: bool = True,
    ):
        """
        Initialize the evaluator.

        Args:
            k_values: List of K values for @K metrics
            compute_diversity: Whether to compute diversity metrics
            compute_novelty: Whether to compute novelty metrics
            compute_coverage: Whether to compute coverage metrics
        """
        if k_values is None:
            k_values = [5, 10, 20, 50]
        self.k_values = tuple(k_values)
        self.compute_diversity = compute_diversity
        self.compute_novelty = compute_novelty
        self.compute_coverage = compute_coverage

    def evaluate(
        self,
        predictions: Union[List[List[str]], List[str]],
        ground_truth: Union[List[List[str]], List[str]],
        item_attributes: Optional[Dict[str, Dict]] = None,
        item_popularity: Optional[Dict[str, float]] = None,
        catalog_size: Optional[int] = None,
        user_history: Optional[List[List[str]]] = None,
        item_similarities: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> Dict[str, float]:
        """
        Run comprehensive evaluation.

        Args:
            predictions: List of recommendation lists or single list
            ground_truth: Corresponding relevant items
            item_attributes: For diversity calculation
            item_popularity: For novelty calculation
            catalog_size: For coverage calculation
            user_history: For serendipity calculation
            item_similarities: For serendipity calculation

        Returns:
            Dictionary of all computed metrics
        """
        # Normalize inputs
        single_query = isinstance(predictions, list) and (
            len(predictions) == 0 or isinstance(predictions[0], str)
        )
        if single_query:
            predictions = [predictions]
            ground_truth = [ground_truth]

        results: Dict[str, float] = {}

        # Basic ranking metrics
        for k in self.k_values:
            recalls = [recall_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"recall@{k}"] = float(np.mean(recalls))

            precisions = [precision_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"precision@{k}"] = float(np.mean(precisions))

            f1_scores = [f1_score_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"f1@{k}"] = float(np.mean(f1_scores))

            ndcgs = [ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"ndcg@{k}"] = float(np.mean(ndcgs))

            hit_rates = [hit_rate_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            results[f"hitrate@{k}"] = float(np.mean(hit_rates))

        # MRR and MAP
        mrrs = [mean_reciprocal_rank(pred, truth) for pred, truth in zip(predictions, ground_truth)]
        results["mrr"] = float(np.mean(mrrs))

        maps = [mean_average_precision(pred, truth) for pred, truth in zip(predictions, ground_truth)]
        results["map"] = float(np.mean(maps))

        # Diversity
        if self.compute_diversity and item_attributes is not None:
            results["diversity"] = diversity_score(predictions, item_attributes)
            results["diversity_entropy"] = diversity_score_entropy(predictions, item_attributes)

        # Novelty
        if self.compute_novelty and item_popularity is not None:
            results["novelty"] = novelty_score(predictions, item_popularity, k=10)

        # Novelty based on history
        if self.compute_novelty and user_history is not None:
            results["novelty_history"] = novelty_score_history(predictions, user_history)

        # Serendipity
        if user_history is not None and item_similarities is not None:
            results["serendipity"] = serendipity(
                predictions, user_history, item_similarities,
                ground_truth=ground_truth
            )

        # Coverage
        if self.compute_coverage and catalog_size is not None:
            results["coverage"] = coverage(predictions, catalog_size)

        return results

    def format_report(self, results: Dict[str, float]) -> str:
        """Format evaluation results as a readable report."""
        lines = ["=" * 60, "Recommendation Evaluation Report", "=" * 60]

        # Group by metric type
        ranking_metrics = [k for k in results if "@" in k or k in ["mrr", "map"]]
        quality_metrics = [k for k in results if k in [
            "diversity", "diversity_entropy", "novelty", "novelty_history",
            "coverage", "serendipity"
        ]]
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
# Comparison and Report Generation
# ============================================================================

class MetricsComparison:
    """
    Compare metrics across different models or configurations.

    Usage:
        comparison = MetricsComparison()
        comparison.add_result("baseline", baseline_metrics)
        comparison.add_result("model_a", model_a_metrics)
        comparison.add_result("model_b", model_b_metrics)
        report = comparison.generate_comparison_report()
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, float]] = {}
        self.timestamps: Dict[str, str] = {}

    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        timestamp: Optional[str] = None,
    ):
        """Add metrics result for a model."""
        self.results[model_name] = metrics
        if timestamp:
            self.timestamps[model_name] = timestamp

    def get_best_model(self, metric: str, higher_is_better: bool = True) -> Tuple[str, float]:
        """
        Get the best performing model for a specific metric.

        Args:
            metric: Metric name
            higher_is_better: Whether higher values are better

        Returns:
            (model_name, metric_value) tuple
        """
        if not self.results:
            return ("N/A", 0.0)

        best_model = None
        best_value = None

        for model, metrics in self.results.items():
            if metric not in metrics:
                continue
            value = metrics[metric]
            if best_value is None:
                best_value = value
                best_model = model
            elif higher_is_better and value > best_value:
                best_value = value
                best_model = model
            elif not higher_is_better and value < best_value:
                best_value = value
                best_model = model

        return (best_model or "N/A", best_value or 0.0)

    def get_improvement(
        self,
        metric: str,
        baseline: str,
        candidate: str,
    ) -> Dict[str, float]:
        """
        Calculate relative improvement of candidate over baseline.

        Args:
            metric: Metric name
            baseline: Baseline model name
            candidate: Candidate model name

        Returns:
            Dictionary with absolute and relative changes
        """
        if baseline not in self.results or candidate not in self.results:
            return {"error": "Model not found"}

        if metric not in self.results[baseline] or metric not in self.results[candidate]:
            return {"error": "Metric not found"}

        base_val = self.results[baseline][metric]
        cand_val = self.results[candidate][metric]

        abs_diff = cand_val - base_val
        rel_diff = (abs_diff / base_val * 100) if base_val != 0 else 0.0

        return {
            "baseline": base_val,
            "candidate": cand_val,
            "absolute": abs_diff,
            "relative_pct": rel_diff,
        }

    def generate_comparison_report(
        self,
        focus_metrics: Optional[List[str]] = None,
        baseline: Optional[str] = None,
    ) -> str:
        """
        Generate a comprehensive comparison report.

        Args:
            focus_metrics: Metrics to highlight (default: common ranking metrics)
            baseline: Baseline model name for comparison

        Returns:
            Formatted report string
        """
        lines = ["=" * 80, "GoAfar Metrics Comparison Report", "=" * 80]

        if not self.results:
            lines.append("\nNo results to compare.")
            return "\n".join(lines)

        # Default focus metrics
        if focus_metrics is None:
            focus_metrics = [
                "recall@10", "recall@20",
                "ndcg@10", "ndcg@20",
                "precision@10",
                "mrr", "map",
                "diversity", "novelty", "coverage",
                "ctr_auc", "visit_auc", "ece"
            ]

        # Available metrics across all models
        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(all_metrics)

        # Create comparison table
        lines.append("\n[" + "=" * 76 + "]")
        lines.append("| Metric " + " | " + " | ".join(f"{m:<15}" for m in self.results.keys()) + " |")
        lines.append("|" + "-" * 78 + "|")

        for metric in all_metrics:
            # Format metric name
            metric_str = f"{metric:<18}"
            line = f"| {metric_str} |"

            for model in self.results:
                if metric in self.results[model]:
                    val = self.results[model][metric]
                    # Format based on metric type
                    if metric in ["ece", "brier"]:
                        # Lower is better
                        line += f" {val:.4f}       |".rjust(18)
                    elif "@" in metric or metric in ["mrr", "map", "diversity", "novelty", "coverage"]:
                        # Higher is better, percentage
                        line += f" {val:.4f}      |".rjust(17)
                    else:
                        line += f" {val:.4f}    |".rjust(15)
                else:
                    line += " " + "N/A".center(13) + " |"

            lines.append(line)

        lines.append("|" + "=" * 78 + "|")

        # Best model per metric
        lines.append("\n[Best Performing Models]")
        for metric in focus_metrics:
            higher_better = metric not in ["ece", "brier"]
            best_model, best_val = self.get_best_model(metric, higher_better)
            lines.append(f"  {metric}: {best_model} ({best_val:.4f})")

        # Comparison with baseline
        if baseline and baseline in self.results:
            lines.append(f"\n[Improvement vs Baseline: {baseline}]")
            for candidate in self.results:
                if candidate == baseline:
                    continue
                lines.append(f"\n  {candidate}:")
                for metric in focus_metrics:
                    if metric in self.results[baseline] and metric in self.results[candidate]:
                        imp = self.get_improvement(metric, baseline, candidate)
                        higher_better = metric not in ["ece", "brier"]
                        direction = "+" if imp["relative_pct"] > 0 else ""
                        status = ""
                        if higher_better:
                            status = "BETTER" if imp["relative_pct"] > 0 else "WORSE" if imp["relative_pct"] < 0 else "SAME"
                        else:
                            status = "BETTER" if imp["relative_pct"] < 0 else "WORSE" if imp["relative_pct"] > 0 else "SAME"
                        lines.append(f"    {metric}: {direction}{imp['relative_pct']:.2f}% [{status}]")

        # Summary statistics
        lines.append("\n[Summary]")
        lines.append(f"  Models compared: {len(self.results)}")
        lines.append(f"  Total metrics: {len(all_metrics)}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


class BusinessMetricsEvaluator:
    """
    Evaluator for business-specific metrics including CTR, Visit, and Calibration.

    Usage:
        evaluator = BusinessMetricsEvaluator()
        results = evaluator.evaluate(
            click_labels=clicks,
            visit_labels=visits,
            predicted_probs=probs
        )
    """

    def __init__(self):
        self.results: Dict[str, float] = {}

    def evaluate(
        self,
        click_labels: Optional[List[int]] = None,
        visit_labels: Optional[List[int]] = None,
        predicted_probs: Optional[List[float]] = None,
        predictions: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate business metrics.

        Args:
            click_labels: Binary click indicators (0 or 1)
            visit_labels: Binary visit indicators (0 or 1)
            predicted_probs: Predicted probabilities for calibration
            predictions: Predicted item IDs
            scores: Prediction scores for AUC calculation

        Returns:
            Dictionary of business metrics
        """
        results = {}

        # CTR AUC
        if click_labels is not None:
            if scores is not None and predictions is not None:
                results["ctr_auc"] = ctr_auc(predictions, click_labels, scores)
            elif predictions is not None:
                results["ctr_auc"] = ctr_auc(predictions, click_labels)

        # Visit AUC
        if visit_labels is not None:
            if scores is not None and predictions is not None:
                results["visit_auc"] = visit_auc(predictions, visit_labels, scores)
            elif predictions is not None:
                results["visit_auc"] = visit_auc(predictions, visit_labels)

        # Calibration (ECE)
        if predicted_probs is not None and click_labels is not None:
            ece_result = expected_calibration_error(predicted_probs, click_labels)
            results["ece"] = ece_result["ece"]
            results["brier"] = brier_score(predicted_probs, click_labels)

        return results

    def format_report(self, results: Dict[str, float]) -> str:
        """Format business metrics as a report."""
        lines = ["=" * 60, "Business Metrics Report", "=" * 60]

        # CTR metrics
        ctr_metrics = [k for k in results if "ctr" in k.lower()]
        if ctr_metrics:
            lines.append("\n[Click-Through Rate]")
            for k in sorted(ctr_metrics):
                lines.append(f"  {k}: {results[k]:.4f}")

        # Visit metrics
        visit_metrics = [k for k in results if "visit" in k.lower()]
        if visit_metrics:
            lines.append("\n[Visit Prediction]")
            for k in sorted(visit_metrics):
                lines.append(f"  {k}: {results[k]:.4f}")

        # Calibration metrics
        calib_metrics = [k for k in results if k in ["ece", "brier"]]
        if calib_metrics:
            lines.append("\n[Calibration]")
            for k in sorted(calib_metrics):
                lines.append(f"  {k}: {results[k]:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# Legacy API Compatibility
# ============================================================================

def evaluate_recall(
    predictions: List[str],
    ground_truth: List[str],
    k_list: List[int] = None,
) -> Dict[str, float]:
    """Legacy wrapper for recall evaluation."""
    if k_list is None:
        k_list = [10, 20, 50]
    results = {}
    for k in k_list:
        results[f"Recall@{k}"] = recall_at_k(predictions, ground_truth, k)
    return results


def evaluate_ndcg(
    predictions_with_scores: List[Tuple[str, float]],
    ground_truth_with_scores: List[Tuple[str, float]],
    k_list: List[int] = None,
) -> Dict[str, float]:
    """Legacy wrapper for NDCG evaluation."""
    if k_list is None:
        k_list = [10, 20]
    results = {}
    predictions = [item for item, _ in predictions_with_scores]
    relevance = dict(ground_truth_with_scores)

    for k in k_list:
        results[f"NDCG@{k}"] = ndcg_at_k(predictions, relevance, k)
    return results


def evaluate_route_quality(
    route_solution: Optional[Dict[str, Any]],
    max_duration: float,
) -> Dict[str, Any]:
    """Legacy wrapper for route quality evaluation."""
    if route_solution is None:
        return {"feasible": 0.0, "duration_utilization": 0.0, "num_visited": 0}

    return {
        "feasible": 1.0,
        "duration_utilization": route_solution.get("total_hours", 0) / max_duration,
        "num_visited": route_solution.get("visited_pois", 0),
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(description="Advanced recommendation metrics")
    parser.add_argument("--predictions", required=True, help="JSON file with predictions")
    parser.add_argument("--ground-truth", required=True, help="JSON file with ground truth")
    parser.add_argument("--output", help="Output JSON for results")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10, 20, 50],
                       help="K values for @K metrics")
    args = parser.parse_args()

    # Load data
    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    # Evaluate
    evaluator = RecommendationEvaluator(k_values=args.k_values)
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
