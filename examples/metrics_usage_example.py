#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage examples for advanced recommendation metrics.

Demonstrates how to use the metrics for evaluating recommendation systems.

Usage:
    python examples/metrics_usage_example.py
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from evaluation.metrics_advanced import (
    # Basic ranking metrics
    recall_at_k,
    precision_at_k,
    f1_score_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate_at_k,
    mean_average_precision,
    # Classification metrics
    auc_score,
    log_loss_score,
    # Diversity/novelty/coverage
    diversity_score,
    diversity_score_entropy,
    novelty_score,
    serendipity,
    coverage,
    # Fairness metrics
    demographic_parity,
    demographic_parity_diff,
    equalized_odds,
    disparate_impact,
    # Batch functions
    batch_recall_at_k,
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
)


def example_single_query_metrics():
    """Example: Compute metrics for a single query."""
    print("=" * 60)
    print("Example 1: Single Query Metrics")
    print("=" * 60)

    predictions = ["poi1", "poi2", "poi3", "poi4", "poi5"]
    ground_truth = ["poi2", "poi4", "poi6"]

    # Compute various metrics
    recall = recall_at_k(predictions, ground_truth, k=5)
    precision = precision_at_k(predictions, ground_truth, k=5)
    f1 = f1_score_at_k(predictions, ground_truth, k=5)
    ndcg = ndcg_at_k(predictions, ground_truth, k=5)
    mrr = mean_reciprocal_rank(predictions, ground_truth)
    hit_rate = hit_rate_at_k(predictions, ground_truth, k=5)
    map_score = mean_average_precision(predictions, ground_truth)

    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    print(f"\nRecall@5: {recall:.4f}")
    print(f"Precision@5: {precision:.4f}")
    print(f"F1@5: {f1:.4f}")
    print(f"NDCG@5: {ndcg:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"HitRate@5: {hit_rate:.4f}")
    print(f"MAP: {map_score:.4f}")


def example_classification_metrics():
    """Example: Compute classification metrics."""
    print("\n" + "=" * 60)
    print("Example 2: Classification Metrics")
    print("=" * 60)

    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]

    auc = auc_score(labels, scores)
    log_loss = log_loss_score(labels, scores)

    print(f"Labels: {labels}")
    print(f"Scores: {scores}")
    print(f"\nAUC: {auc:.4f}")
    print(f"Log Loss: {log_loss:.4f}")


def example_diversity_metrics():
    """Example: Compute diversity and novelty metrics."""
    print("\n" + "=" * 60)
    print("Example 3: Diversity and Novelty Metrics")
    print("=" * 60)

    # Sample recommendations with categories
    recommendations = [
        ["poi1", "poi2", "poi3"],
        ["poi4", "poi5", "poi6"]
    ]

    item_attributes = {
        "poi1": {"category": "nature"},
        "poi2": {"category": "culture"},
        "poi3": {"category": "nature"},
        "poi4": {"category": "culture"},
        "poi5": {"category": "food"},
        "poi6": {"category": "culture"}
    }

    # Compute diversity
    diversity = diversity_score(recommendations, item_attributes, "category")
    diversity_ent = diversity_score_entropy(recommendations, item_attributes, "category")

    # Compute novelty (lower popularity = higher novelty)
    item_popularity = {
        "poi1": 0.8,  # popular
        "poi2": 0.6,
        "poi3": 0.7,
        "poi4": 0.3,  # less popular
        "poi5": 0.2,
        "poi6": 0.4
    }
    novelty = novelty_score(recommendations, item_popularity, k=3)

    # Compute coverage
    all_items = ["poi1", "poi2", "poi3", "poi4", "poi5", "poi6", "poi7", "poi8"]
    cov = coverage(recommendations, catalog_size=len(all_items))

    print(f"Recommendations: {recommendations}")
    print(f"Diversity (ILD): {diversity:.4f}")
    print(f"Diversity (Entropy): {diversity_ent:.4f}")
    print(f"Novelty: {novelty:.4f}")
    print(f"Coverage: {cov:.4f}")


def example_serendipity():
    """Example: Compute serendipity metric."""
    print("\n" + "=" * 60)
    print("Example 4: Serendipity Metric")
    print("=" * 60)

    recommendations = [["new_poi", "similar_poi"]]
    user_history = [["old_poi"]]

    # Item similarities (lower = more different)
    item_similarities = {
        ("new_poi", "old_poi"): 0.1,  # very different
        ("similar_poi", "old_poi"): 0.9  # very similar
    }

    # Ground truth items (relevant)
    ground_truth = [["new_poi"]]

    serend = serendipity(
        recommendations,
        user_history,
        item_similarities,
        ground_truth=ground_truth,
        k=2
    )

    print(f"Recommendations: {recommendations}")
    print(f"User History: {user_history}")
    print(f"Serendipity: {serend:.4f}")


def example_fairness_metrics():
    """Example: Compute fairness metrics."""
    print("\n" + "=" * 60)
    print("Example 5: Fairness Metrics")
    print("=" * 60)

    # User recommendations
    recommendations = {
        "user1": ["poi_a", "poi_b"],
        "user2": ["poi_a", "poi_c"],
        "user3": ["poi_b", "poi_c"]
    }

    # Item attributes (e.g., region)
    item_attributes = {
        "poi_a": {"region": "A"},
        "poi_b": {"region": "B"},
        "poi_c": {"region": "A"}
    }

    # Compute demographic parity
    parity = demographic_parity(recommendations, item_attributes, "region")
    parity_diff = demographic_parity_diff(recommendations, item_attributes, "region")

    # Compute disparate impact
    impact = disparate_impact(recommendations, item_attributes, "region", "A")

    print(f"Recommendations: {recommendations}")
    print(f"Item Attributes: {item_attributes}")
    print(f"\nDemographic Parity: {parity}")
    print(f"Parity Difference: {parity_diff:.4f}")
    print(f"Disparate Impact: {impact}")


def example_batch_evaluation():
    """Example: Batch evaluation with RecommendationEvaluator."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Evaluation with RecommendationEvaluator")
    print("=" * 60)

    # Sample data: 3 users
    predictions = [
        ["poi1", "poi2", "poi3", "poi4", "poi5"],
        ["poi2", "poi6", "poi7", "poi8", "poi9"],
        ["poi1", "poi3", "poi5", "poi7", "poi9"]
    ]

    ground_truth = [
        ["poi2", "poi4", "poi6"],
        ["poi6", "poi8"],
        ["poi1", "poi5"]
    ]

    # Item attributes for diversity
    item_attributes = {
        "poi1": {"category": "nature"},
        "poi2": {"category": "culture"},
        "poi3": {"category": "nature"},
        "poi4": {"category": "culture"},
        "poi5": {"category": "nature"},
        "poi6": {"category": "culture"},
        "poi7": {"category": "nature"},
        "poi8": {"category": "culture"},
        "poi9": {"category": "nature"}
    }

    # Item popularity for novelty
    item_popularity = {
        f"poi{i}": 0.1 * i for i in range(1, 10)
    }

    # Create evaluator and evaluate
    evaluator = RecommendationEvaluator(k_values=[5, 10])
    results = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        item_attributes=item_attributes,
        item_popularity=item_popularity,
        catalog_size=9
    )

    # Print report
    print(evaluator.format_report(results))


def example_equalized_odds():
    """Example: Compute equalized odds metric."""
    print("\n" + "=" * 60)
    print("Example 7: Equalized Odds Metric")
    print("=" * 60)

    predictions = ["item_a", "item_b", "item_c", "item_d"]
    ground_truth = ["item_a", "item_b", "item_e"]
    group_ids = ["group1", "group1", "group2", "group2"]

    odds = equalized_odds(predictions, ground_truth, group_ids)

    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Group IDs: {group_ids}")
    print(f"\nEqualized Odds (TPR by group): {odds}")


def example_business_metrics():
    """Example: Compute business-specific metrics."""
    print("\n" + "=" * 60)
    print("Example 8: Business Metrics (CTR AUC, Visit AUC, ECE)")
    print("=" * 60)

    # Sample predictions and labels
    predictions = ["poi1", "poi2", "poi3", "poi4", "poi5", "poi6", "poi7", "poi8"]
    click_labels = [1, 0, 1, 1, 0, 1, 0, 0]
    visit_labels = [1, 0, 1, 0, 0, 1, 0, 0]

    # CTR AUC
    ctr = ctr_auc(predictions, click_labels)
    print(f"CTR AUC: {ctr:.4f}")

    # Visit AUC
    visit = visit_auc(predictions, visit_labels)
    print(f"Visit AUC: {visit:.4f}")

    # Expected Calibration Error
    predicted_probs = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.1, 0.4]
    ece_result = expected_calibration_error(predicted_probs, click_labels, n_bins=5)
    print(f"ECE: {ece_result['ece']:.4f}")

    # Brier Score
    brier = brier_score(predicted_probs, click_labels)
    print(f"Brier Score: {brier:.4f}")

    # CTR Metrics
    ctr_result = ctr_metrics(predictions, click_labels)
    print(f"\nCTR Metrics: {ctr_result}")


def example_metrics_comparison():
    """Example: Compare metrics across multiple models."""
    print("\n" + "=" * 60)
    print("Example 9: Metrics Comparison Across Models")
    print("=" * 60)

    # Create comparison
    comparison = MetricsComparison()

    # Add results from different models
    comparison.add_result("baseline", {
        "recall@10": 0.45,
        "ndcg@10": 0.52,
        "precision@10": 0.35,
        "mrr": 0.48,
        "diversity": 0.60,
        "novelty": 2.1,
        "ctr_auc": 0.72,
        "visit_auc": 0.68,
        "ece": 0.15,
    })

    comparison.add_result("model_a", {
        "recall@10": 0.52,
        "ndcg@10": 0.58,
        "precision@10": 0.38,
        "mrr": 0.55,
        "diversity": 0.65,
        "novelty": 2.3,
        "ctr_auc": 0.78,
        "visit_auc": 0.72,
        "ece": 0.12,
    })

    comparison.add_result("model_b", {
        "recall@10": 0.48,
        "ndcg@10": 0.55,
        "precision@10": 0.36,
        "mrr": 0.51,
        "diversity": 0.70,
        "novelty": 2.5,
        "ctr_auc": 0.75,
        "visit_auc": 0.70,
        "ece": 0.10,
    })

    # Generate comparison report
    report = comparison.generate_comparison_report(baseline="baseline")
    print(report)


def example_business_evaluator():
    """Example: Use BusinessMetricsEvaluator."""
    print("\n" + "=" * 60)
    print("Example 10: BusinessMetricsEvaluator")
    print("=" * 60)

    evaluator = BusinessMetricsEvaluator()

    # Sample data
    click_labels = [1, 0, 1, 1, 0, 1, 0, 0]
    visit_labels = [1, 0, 1, 0, 0, 1, 0, 0]
    predicted_probs = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.1, 0.4]
    predictions = ["poi1", "poi2", "poi3", "poi4", "poi5", "poi6", "poi7", "poi8"]
    scores = [0.9, 0.3, 0.8, 0.7, 0.2, 0.85, 0.15, 0.4]

    results = evaluator.evaluate(
        click_labels=click_labels,
        visit_labels=visit_labels,
        predicted_probs=predicted_probs,
        predictions=predictions,
        scores=scores,
    )

    print(evaluator.format_report(results))


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Advanced Recommendation Metrics - Usage Examples")
    print("=" * 60)

    example_single_query_metrics()
    example_classification_metrics()
    example_diversity_metrics()
    example_serendipity()
    example_fairness_metrics()
    example_equalized_odds()
    example_batch_evaluation()
    example_business_metrics()
    example_metrics_comparison()
    example_business_evaluator()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
