#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for advanced metrics in GoAfar.

This script tests the new business metrics:
- CTR AUC
- Visit AUC
- Expected Calibration Error (ECE)
- Brier Score
- Metrics Comparison

Usage:
    python test_metrics_advanced.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from evaluation.metrics_advanced import (
    # Ranking metrics
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    mean_average_precision,
    # Business metrics
    ctr_auc,
    visit_auc,
    expected_calibration_error,
    brier_score,
    # Comparison and reporting
    MetricsComparison,
    BusinessMetricsEvaluator,
    RecommendationEvaluator,
)


def test_business_metrics():
    """Test business metrics: CTR AUC, Visit AUC, ECE, Brier Score."""
    print("=" * 60)
    print("Test 1: Business Metrics")
    print("=" * 60)

    # Sample data
    predictions = ["poi1", "poi2", "poi3", "poi4", "poi5", "poi6", "poi7", "poi8"]
    click_labels = [1, 0, 1, 1, 0, 1, 0, 0]
    visit_labels = [1, 0, 1, 0, 0, 1, 0, 0]
    predicted_probs = [0.1, 0.3, 0.7, 0.8, 0.2, 0.9, 0.1, 0.4]

    # Test CTR AUC
    ctr = ctr_auc(predictions, click_labels)
    assert 0 <= ctr <= 1, "CTR AUC should be between 0 and 1"
    print(f"CTR AUC: {ctr:.4f} - PASS")

    # Test Visit AUC
    visit = visit_auc(predictions, visit_labels)
    assert 0 <= visit <= 1, "Visit AUC should be between 0 and 1"
    print(f"Visit AUC: {visit:.4f} - PASS")

    # Test ECE
    ece_result = expected_calibration_error(predicted_probs, click_labels)
    assert 0 <= ece_result["ece"] <= 1, "ECE should be between 0 and 1"
    print(f"ECE: {ece_result['ece']:.4f} - PASS")

    # Test Brier Score
    brier = brier_score(predicted_probs, click_labels)
    assert 0 <= brier <= 1, "Brier Score should be between 0 and 1"
    print(f"Brier Score: {brier:.4f} - PASS")

    print()


def test_metrics_comparison():
    """Test MetricsComparison class."""
    print("=" * 60)
    print("Test 2: MetricsComparison")
    print("=" * 60)

    comparison = MetricsComparison()

    # Add baseline
    comparison.add_result("baseline", {
        "recall@10": 0.45,
        "ndcg@10": 0.52,
        "precision@10": 0.35,
        "mrr": 0.48,
        "diversity": 0.60,
        "ctr_auc": 0.72,
        "ece": 0.15,
    })

    # Add model A
    comparison.add_result("model_a", {
        "recall@10": 0.52,
        "ndcg@10": 0.58,
        "precision@10": 0.38,
        "mrr": 0.55,
        "diversity": 0.65,
        "ctr_auc": 0.78,
        "ece": 0.12,
    })

    # Add model B
    comparison.add_result("model_b", {
        "recall@10": 0.48,
        "ndcg@10": 0.55,
        "precision@10": 0.36,
        "mrr": 0.51,
        "diversity": 0.70,
        "ctr_auc": 0.75,
        "ece": 0.10,
    })

    # Test get_best_model
    best_recall, val = comparison.get_best_model("recall@10")
    assert best_recall == "model_a", "model_a should have best recall@10"
    print(f"Best Recall@10: {best_recall} ({val:.4f}) - PASS")

    best_ece, val = comparison.get_best_model("ece", higher_is_better=False)
    assert best_ece == "model_b", "model_b should have best (lowest) ECE"
    print(f"Best ECE: {best_ece} ({val:.4f}) - PASS")

    # Test get_improvement
    imp = comparison.get_improvement("recall@10", "baseline", "model_a")
    assert imp["relative_pct"] > 0, "model_a should improve recall"
    print(f"Recall improvement: {imp['relative_pct']:.2f}% - PASS")

    # Generate report
    report = comparison.generate_comparison_report(baseline="baseline")
    print("\n" + report)

    print()


def test_business_evaluator():
    """Test BusinessMetricsEvaluator class."""
    print("=" * 60)
    print("Test 3: BusinessMetricsEvaluator")
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

    assert "ctr_auc" in results, "CTR AUC should be in results"
    assert "visit_auc" in results, "Visit AUC should be in results"
    assert "ece" in results, "ECE should be in results"
    assert "brier" in results, "Brier score should be in results"

    print("BusinessMetricsEvaluator results:")
    for key, val in results.items():
        print(f"  {key}: {val:.4f}")

    print("\n" + evaluator.format_report(results))
    print()


def test_recommendation_evaluator_with_business():
    """Test RecommendationEvaluator with business metrics."""
    print("=" * 60)
    print("Test 4: RecommendationEvaluator with Business Metrics")
    print("=" * 60)

    # Sample data
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

    evaluator = RecommendationEvaluator(k_values=[5, 10])
    results = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        item_attributes=item_attributes,
        item_popularity=item_popularity,
        catalog_size=9
    )

    print("RecommendationEvaluator results:")
    for key, val in sorted(results.items()):
        print(f"  {key}: {val:.4f}")

    print("\n" + evaluator.format_report(results))
    print()


def test_recall_precision_f1():
    """Test Recall, Precision, and F1@K metrics."""
    print("=" * 60)
    print("Test 5: Recall, Precision, F1@K")
    print("=" * 60)

    predictions = ["poi1", "poi2", "poi3", "poi4", "poi5"]
    ground_truth = ["poi2", "poi4", "poi6"]

    k = 5
    recall = recall_at_k(predictions, ground_truth, k)
    precision = precision_at_k(predictions, ground_truth, k)

    # Expected: 2/3 recall, 2/5 precision
    assert abs(recall - 2/3) < 0.01, f"Recall should be ~0.667, got {recall}"
    assert abs(precision - 2/5) < 0.01, f"Precision should be ~0.4, got {precision}"

    print(f"Recall@{k}: {recall:.4f} - PASS")
    print(f"Precision@{k}: {precision:.4f} - PASS")

    # Test F1
    from evaluation.metrics_advanced import f1_score_at_k
    f1 = f1_score_at_k(predictions, ground_truth, k)
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert abs(f1 - expected_f1) < 0.01, f"F1 should be ~{expected_f1:.4f}, got {f1}"
    print(f"F1@{k}: {f1:.4f} - PASS")

    print()


def test_mrr_map():
    """Test MRR and MAP metrics."""
    print("=" * 60)
    print("Test 6: MRR and MAP")
    print("=" * 60)

    # MRR test
    predictions = ["a", "b", "c", "d", "e"]
    ground_truth = ["b", "d", "f"]

    # First relevant item (b) is at position 2, so MRR = 1/2
    mrr = mean_reciprocal_rank(predictions, ground_truth)
    assert abs(mrr - 0.5) < 0.01, f"MRR should be ~0.5, got {mrr}"
    print(f"MRR: {mrr:.4f} - PASS")

    # MAP test
    predictions = ["a", "b", "c", "d", "e"]
    ground_truth = ["b", "d", "f"]

    # AP = (1/2 + 2/4) / 3 = (0.5 + 0.5) / 3 = 1/3
    ap = mean_average_precision(predictions, ground_truth)
    assert abs(ap - 1/3) < 0.01, f"AP should be ~0.333, got {ap}"
    print(f"MAP: {ap:.4f} - PASS")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Advanced Metrics for GoAfar")
    print("=" * 60)

    try:
        test_business_metrics()
        test_metrics_comparison()
        test_business_evaluator()
        test_recommendation_evaluator_with_business()
        test_recall_precision_f1()
        test_mrr_map()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
