#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for advanced evaluation metrics.

Tests all ranking metrics, diversity/novelty metrics, and fairness metrics.

Usage:
    pytest tests/test_metrics_advanced.py -v
    python -m pytest tests/test_metrics_advanced.py::TestRecallMetrics -v
"""
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import math
import numpy as np
import pytest

from evaluation.metrics_advanced import (
    # Basic ranking metrics
    recall_at_k,
    precision_at_k,
    f1_score_at_k,
    ndcg_at_k,
    mean_reciprocal_rank,
    mrr,
    hit_rate_at_k,
    mean_average_precision,
    map_score,
    # Classification metrics
    auc_score,
    log_loss_score,
    # Diversity/novelty/coverage
    diversity_score,
    diversity_score_entropy,
    novelty_score,
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
    # Legacy
    evaluate_recall,
    evaluate_ndcg,
)


# ============================================================================
# Test Basic Ranking Metrics
# ============================================================================

class TestRecallMetrics:
    """Test recall@K metric."""

    def test_recall_at_k_perfect(self):
        """Test perfect recall (all relevant items found)."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item1", "item2", "item3"]
        assert recall_at_k(predictions, ground_truth, k=5) == 1.0

    def test_recall_at_k_partial(self):
        """Test partial recall."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item2", "item4", "item6"]
        # 2 out of 3 relevant items found
        assert abs(recall_at_k(predictions, ground_truth, k=5) - 2/3) < 1e-6

    def test_recall_at_k_empty_ground_truth(self):
        """Test recall with empty ground truth."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = []
        assert recall_at_k(predictions, ground_truth, k=5) == 0.0

    def test_recall_at_k_cutoff(self):
        """Test recall with k smaller than prediction list."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item4", "item5"]
        # With k=3, no relevant items found
        assert recall_at_k(predictions, ground_truth, k=3) == 0.0


class TestPrecisionMetrics:
    """Test precision@K metric."""

    def test_precision_at_k_perfect(self):
        """Test perfect precision."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item1", "item2", "item3", "item4", "item5"]
        assert precision_at_k(predictions, ground_truth, k=5) == 1.0

    def test_precision_at_k_partial(self):
        """Test partial precision."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item2", "item4"]
        # 2 out of 5 items are relevant
        assert precision_at_k(predictions, ground_truth, k=5) == 0.4

    def test_precision_at_k_zero(self):
        """Test precision with k=0."""
        predictions = ["item1", "item2"]
        ground_truth = ["item1"]
        assert precision_at_k(predictions, ground_truth, k=0) == 0.0


class TestF1Metrics:
    """Test F1 score@K metric."""

    def test_f1_score_at_k(self):
        """Test F1 score calculation."""
        predictions = ["item1", "item2", "item3", "item4", "item5"]
        ground_truth = ["item2", "item4"]
        precision = 0.4  # 2/5
        recall = 1.0     # 2/2
        expected_f1 = 2 * precision * recall / (precision + recall)
        assert abs(f1_score_at_k(predictions, ground_truth, k=5) - expected_f1) < 1e-6

    def test_f1_score_zero(self):
        """Test F1 score when both precision and recall are zero."""
        predictions = ["item1", "item2"]
        ground_truth = ["item3"]
        assert f1_score_at_k(predictions, ground_truth, k=2) == 0.0


class TestNDCGMetrics:
    """Test NDCG@K metric."""

    def test_ndcg_at_k_perfect(self):
        """Test perfect NDCG (all relevant items at top)."""
        predictions = ["item1", "item2", "item3", "item4"]
        ground_truth = ["item1", "item2", "item3"]
        # All relevant items at top positions
        result = ndcg_at_k(predictions, ground_truth, k=4)
        assert abs(result - 1.0) < 1e-6

    def test_ndcg_at_k_partial(self):
        """Test NDCG with some relevant items."""
        predictions = ["item1", "item2", "item3", "item4"]
        ground_truth = ["item2", "item4"]
        result = ndcg_at_k(predictions, ground_truth, k=4)
        # Should be less than 1.0 since item2 is at position 2, item4 at position 4
        assert 0.0 < result < 1.0

    def test_ndcg_at_k_with_relevance(self):
        """Test NDCG with graded relevance."""
        predictions = ["item1", "item2", "item3", "item4"]
        ground_truth = {"item1": 3.0, "item2": 2.0, "item3": 1.0}
        result = ndcg_at_k(predictions, ground_truth, k=4)
        assert abs(result - 1.0) < 1e-6

    def test_ndcg_at_k_empty_ground_truth(self):
        """Test NDCG with empty ground truth."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = []
        assert ndcg_at_k(predictions, ground_truth, k=3) == 0.0


class TestMRRMetrics:
    """Test Mean Reciprocal Rank metric."""

    def test_mrr_first_relevant(self):
        """Test MRR when first item is relevant."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item1", "item5"]
        assert mean_reciprocal_rank(predictions, ground_truth) == 1.0

    def test_mrr_second_relevant(self):
        """Test MRR when first relevant is at position 2."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item2", "item5"]
        assert mean_reciprocal_rank(predictions, ground_truth) == 0.5

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant items found."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item4", "item5"]
        assert mean_reciprocal_rank(predictions, ground_truth) == 0.0

    def test_mrr_alias(self):
        """Test mrr alias function."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item2"]
        assert mrr(predictions, ground_truth) == 0.5


class TestHitRateMetrics:
    """Test HitRate@K metric."""

    def test_hit_rate_at_k_hit(self):
        """Test hit rate when relevant item found."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item2"]
        assert hit_rate_at_k(predictions, ground_truth, k=5) == 1.0

    def test_hit_rate_at_k_miss(self):
        """Test hit rate when no relevant items found."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item4"]
        assert hit_rate_at_k(predictions, ground_truth, k=3) == 0.0


class TestMAPMetrics:
    """Test Mean Average Precision metric."""

    def test_map_perfect(self):
        """Test perfect MAP."""
        predictions = ["item1", "item2", "item3"]
        ground_truth = ["item1", "item2", "item3"]
        # AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        assert abs(mean_average_precision(predictions, ground_truth) - 1.0) < 1e-6

    def test_map_partial(self):
        """Test MAP with partial matches."""
        predictions = ["a", "b", "c", "d", "e"]
        ground_truth = ["b", "d", "f"]
        # Only b and d are in predictions, f is not found
        # AP = (1/2 + 2/4) / 3 (all relevant items)
        # AP = (0.5 + 0.5) / 3 = 0.333...
        assert abs(mean_average_precision(predictions, ground_truth) - 1/3) < 1e-6

    def test_map_empty_ground_truth(self):
        """Test MAP with empty ground truth."""
        predictions = ["item1", "item2"]
        ground_truth = []
        assert mean_average_precision(predictions, ground_truth) == 0.0

    def test_map_alias(self):
        """Test map_score alias function."""
        predictions = ["item1", "item2"]
        ground_truth = ["item1"]
        assert abs(map_score(predictions, ground_truth) - 1.0) < 1e-6


# ============================================================================
# Test Classification Metrics
# ============================================================================

class TestAUCMetrics:
    """Test AUC metric."""

    def test_auc_perfect(self):
        """Test perfect AUC (perfect separation)."""
        labels = [0, 1, 0, 1]
        scores = [0.1, 0.9, 0.2, 0.8]
        result = auc_score(labels, scores)
        assert abs(result - 1.0) < 0.01

    def test_auc_random(self):
        """Test AUC for random predictions."""
        labels = [0, 0, 1, 1]
        scores = [0.5, 0.5, 0.5, 0.5]
        # Should be around 0.5 for random
        result = auc_score(labels, scores)
        assert 0.4 <= result <= 0.6

    def test_auc_single_class(self):
        """Test AUC when only one class present."""
        labels = [0, 0, 0]
        scores = [0.1, 0.2, 0.3]
        # sklearn returns NaN for single class, our implementation returns 0.5
        result = auc_score(labels, scores)
        assert result == 0.5 or (isinstance(result, float) and np.isnan(result))


class TestLogLossMetrics:
    """Test Log Loss metric."""

    def test_log_loss_perfect(self):
        """Test log loss for perfect predictions."""
        labels = [0, 1, 0, 1]
        scores = [0.01, 0.99, 0.01, 0.99]
        # Should be very low
        result = log_loss_score(labels, scores)
        assert result < 0.1

    def test_log_loss_imperfect(self):
        """Test log loss for imperfect predictions."""
        labels = [0, 1, 0, 1]
        scores = [0.9, 0.1, 0.8, 0.2]
        # Should be higher than perfect
        result = log_loss_score(labels, scores)
        assert result > 1.0


# ============================================================================
# Test Diversity/Novelty/Coverage Metrics
# ============================================================================

class TestDiversityMetrics:
    """Test diversity metrics."""

    def test_diversity_score_no_attributes(self):
        """Test diversity without attributes."""
        recommendations = [["item1", "item2", "item3"], ["a", "b", "c"]]
        result = diversity_score(recommendations)
        # All items are different, so diversity should be 1.0
        assert result == 1.0

    def test_diversity_score_with_attributes(self):
        """Test diversity with category attributes."""
        recommendations = [["item1", "item2", "item3"]]
        item_attributes = {
            "item1": {"category": "A"},
            "item2": {"category": "B"},
            "item3": {"category": "A"}
        }
        result = diversity_score(recommendations, item_attributes, "category")
        # Some pairs have same category, some don't
        assert 0.0 < result < 1.0

    def test_diversity_score_empty(self):
        """Test diversity with empty recommendations."""
        result = diversity_score([])
        assert result == 0.0

    def test_diversity_score_entropy(self):
        """Test entropy-based diversity."""
        recommendations = [["item1", "item2", "item3"]]
        item_attributes = {
            "item1": {"category": "A"},
            "item2": {"category": "B"},
            "item3": {"category": "C"}
        }
        result = diversity_score_entropy(recommendations, item_attributes, "category")
        # Maximum entropy when all categories are different
        assert abs(result - 1.0) < 1e-6


class TestNoveltyMetrics:
    """Test novelty metrics."""

    def test_novelty_score(self):
        """Test novelty score based on popularity."""
        recommendations = [["popular", "rare"]]
        item_popularity = {"popular": 0.9, "rare": 0.1}
        result = novelty_score(recommendations, item_popularity, k=2)
        # Rare item should contribute more to novelty
        assert result > 0

    def test_novelty_score_history(self):
        """Test novelty based on user history."""
        recommendations = [["old_item", "new_item"]]
        user_history = [["old_item", "another_old"]]
        result = novelty_score_history(recommendations, user_history)
        # 50% of items are new
        assert abs(result - 0.5) < 1e-6


class TestSerendipityMetrics:
    """Test serendipity metric."""

    def test_serendipity(self):
        """Test serendipity calculation."""
        recommendations = [["new_item", "similar_item"]]
        user_history = [["old_item"]]
        item_similarities = {
            ("new_item", "old_item"): 0.1,
            ("similar_item", "old_item"): 0.9
        }
        ground_truth = [["new_item", "similar_item"]]
        result = serendipity(recommendations, user_history, item_similarities, ground_truth, k=2)
        # new_item is more unexpected and relevant
        assert result > 0


class TestCoverageMetrics:
    """Test coverage metrics."""

    def test_coverage(self):
        """Test catalog coverage."""
        all_recommendations = [["a", "b"], ["b", "c"], ["a", "c"]]
        result = coverage(all_recommendations, catalog_size=5)
        # 3 out of 5 items recommended
        assert abs(result - 0.6) < 1e-6

    def test_coverage_by_category(self):
        """Test coverage by category."""
        all_recommendations = [["a", "b"], ["b", "c"]]
        item_attributes = {
            "a": {"category": "A"},
            "b": {"category": "B"},
            "c": {"category": "A"}
        }
        result = coverage_by_category(all_recommendations, item_attributes, "category")
        # Category A: items a and c in catalog, both are recommended (1.0)
        # Category B: item b in catalog, b is recommended (1.0)
        assert abs(result["A"] - 1.0) < 1e-6
        assert abs(result["B"] - 1.0) < 1e-6


# ============================================================================
# Test Fairness Metrics
# ============================================================================

class TestDemographicParityMetrics:
    """Test demographic parity metrics."""

    def test_demographic_parity(self):
        """Test demographic parity calculation."""
        recommendations = {
            "user1": ["a", "b"],
            "user2": ["a", "c"]
        }
        item_attributes = {
            "a": {"category": "A"},
            "b": {"category": "B"},
            "c": {"category": "A"}
        }
        result = demographic_parity(recommendations, item_attributes, "category")
        # Category A: user1 has 1, user2 has 2 -> avg 1.5
        # Category B: user1 has 1, user2 has 0 -> avg 0.5
        assert abs(result["A"] - 1.5) < 1e-6
        assert abs(result["B"] - 0.5) < 1e-6

    def test_demographic_parity_diff(self):
        """Test demographic parity difference."""
        recommendations = {
            "user1": ["a", "b"],
            "user2": ["a", "c"]
        }
        item_attributes = {
            "a": {"category": "A"},
            "b": {"category": "B"},
            "c": {"category": "A"}
        }
        result = demographic_parity_diff(recommendations, item_attributes, "category")
        # Difference between 1.5 and 0.5
        assert abs(result - 1.0) < 1e-6


class TestEqualizedOddsMetrics:
    """Test equalized odds metrics."""

    def test_equalized_odds(self):
        """Test equalized odds calculation."""
        predictions = ["a", "b", "c"]
        ground_truth = ["a", "b"]
        group_ids = ["group1", "group1", "group2"]
        result = equalized_odds(predictions, ground_truth, group_ids)
        # group1: 2/2 = 1.0, group2: 0/1 = 0.0
        assert abs(result["group1"] - 1.0) < 1e-6
        assert abs(result["group2"] - 0.0) < 1e-6

    def test_equalized_odds_diff(self):
        """Test equalized odds difference."""
        predictions = ["a", "b", "c"]
        ground_truth = ["a", "b"]
        group_ids = ["group1", "group1", "group2"]
        result = equalized_odds_diff(predictions, ground_truth, group_ids)
        # Difference between 1.0 and 0.0
        assert abs(result - 1.0) < 1e-6


class TestDisparateImpactMetrics:
    """Test disparate impact metrics."""

    def test_disparate_impact(self):
        """Test disparate impact calculation."""
        recommendations = {
            "user1": ["a", "b"],
            "user2": ["a", "c"]
        }
        item_attributes = {
            "a": {"category": "A"},
            "b": {"category": "B"},
            "c": {"category": "A"}
        }
        result = disparate_impact(recommendations, item_attributes, "category", "A")
        # A (privileged): avg 1.5 per user, B: avg 0.5 per user
        # disparate_impact returns ratio: A=1.0, B=0.5/1.5=0.333...
        assert abs(result["A"] - 1.0) < 1e-6
        assert abs(result["B"] - 0.5/1.5) < 1e-6


# ============================================================================
# Test Batch Functions
# ============================================================================

class TestBatchFunctions:
    """Test batch computation functions."""

    def test_batch_recall_at_k(self):
        """Test batch recall computation."""
        predictions = [["a", "b"], ["c", "d"]]
        ground_truth = [["a", "c"], ["b", "d"]]
        results = batch_recall_at_k(predictions, ground_truth, k=2)
        assert len(results) == 2
        assert abs(results[0] - 0.5) < 1e-6
        assert abs(results[1] - 0.5) < 1e-6

    def test_batch_precision_at_k(self):
        """Test batch precision computation."""
        predictions = [["a", "b"], ["c", "d"]]
        ground_truth = [["a"], ["c"]]
        results = batch_precision_at_k(predictions, ground_truth, k=2)
        assert len(results) == 2
        assert abs(results[0] - 0.5) < 1e-6
        assert abs(results[1] - 0.5) < 1e-6

    def test_batch_ndcg_at_k(self):
        """Test batch NDCG computation."""
        predictions = [["a", "b", "c"], ["d", "e", "f"]]
        ground_truth = [["a", "b"], ["d", "e"]]
        results = batch_ndcg_at_k(predictions, ground_truth, k=3)
        assert len(results) == 2
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_batch_mrr(self):
        """Test batch MRR computation."""
        predictions = [["a", "b"], ["c", "d"]]
        ground_truth = [["b"], ["e"]]
        results = batch_mrr(predictions, ground_truth)
        assert len(results) == 2
        # First query: MRR = 0.5 (b at position 2)
        # Second query: MRR = 0.0 (e not found)
        assert abs(results[0] - 0.5) < 1e-6
        assert abs(results[1] - 0.0) < 1e-6


# ============================================================================
# Test RecommendationEvaluator
# ============================================================================

class TestRecommendationEvaluator:
    """Test the RecommendationEvaluator class."""

    def test_evaluator_single_query(self):
        """Test evaluation of a single query."""
        evaluator = RecommendationEvaluator(k_values=[5, 10])
        predictions = ["a", "b", "c", "d", "e"]
        ground_truth = ["b", "d"]
        results = evaluator.evaluate(predictions, ground_truth)
        assert "recall@5" in results
        assert "precision@5" in results
        assert "ndcg@5" in results
        assert "mrr" in results

    def test_evaluator_multiple_queries(self):
        """Test evaluation of multiple queries."""
        evaluator = RecommendationEvaluator(k_values=[5])
        predictions = [["a", "b", "c"], ["d", "e", "f"]]
        ground_truth = [["a"], ["d"]]
        results = evaluator.evaluate(predictions, ground_truth)
        assert "recall@5" in results
        assert results["recall@5"] > 0

    def test_evaluator_with_diversity(self):
        """Test evaluation with diversity metrics."""
        evaluator = RecommendationEvaluator(k_values=[5])
        predictions = [["a", "b", "c"]]
        ground_truth = [["a"]]
        item_attributes = {
            "a": {"category": "A"},
            "b": {"category": "B"},
            "c": {"category": "C"}
        }
        results = evaluator.evaluate(predictions, ground_truth, item_attributes=item_attributes)
        assert "diversity" in results
        assert "diversity_entropy" in results

    def test_evaluator_with_novelty(self):
        """Test evaluation with novelty metrics."""
        evaluator = RecommendationEvaluator(k_values=[5])
        predictions = [["a", "b"]]
        ground_truth = [["a"]]
        item_popularity = {"a": 0.5, "b": 0.5}
        results = evaluator.evaluate(predictions, ground_truth, item_popularity=item_popularity)
        assert "novelty" in results

    def test_evaluator_format_report(self):
        """Test report formatting."""
        evaluator = RecommendationEvaluator(k_values=[5, 10])
        results = {
            "recall@5": 0.5,
            "precision@5": 0.4,
            "ndcg@5": 0.6,
            "mrr": 0.5,
            "map": 0.5
        }
        report = evaluator.format_report(results)
        assert "Recommendation Evaluation Report" in report
        assert "recall@5" in report
        assert "mrr" in report

    def test_evaluator_with_user_history(self):
        """Test evaluation with user history for serendipity."""
        evaluator = RecommendationEvaluator(k_values=[5])
        predictions = [["a", "b"]]
        ground_truth = [["a"]]
        user_history = [["c"]]
        item_similarities = {("a", "c"): 0.5, ("b", "c"): 0.8}
        results = evaluator.evaluate(
            predictions, ground_truth,
            user_history=user_history,
            item_similarities=item_similarities
        )
        assert "serendipity" in results

    def test_evaluator_with_coverage(self):
        """Test evaluation with coverage metric."""
        evaluator = RecommendationEvaluator(k_values=[5])
        predictions = [["a", "b"], ["b", "c"]]
        ground_truth = [["a"], ["b"]]
        results = evaluator.evaluate(predictions, ground_truth, catalog_size=5)
        assert "coverage" in results


# ============================================================================
# Test Legacy API
# ============================================================================

class TestLegacyAPI:
    """Test legacy API functions."""

    def test_evaluate_recall(self):
        """Test legacy recall evaluation."""
        predictions = ["a", "b", "c", "d", "e"]
        ground_truth = ["b", "d"]
        results = evaluate_recall(predictions, ground_truth, k_list=[5, 10])
        assert "Recall@5" in results
        assert "Recall@10" in results
        assert abs(results["Recall@5"] - 1.0) < 1e-6

    def test_evaluate_ndcg(self):
        """Test legacy NDCG evaluation."""
        predictions = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        ground_truth = [("a", 1.0), ("b", 1.0)]
        results = evaluate_ndcg(predictions, ground_truth, k_list=[3])
        assert "NDCG@3" in results
        assert 0.0 <= results["NDCG@3"] <= 1.0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for metrics."""

    def test_full_evaluation_pipeline(self):
        """Test a complete evaluation pipeline."""
        # Setup
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

        item_popularity = {
            f"poi{i}": 0.1 * i for i in range(1, 10)
        }

        # Evaluate
        evaluator = RecommendationEvaluator(k_values=[5])
        results = evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            item_attributes=item_attributes,
            item_popularity=item_popularity,
            catalog_size=9
        )

        # Verify all metrics are computed
        assert "recall@5" in results
        assert "precision@5" in results
        assert "f1@5" in results
        assert "ndcg@5" in results
        assert "hitrate@5" in results
        assert "mrr" in results
        assert "map" in results
        assert "diversity" in results
        assert "novelty" in results
        assert "coverage" in results

        # Verify metric ranges
        assert 0.0 <= results["recall@5"] <= 1.0
        assert 0.0 <= results["precision@5"] <= 1.0
        assert 0.0 <= results["ndcg@5"] <= 1.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
