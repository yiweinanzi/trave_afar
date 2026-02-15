"""
Unit tests for A/B Testing Framework.

Tests for:
- Traffic splitting strategies
- Statistical tests
- Metrics computation
- A/B test runner
- Online A/B testing
"""
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from evaluation.ab_test import (
    TrafficSplitter,
    HashBasedSplit,
    ConsistentHashSplit,
    LayeredSplit,
    StatisticalTest,
    ABMetrics,
    ABTestConfig,
    ABTest,
    ABTestResult,
    ExperimentConfig,
    ExperimentManager,
    OnlineABTest,
    SplitStrategy,
)


# =============================================================================
# Test Traffic Splitting
# =============================================================================

class TestTrafficSplitter:
    """Tests for traffic splitting functionality."""

    def test_split_ratio(self):
        """Test that split ratio is approximately correct."""
        splitter = TrafficSplitter(split_ratio=0.5, salt="test")

        identifiers = [f"user_{i}" for i in range(1000)]
        groups = splitter.split_traffic(identifiers)

        # Check distribution
        control_count = len(groups["control"])
        treatment_count = len(groups["treatment"])

        assert control_count + treatment_count == 1000
        # Allow some deviation but should be close to 50%
        ratio = treatment_count / 1000
        assert 0.45 <= ratio <= 0.55, f"Ratio {ratio} outside expected range"

    def test_consistent_assignment(self):
        """Test that assignment is consistent for same identifier."""
        splitter = TrafficSplitter(split_ratio=0.5, salt="test")

        user_id = "user_123"
        group1 = splitter.assign_group(user_id)
        group2 = splitter.assign_group(user_id)

        assert group1 == group2, "Assignment should be consistent"

    def test_salt_changes_assignment(self):
        """Test that different salts produce different assignments."""
        splitter1 = TrafficSplitter(split_ratio=0.5, salt="salt1")
        splitter2 = TrafficSplitter(split_ratio=0.5, salt="salt2")

        user_id = "user_123"
        group1 = splitter1.assign_group(user_id)
        group2 = splitter2.assign_group(user_id)

        # Different salt may produce different group (not guaranteed for single user)
        # But overall distribution should differ
        identifiers = [f"user_{i}" for i in range(100)]
        groups1 = splitter1.split_traffic(identifiers)
        groups2 = splitter2.split_traffic(identifiers)

        # Count differences
        diff_count = sum(
            1 for uid in identifiers
            if splitter1.assign_group(uid) != splitter2.assign_group(uid)
        )
        # Should have some differences
        assert diff_count > 30, f"Only {diff_count} differences with different salts"

    def test_hash_based_strategy(self):
        """Test hash-based splitting strategy."""
        strategy = HashBasedSplit()

        hash1 = strategy._hash("user_123", "salt1")
        hash2 = strategy._hash("user_123", "salt1")
        hash3 = strategy._hash("user_123", "salt2")

        assert hash1 == hash2, "Same input should produce same hash"
        assert hash1 != hash3, "Different salt should produce different hash"

        # Test assignment
        group = strategy.assign("user_123", 0.5, "salt1")
        assert group in ["control", "treatment"]

    def test_consistent_hash_strategy(self):
        """Test consistent hashing strategy."""
        strategy = ConsistentHashSplit(buckets=1000)

        # Should distribute evenly across buckets
        identifiers = [f"user_{i}" for i in range(1000)]
        buckets = [strategy._hash(uid, "salt1") for uid in identifiers]

        # Check distribution
        unique_buckets = len(set(buckets))
        assert unique_buckets > 900, "Should distribute across most buckets"

    def test_layered_split_strategy(self):
        """Test layered splitting for orthogonal experiments."""
        strategy = LayeredSplit(num_layers=5)

        identifier = "user_123"
        assignments = strategy.assign_all_layers(identifier, 0.5, "salt1")

        assert len(assignments) == 5
        for layer, group in assignments.items():
            assert group in ["control", "treatment"]

    def test_multiple_groups(self):
        """Test splitting into multiple groups."""
        splitter = TrafficSplitter(split_ratio=0.33, salt="test")

        identifiers = [f"user_{i}" for i in range(300)]
        groups = splitter.split_traffic_multiple(identifiers, num_groups=3)

        assert len(groups) == 3
        assert sum(len(v) for v in groups.values()) == 300

        # Each group should have roughly equal size
        sizes = [len(groups[f"group_{i}"]) for i in range(3)]
        avg_size = 300 / 3
        for size in sizes:
            assert avg_size * 0.7 <= size <= avg_size * 1.3, f"Size {size} outside range"


# =============================================================================
# Test Statistical Tests
# =============================================================================

class TestStatisticalTests:
    """Tests for statistical significance tests."""

    def test_t_test_different_means(self):
        """Test t-test with different means."""
        control = [1.0] * 100
        treatment = [2.0] * 100

        result = StatisticalTest.t_test(control, treatment)

        assert result["significant"], "Should detect significant difference"
        assert result["pvalue"] < 0.01
        assert result["mean_diff"] > 0

    def test_t_test_same_means(self):
        """Test t-test with same means."""
        np.random.seed(42)
        control = np.random.normal(1.0, 0.1, 100).tolist()
        treatment = np.random.normal(1.0, 0.1, 100).tolist()

        result = StatisticalTest.t_test(control, treatment)

        assert not result["significant"], "Should not detect difference"
        assert result["pvalue"] > 0.05

    def test_t_test_confidence_interval(self):
        """Test t-test confidence interval."""
        control = [1.0] * 100
        treatment = [2.0] * 100

        result = StatisticalTest.t_test(control, treatment, confidence=0.95)

        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] < result["ci_upper"]
        # CI should contain the true difference (1.0)
        assert result["ci_lower"] < 1.0 < result["ci_upper"]

    def test_t_test_without_scipy_fallback_pvalue_range(self):
        """Test t-test fallback (no scipy) keeps p-value in valid range."""
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy not available")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = StatisticalTest.t_test([1.0, 1.1, 0.9], [1.2, 1.3, 1.1])

        assert 0.0 <= result["pvalue"] <= 1.0

    def test_t_test_without_scipy_respects_alternative(self):
        """Test t-test fallback (no scipy) respects one-sided alternatives."""
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy not available")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            greater = StatisticalTest.t_test(
                [1.0, 1.1, 0.9],
                [1.3, 1.2, 1.4],
                alternative="greater",
            )
            less = StatisticalTest.t_test(
                [1.0, 1.1, 0.9],
                [1.3, 1.2, 1.4],
                alternative="less",
            )

        assert 0.0 <= greater["pvalue"] <= 1.0
        assert 0.0 <= less["pvalue"] <= 1.0
        assert greater["pvalue"] < less["pvalue"]

    def test_mann_whitney(self):
        """Test Mann-Whitney U test."""
        control = [1.0] * 50 + [2.0] * 50
        treatment = [3.0] * 50 + [4.0] * 50

        result = StatisticalTest.mann_whitney(control, treatment)

        assert result["significant"]
        assert "rank_biserial" in result

    def test_proportion_test(self):
        """Test proportion test."""
        # Control: 10% conversion
        # Treatment: 15% conversion
        result = StatisticalTest.proportion_test(
            control_successes=100,
            control_total=1000,
            treatment_successes=150,
            treatment_total=1000,
        )

        assert result["lift"] > 0
        assert result["lift"] == 0.5  # 50% lift
        # Should be significant with these sample sizes
        assert result["significant"]

    def test_proportion_test_no_difference(self):
        """Test proportion test with no difference."""
        result = StatisticalTest.proportion_test(
            control_successes=100,
            control_total=1000,
            treatment_successes=100,
            treatment_total=1000,
        )

        assert abs(result["lift"]) < 0.01
        assert not result["significant"]

    def test_proportion_test_without_scipy_fallback_pvalue_range(self):
        """Test proportion test fallback (no scipy) keeps p-value in valid range."""
        original_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "scipy" or name.startswith("scipy."):
                raise ImportError("scipy not available")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            result = StatisticalTest.proportion_test(
                control_successes=100,
                control_total=1000,
                treatment_successes=120,
                treatment_total=1000,
            )

        assert 0.0 <= result["pvalue"] <= 1.0

    def test_chi_square_test(self):
        """Test chi-square test."""
        result = StatisticalTest.chi_square_test(
            control_successes=100,
            control_failures=900,
            treatment_successes=150,
            treatment_failures=850,
        )

        assert "cramer_v" in result
        assert 0 <= result["cramer_v"] <= 1

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        np.random.seed(42)
        control = np.random.normal(1.0, 0.1, 100).tolist()
        treatment = np.random.normal(1.2, 0.1, 100).tolist()

        result = StatisticalTest.bootstrap_ci(control, treatment, n_bootstrap=1000)

        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "diff_mean" in result
        assert result["diff_mean"] > 0


# =============================================================================
# Test Metrics Computation
# =============================================================================

class TestABMetrics:
    """Tests for metrics computation."""

    def test_recall_at_k(self):
        """Test Recall@K computation."""
        predictions = ["poi1", "poi2", "poi3", "poi4", "poi5"]
        ground_truth = ["poi2", "poi4", "poi6"]

        results = ABMetrics.compute_metrics(predictions, ground_truth)

        # Recall@5 should be 2/3 (poi2, poi4 matched)
        assert abs(results["recall@5"] - 2/3) < 0.01

    def test_ndcg_at_k(self):
        """Test NDCG@K computation."""
        predictions = ["poi1", "poi2", "poi3", "poi4", "poi5"]
        ground_truth = ["poi1", "poi2", "poi3"]

        results = ABMetrics.compute_metrics(predictions, ground_truth)

        # NDCG@5 should be 1.0 (all relevant items at top)
        assert abs(results["ndcg@5"] - 1.0) < 0.01

    def test_hit_rate(self):
        """Test Hit Rate@K computation."""
        predictions = ["poi1", "poi2", "poi3"]
        ground_truth = ["poi5", "poi6"]

        results = ABMetrics.compute_metrics(predictions, ground_truth)

        assert results["hit_rate@1"] == 0.0
        assert results["hit_rate@5"] == 0.0

    def test_ctr_calculation(self):
        """Test CTR calculation."""
        ctr = ABMetrics.calculate_ctr(clicks=10, impressions=100)
        assert ctr == 0.1

    def test_cvr_calculation(self):
        """Test CVR calculation."""
        cvr = ABMetrics.calculate_cvr(conversions=5, clicks=50)
        assert cvr == 0.1

    def test_dwell_metrics(self):
        """Test dwell time metrics."""
        dwell_times = [10.0, 20.0, 30.0, 40.0, 50.0]

        metrics = ABMetrics.calculate_dwell_metrics(dwell_times)

        assert metrics["mean"] == 30.0
        assert metrics["median"] == 30.0

    def test_with_interactions(self):
        """Test metrics with user interactions."""
        predictions = ["poi1", "poi2", "poi3", "poi4", "poi5"]
        ground_truth = ["poi1", "poi2"]
        interactions = [True, True, False, False, False]

        results = ABMetrics.compute_metrics(
            predictions, ground_truth, user_interactions=interactions
        )

        assert results["ctr"] == 0.4  # 2 out of 5 clicked
        assert results["clicks"] == 2


# =============================================================================
# Test A/B Test Runner
# =============================================================================

class TestABTest:
    """Tests for A/B test runner."""

    def setup_test_queries(self):
        """Create test queries for experiments."""
        return [
            {"query_id": f"q{i}", "text": f"query {i}",
             "ground_truth": [f"poi{i*3}", f"poi{i*3+1}"]}
            for i in range(100)
        ]

    def test_run_experiment(self):
        """Test running a full experiment."""
        queries = self.setup_test_queries()

        # Define mock recommenders
        def control_fn(query):
            # Returns ground truth items first
            return query["ground_truth"] + ["poi_other"]

        def treatment_fn(query):
            # Returns all items in random order
            all_pois = [f"poi{j}" for j in range(100)]
            return all_pois[:10]

        ab_test = ABTest()
        result = ab_test.run_experiment(
            control_fn=control_fn,
            treatment_fn=treatment_fn,
            test_queries=queries,
            metrics=["recall@10", "ndcg@10"],
        )

        assert len(result.control_results) > 0
        assert len(result.treatment_results) > 0
        assert len(result.control_results) + len(result.treatment_results) == 100

    def test_analysis(self):
        """Test statistical analysis."""
        control_data = [
            {"query_id": f"q{i}", "recall@10": 0.5, "ndcg@10": 0.6}
            for i in range(50)
        ]
        treatment_data = [
            {"query_id": f"q{i}", "recall@10": 0.6, "ndcg@10": 0.7}
            for i in range(50)
        ]

        result = ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=["recall@10", "ndcg@10"],
            config=ABTestConfig(),
        )

        analysis = result.analyze()

        assert "recall@10" in analysis
        assert "ndcg@10" in analysis

        recall_results = analysis["recall@10"]
        assert "control_mean" in recall_results
        assert "treatment_mean" in recall_results
        assert "relative_lift" in recall_results
        assert "t_test" in recall_results

    def test_summary(self):
        """Test summary generation."""
        control_data = [
            {"query_id": f"q{i}", "recall@10": 0.5, "ndcg@10": 0.6}
            for i in range(50)
        ]
        treatment_data = [
            {"query_id": f"q{i}", "recall@10": 0.6, "ndcg@10": 0.7}
            for i in range(50)
        ]

        result = ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=["recall@10"],
            config=ABTestConfig(),
        )

        summary = result.summary()
        assert "A/B Test Results" in summary
        assert "recall@10" in summary
        assert "Control" in summary
        assert "Treatment" in summary

    def test_to_dataframe(self):
        """Test DataFrame export."""
        control_data = [
            {"query_id": f"q{i}", "recall@10": 0.5}
            for i in range(50)
        ]
        treatment_data = [
            {"query_id": f"q{i}", "recall@10": 0.6}
            for i in range(50)
        ]

        result = ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=["recall@10"],
            config=ABTestConfig(),
        )

        df = result.to_dataframe()
        assert len(df) == 1
        assert "metric" in df.columns
        assert "relative_lift" in df.columns
        assert "pvalue" in df.columns

    def test_save_and_load(self):
        """Test saving and loading results."""
        control_data = [
            {"query_id": f"q{i}", "recall@10": 0.5}
            for i in range(10)
        ]
        treatment_data = [
            {"query_id": f"q{i}", "recall@10": 0.6}
            for i in range(10)
        ]

        result = ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=["recall@10"],
            config=ABTestConfig(),
        )

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            result.save(temp_path)

            # Load and verify
            with open(temp_path) as f:
                data = json.load(f)

            assert "timestamp" in data
            assert "analysis" in data
            assert "sample_sizes" in data
        finally:
            os.unlink(temp_path)


# =============================================================================
# Test Experiment Manager
# =============================================================================

class TestExperimentManager:
    """Tests for experiment management."""

    def test_register_and_list(self):
        """Test registering and listing experiments."""
        manager = ExperimentManager(storage_path=tempfile.mkdtemp())

        control_data = [{"query_id": "q1", "recall@10": 0.5}]
        treatment_data = [{"query_id": "q1", "recall@10": 0.6}]

        result = ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=["recall@10"],
            config=ABTestConfig(),
        )

        config = ExperimentConfig(
            name="test_exp",
            description="Test experiment",
        )

        manager.register(config, result)

        experiments = manager.list_experiments()
        assert "test_exp" in experiments

    def test_compare_experiments(self):
        """Test comparing multiple experiments."""
        manager = ExperimentManager(storage_path=tempfile.mkdtemp())

        # Create two experiments
        for i, (control_val, treatment_val) in enumerate([(0.5, 0.6), (0.4, 0.7)]):
            control_data = [{"query_id": "q1", "recall@10": control_val}]
            treatment_data = [{"query_id": "q1", "recall@10": treatment_val}]

            result = ABTestResult(
                control_results=control_data,
                treatment_results=treatment_data,
                metrics=["recall@10"],
                config=ABTestConfig(),
            )

            config = ExperimentConfig(name=f"exp_{i}")
            manager.register(config, result)

        df = manager.compare_experiments(["exp_0", "exp_1"])
        assert len(df) == 2
        assert "experiment" in df.columns
        assert "relative_lift" in df.columns


# =============================================================================
# Test Online A/B Testing
# =============================================================================

class TestOnlineABTest:
    """Tests for online A/B testing."""

    def test_user_assignment(self):
        """Test user assignment to groups."""
        config = ABTestConfig()
        online_test = OnlineABTest(config)

        user_id = "user_123"
        group = online_test.assign_user(user_id)

        assert group in ["control", "treatment"]

        # Same user should get same group
        group2 = online_test.assign_user(user_id)
        assert group == group2

    def test_record_event(self):
        """Test recording events."""
        config = ABTestConfig()
        online_test = OnlineABTest(config)

        online_test.assign_user("user_1")
        online_test.record_event("user_1", {"recall@10": 0.5})

        online_test.assign_user("user_2")
        online_test.record_event("user_2", {"recall@10": 0.6})

        assert len(online_test.control_data) + len(online_test.treatment_data) == 2

    def test_get_current_results(self):
        """Test getting current results."""
        config = ABTestConfig(min_sample_size=2)
        online_test = OnlineABTest(config)

        # Add some data
        for i in range(10):
            user_id = f"user_{i}"
            online_test.assign_user(user_id)
            online_test.record_event(user_id, {"recall@10": 0.5})

        result = online_test.get_current_results(["recall@10"])

        assert len(result.control_results) + len(result.treatment_results) == 10

    def test_should_stop(self):
        """Test early stopping logic."""
        config = ABTestConfig(min_sample_size=10, significance_level=0.05)
        online_test = OnlineABTest(config)

        # Not enough data
        should_stop, reason = online_test.should_stop("recall@10")
        assert not should_stop
        assert "minimum" in reason.lower()

        # Add significant data
        for i in range(100):
            user_id = f"user_{i}"
            online_test.assign_user(user_id)
            # Make treatment clearly better
            value = 0.3 if online_test.assign_user(user_id) == "control" else 0.6
            online_test.record_event(user_id, {"recall@10": value})

        # May or may not stop depending on randomness
        should_stop, reason = online_test.should_stop("recall@10", min_effect=0.1)
        # At least should have enough data now
        assert "minimum" not in reason.lower()


# =============================================================================
# Test Integration
# =============================================================================

class TestIntegration:
    """Integration tests for the full framework."""

    def test_full_workflow(self):
        """Test a complete A/B testing workflow."""
        # Create test data
        queries = [
            {"query_id": f"q{i}", "text": f"query {i}",
             "ground_truth": [f"poi{i*3}", f"poi{i*3+1}"]}
            for i in range(200)
        ]

        # Define recommenders
        def control_fn(query):
            return query["ground_truth"] + ["poi_other1", "poi_other2"]

        def treatment_fn(query):
            # Slightly better ordering
            return query["ground_truth"][::-1] + [f"poi_{i}" for i in range(10)]

        # Run experiment
        config = ABTestConfig(split_ratio=0.5, salt="integration_test")
        ab_test = ABTest(config)
        result = ab_test.run_experiment(
            control_fn=control_fn,
            treatment_fn=treatment_fn,
            test_queries=queries,
            metrics=["recall@10", "ndcg@10", "hit_rate@5"],
        )

        # Analyze results
        analysis = result.analyze()

        # Verify all metrics are analyzed
        assert len(analysis) == 3

        # Generate summary
        summary = result.summary()
        assert "200" in summary  # Total queries

        # Export to DataFrame
        df = result.to_dataframe()
        assert len(df) == 3

    def test_log_based_analysis(self):
        """Test analysis from pre-collected logs."""
        # Simulate logged data
        control_logs = []
        treatment_logs = []

        for i in range(100):
            control_logs.append({
                "query_id": f"q{i}",
                "recall@10": np.random.normal(0.5, 0.1),
                "ndcg@10": np.random.normal(0.6, 0.1),
            })
            treatment_logs.append({
                "query_id": f"q{i}",
                "recall@10": np.random.normal(0.55, 0.1),
                "ndcg@10": np.random.normal(0.65, 0.1),
            })

        config = ABTestConfig()
        ab_test = ABTest(config)
        result = ab_test.run_from_logs(
            control_data=control_logs,
            treatment_data=treatment_logs,
            metrics=["recall@10", "ndcg@10"],
        )

        analysis = result.analyze()
        assert "recall@10" in analysis
        assert "ndcg@10" in analysis


# =============================================================================
# Run Tests
# =============================================================================

def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestTrafficSplitter,
        TestStatisticalTests,
        TestABMetrics,
        TestABTest,
        TestExperimentManager,
        TestOnlineABTest,
        TestIntegration,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        test_instance = test_class()

        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                method = getattr(test_instance, method_name)
                try:
                    method()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    traceback.print_exc()
                    failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print('='*60)

    return failed == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run A/B Test Framework Unit Tests")
    parser.add_argument("--cls", help="Run specific test class", dest="cls_name")
    parser.add_argument("--method", help="Run specific test method")
    args = parser.parse_args()

    if args.cls_name:
        # Run specific class
        test_class = globals()[f"Test{args.cls_name}"]
        test_instance = test_class()

        if args.method:
            # Run specific method
            method = getattr(test_instance, f"test_{args.method}")
            method()
        else:
            # Run all methods in class
            for method_name in dir(test_instance):
                if method_name.startswith('test_'):
                    method = getattr(test_instance, method_name)
                    method()
    else:
        # Run all tests
        success = run_tests()
        sys.exit(0 if success else 1)
