"""
A/B Testing Framework - Usage Examples for GoAfar Project.

This script demonstrates how to use the A/B testing framework for:
1. Offline recommendation model comparison
2. Online A/B testing with incremental data
3. Statistical analysis and visualization
4. Experiment management
"""
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

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
)


# =============================================================================
# Example 1: Traffic Splitting
# =============================================================================

def example_traffic_splitting():
    """Demonstrate traffic splitting strategies."""
    print("\n" + "="*70)
    print("Example 1: Traffic Splitting")
    print("="*70)

    # Create some test user IDs
    user_ids = [f"user_{i}" for i in range(1000)]

    # Strategy 1: Hash-based splitting
    print("\n1. Hash-based Splitting (MD5):")
    splitter = TrafficSplitter(split_ratio=0.5, salt="experiment_1", strategy="hash")
    groups = splitter.split_traffic(user_ids)
    print(f"   Control: {len(groups['control'])} users")
    print(f"   Treatment: {len(groups['treatment'])} users")

    # Strategy 2: Consistent hashing
    print("\n2. Consistent Hashing:")
    splitter = TrafficSplitter(split_ratio=0.5, salt="experiment_1", strategy="consistent")
    groups = splitter.split_traffic(user_ids)
    print(f"   Control: {len(groups['control'])} users")
    print(f"   Treatment: {len(groups['treatment'])} users")

    # Strategy 3: Layered splitting (for orthogonal experiments)
    print("\n3. Layered Splitting (10 layers):")
    splitter = TrafficSplitter(split_ratio=0.5, salt="experiment_1", strategy="layered")
    layered_assignments = splitter.assign_all_layers("user_123")
    print(f"   User 'user_123' assignments across 10 layers:")
    for layer, group in sorted(layered_assignments.items()):
        print(f"     Layer {layer}: {group}")


# =============================================================================
# Example 2: Statistical Tests
# =============================================================================

def example_statistical_tests():
    """Demonstrate statistical significance tests."""
    print("\n" + "="*70)
    print("Example 2: Statistical Tests")
    print("="*70)

    # Simulate metric values for two groups
    np.random.seed(42)
    control_metrics = np.random.normal(0.5, 0.1, 1000).tolist()
    treatment_metrics = np.random.normal(0.55, 0.1, 1000).tolist()

    # T-test
    print("\n1. Welch's T-test:")
    result = StatisticalTest.t_test(control_metrics, treatment_metrics)
    print(f"   Control mean: {np.mean(control_metrics):.4f}")
    print(f"   Treatment mean: {np.mean(treatment_metrics):.4f}")
    print(f"   P-value: {result['pvalue']:.4f}")
    print(f"   Significant: {result['significant']}")
    print(f"   95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    print(f"   Effect size (Cohen's d): {result['effect_size']:.4f}")

    # Mann-Whitney U test (non-parametric)
    print("\n2. Mann-Whitney U Test:")
    result = StatisticalTest.mann_whitney(control_metrics, treatment_metrics)
    print(f"   P-value: {result['pvalue']:.4f}")
    print(f"   Significant: {result['significant']}")
    print(f"   Rank-biserial correlation: {result['rank_biserial']:.4f}")

    # Proportion test (for CTR, CVR, etc.)
    print("\n3. Proportion Test (Z-test):")
    result = StatisticalTest.proportion_test(
        control_successes=100, control_total=1000,  # 10% CTR
        treatment_successes=120, treatment_total=1000,  # 12% CTR
    )
    print(f"   Control CTR: 10.0%")
    print(f"   Treatment CTR: 12.0%")
    print(f"   Lift: {result['lift']:.1%}")
    print(f"   P-value: {result['pvalue']:.4f}")
    print(f"   Significant: {result['significant']}")

    # Bootstrap confidence interval
    print("\n4. Bootstrap 95% CI:")
    result = StatisticalTest.bootstrap_ci(
        control_metrics, treatment_metrics, n_bootstrap=5000
    )
    print(f"   Mean difference: {result['diff_mean']:.4f}")
    print(f"   95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")


# =============================================================================
# Example 3: Offline A/B Testing
# =============================================================================

def example_offline_ab_test():
    """Run an offline A/B test comparing recommenders."""
    print("\n" + "="*70)
    print("Example 3: Offline A/B Test")
    print("="*70)

    # Create test queries
    queries = []
    for i in range(100):
        queries.append({
            "query_id": f"q_{i}",
            "text": f"Find attractions in region {i % 10}",
            "ground_truth": [f"poi_{i*3+j}" for j in range(3)],
        })

    # Define baseline recommender (simple popularity-based)
    def baseline_recommender(query):
        """Simple popularity-based recommender."""
        popular_pois = [f"poi_{j}" for j in range(100)]
        return popular_pois[:10]

    # Define new model (simulated better model)
    def new_model_recommender(query):
        """Improved model with better personalization."""
        # Simulate putting ground truth items higher in ranking
        gt_items = query["ground_truth"]
        other_items = [f"poi_{j}" for j in range(100) if f"poi_{j}" not in gt_items]
        return gt_items + other_items[:7]

    # Run A/B test
    print("\nRunning A/B experiment...")
    config = ABTestConfig(
        split_ratio=0.5,
        salt="rec_comparison",
        min_sample_size=30,
    )
    ab_test = ABTest(config)

    result = ab_test.run_experiment(
        control_fn=baseline_recommender,
        treatment_fn=new_model_recommender,
        test_queries=queries,
        metrics=["recall@10", "ndcg@10", "hit_rate@5"],
    )

    # Print results summary
    print("\n" + result.summary())

    # Export results to DataFrame
    df = result.to_dataframe()
    print("\nResults DataFrame:")
    print(df.to_string())


# =============================================================================
# Example 4: Online A/B Testing
# =============================================================================

def example_online_ab_test():
    """Demonstrate online A/B testing with incremental data."""
    print("\n" + "="*70)
    print("Example 4: Online A/B Testing")
    print("="*70)

    # Setup online test
    config = ABTestConfig(
        split_ratio=0.5,
        salt="online_experiment",
        min_sample_size=50,
    )
    online_test = OnlineABTest(config, max_samples=1000)

    # Simulate users coming in and recording events
    print("\nSimulating user events...")

    for i in range(100):
        user_id = f"user_{i}"

        # User gets assigned (automatically done in record_event)
        group = online_test.assign_user(user_id)

        # Record impression and metrics
        # Simulate treatment being slightly better
        if group == "control":
            recall = np.random.normal(0.45, 0.1)
        else:
            recall = np.random.normal(0.52, 0.1)

        online_test.record_event(user_id, {
            "recall@10": max(0, min(1, recall)),
            "ndcg@10": max(0, min(1, recall + 0.1)),
        })

    print(f"Recorded {len(online_test.control_data)} control events")
    print(f"Recorded {len(online_test.treatment_data)} treatment events")

    # Get current results
    result = online_test.get_current_results(["recall@10", "ndcg@10"])
    print("\nCurrent Results:")
    print(result.summary())

    # Check if we should stop
    should_stop, reason = online_test.should_stop("recall@10", min_effect=0.05)
    print(f"\nShould stop experiment: {should_stop}")
    print(f"Reason: {reason}")


# =============================================================================
# Example 5: Metrics Calculation
# =============================================================================

def example_metrics_calculation():
    """Demonstrate metrics calculation."""
    print("\n" + "="*70)
    print("Example 5: Metrics Calculation")
    print("="*70)

    # Simulate a recommendation scenario
    predictions = ["poi_1", "poi_5", "poi_3", "poi_7", "poi_2", "poi_8", "poi_4", "poi_6", "poi_9", "poi_10"]
    ground_truth = ["poi_1", "poi_2", "poi_3"]

    # User interactions (clicked or not)
    interactions = [True, True, False, True, False, False, True, False, False, False]

    # Dwell times (seconds spent on each POI)
    dwell_times = [120, 60, 30, 90, 45, 20, 150, 15, 10, 5]

    # Calculate metrics
    metrics = ABMetrics.compute_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        user_interactions=interactions,
        dwell_times=dwell_times,
    )

    print("\nRecommendation Metrics:")
    print(f"  Recall@5: {metrics['recall@5']:.4f}")
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  NDCG@5: {metrics['ndcg@5']:.4f}")
    print(f"  NDCG@10: {metrics['ndcg@10']:.4f}")
    print(f"  Hit Rate@1: {metrics['hit_rate@1']:.4f}")
    print(f"  Hit Rate@5: {metrics['hit_rate@5']:.4f}")
    print(f"  CTR: {metrics['ctr']:.4f}")
    print(f"  Clicks: {metrics['clicks']}")
    print(f"  Avg Dwell Time: {metrics['avg_dwell_time']:.2f}s")

    # Business metrics
    print("\nBusiness Metrics:")
    clicks = metrics['clicks']
    impressions = len(predictions)
    conversions = sum(1 for dt in dwell_times if dt > 60)  # Conversion: dwell > 60s

    ctr = ABMetrics.calculate_ctr(clicks, impressions)
    cvr = ABMetrics.calculate_cvr(conversions, clicks)

    print(f"  CTR: {ctr:.2%}")
    print(f"  CVR: {cvr:.2%}")


# =============================================================================
# Example 6: Experiment Management
# =============================================================================

def example_experiment_management():
    """Demonstrate experiment management."""
    print("\n" + "="*70)
    print("Example 6: Experiment Management")
    print("="*70)

    # Create experiment manager
    storage_dir = tempfile.mkdtemp()
    manager = ExperimentManager(storage_path=storage_dir)

    # Create sample experiment result
    control_data = [
        {"query_id": f"q{i}", "recall@10": 0.45 + np.random.normal(0, 0.05)}
        for i in range(50)
    ]
    treatment_data = [
        {"query_id": f"q{i}", "recall@10": 0.52 + np.random.normal(0, 0.05)}
        for i in range(50)
    ]

    result = ABTestResult(
        control_results=control_data,
        treatment_results=treatment_data,
        metrics=["recall@10"],
        config=ABTestConfig(),
    )

    # Register experiment
    config = ExperimentConfig(
        name="recall_improvement_v1",
        description="Test new embedding model for POI retrieval",
        control_name="BGE-M3",
        treatment_name="Qwen-Embedding",
    )
    manager.register(config, result)
    print(f"\nRegistered experiment: {config.name}")

    # List experiments
    experiments = manager.list_experiments()
    print(f"Total experiments: {len(experiments)}")

    # Compare multiple experiments
    # Register another experiment
    control_data2 = [
        {"query_id": f"q{i}", "recall@10": 0.45 + np.random.normal(0, 0.05)}
        for i in range(50)
    ]
    treatment_data2 = [
        {"query_id": f"q{i}", "recall@10": 0.48 + np.random.normal(0, 0.05)}
        for i in range(50)
    ]
    result2 = ABTestResult(
        control_results=control_data2,
        treatment_results=treatment_data2,
        metrics=["recall@10"],
        config=ABTestConfig(),
    )
    config2 = ExperimentConfig(
        name="recall_improvement_v2",
        description="Test different retrieval strategy",
    )
    manager.register(config2, result2)

    # Compare
    comparison = manager.compare_experiments(["recall_improvement_v1", "recall_improvement_v2"])
    print("\nExperiment Comparison:")
    print(comparison.to_string())


# =============================================================================
# Example 7: Complete Workflow
# =============================================================================

def example_complete_workflow():
    """Demonstrate a complete A/B testing workflow."""
    print("\n" + "="*70)
    print("Example 7: Complete A/B Testing Workflow")
    print("="*70)

    # Step 1: Define the experiment
    print("\nStep 1: Define Experiment")
    config = ABTestConfig(
        split_ratio=0.5,
        salt="poi_ranking_experiment",
        significance_level=0.05,
        min_sample_size=100,
        confidence_level=0.95,
    )

    # Step 2: Create test queries
    print("Step 2: Prepare Test Data")
    queries = []
    for i in range(200):
        queries.append({
            "query_id": f"query_{i}",
            "text": f"Attractions in city {i % 20}",
            "ground_truth": [f"poi_{i*3+j}" for j in range(np.random.randint(1, 5))],
        })
    print(f"  Created {len(queries)} test queries")

    # Step 3: Define recommenders
    print("\nStep 3: Define Recommenders")

    def baseline_ranker(query):
        """Simple TF-IDF based ranker."""
        all_pois = [f"poi_{j}" for j in range(500)]
        # Shuffle for randomness
        np.random.shuffle(all_pois)
        return all_pois[:10]

    def neural_ranker(query):
        """Neural semantic search ranker."""
        # Put relevant items first (simulating better model)
        relevant = query["ground_truth"]
        others = [f"poi_{j}" for j in range(500) if f"poi_{j}" not in relevant]
        return relevant + others[:10-len(relevant)]

    print("  Baseline: TF-IDF Ranker")
    print("  Treatment: Neural Semantic Ranker")

    # Step 4: Run experiment
    print("\nStep 4: Run A/B Experiment")
    ab_test = ABTest(config)
    result = ab_test.run_experiment(
        control_fn=baseline_ranker,
        treatment_fn=neural_ranker,
        test_queries=queries,
        metrics=["recall@10", "ndcg@10", "hit_rate@5"],
    )

    # Step 5: Analyze results
    print("\nStep 5: Analyze Results")
    print(result.summary())

    # Step 6: Save results
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        output_path = f.name
    result.save(output_path)
    print(f"\nResults saved to: {output_path}")

    # Step 7: Export for reporting
    df = result.to_dataframe()
    print("\nResults for Report:")
    print(df[["metric", "control_mean", "treatment_mean", "relative_lift", "significant"]].to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    examples = [
        ("Traffic Splitting", example_traffic_splitting),
        ("Statistical Tests", example_statistical_tests),
        ("Offline A/B Test", example_offline_ab_test),
        ("Online A/B Testing", example_online_ab_test),
        ("Metrics Calculation", example_metrics_calculation),
        ("Experiment Management", example_experiment_management),
        ("Complete Workflow", example_complete_workflow),
    ]

    print("\n" + "="*70)
    print("GoAfar A/B Testing Framework - Usage Examples")
    print("="*70)

    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A/B Test Framework Examples")
    parser.add_argument(
        "--example",
        choices=["splitting", "statistical", "offline", "online", "metrics", "management", "workflow"],
        help="Run specific example"
    )
    args = parser.parse_args()

    if args.example:
        example_map = {
            "splitting": example_traffic_splitting,
            "statistical": example_statistical_tests,
            "offline": example_offline_ab_test,
            "online": example_online_ab_test,
            "metrics": example_metrics_calculation,
            "management": example_experiment_management,
            "workflow": example_complete_workflow,
        }
        example_map[args.example]()
    else:
        main()
