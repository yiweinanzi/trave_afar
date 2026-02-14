"""
A/B Testing Framework for Recommendation Systems.

Implements:
- Traffic splitting (hash-based)
- Metric computation with statistical significance testing
- Visualization and reporting
- Experiment management

Usage:
    ab_test = ABTest()
    results = ab_test.run_experiment(
        control_fn=baseline_recommender,
        treatment_fn=new_recommender,
        test_queries=queries,
        metrics=["recall@10", "ndcg@10"]
    )
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Traffic Splitting
# ============================================================================

class TrafficSplitter:
    """
    Split users/items into control and treatment groups.

    Uses deterministic hashing for consistent assignment.
    """

    def __init__(
        self,
        split_ratio: float = 0.5,
        salt: str = "default",
    ):
        """
        Args:
            split_ratio: Proportion for treatment group (0-1)
            salt: Salt for hash to enable independent experiments
        """
        self.split_ratio = split_ratio
        self.salt = salt

    def _hash(self, identifier: str) -> int:
        """Compute hash value for identifier."""
        hash_input = f"{self.salt}:{identifier}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def assign_group(
        self,
        identifier: str,
    ) -> str:
        """
        Assign an identifier to a group.

        Args:
            identifier: User ID, item ID, or query ID

        Returns:
            "control" or "treatment"
        """
        hash_value = self._hash(identifier)
        hash_ratio = hash_value / (2**32)

        return "treatment" if hash_ratio < self.split_ratio else "control"

    def split_traffic(
        self,
        identifiers: List[str],
    ) -> Dict[str, List[str]]:
        """
        Split identifiers into groups.

        Args:
            identifiers: List of user/query IDs

        Returns:
            {"control": [...], "treatment": [...]}
        """
        groups = {"control": [], "treatment": []}

        for identifier in identifiers:
            group = self.assign_group(identifier)
            groups[group].append(identifier)

        return groups


# ============================================================================
# Statistical Testing
# ============================================================================

class StatisticalTest:
    """Statistical significance tests."""

    @staticmethod
    def t_test(
        control: List[float] | np.ndarray,
        treatment: List[float] | np.ndarray,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """
        Two-sample t-test for difference in means.

        Args:
            control: Control group metric values
            treatment: Treatment group metric values
            alternative: "two-sided", "less", or "greater"

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool}
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, using simple t-test")
            # Simple implementation
            control = np.array(control)
            treatment = np.array(treatment)

            n1, n2 = len(control), len(treatment)
            mean1, mean2 = np.mean(control), np.mean(treatment)
            var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)

            pooled_se = np.sqrt(var1/n1 + var2/n2)
            statistic = (mean2 - mean1) / pooled_se if pooled_se > 0 else 0

            # Approximate p-value (assuming normal)
            from math import erf, sqrt
            pvalue = 2 * (1 - erf(abs(statistic) / sqrt(2)))
            significant = pvalue < 0.05

            return {"statistic": statistic, "pvalue": pvalue, "significant": significant}

        statistic, pvalue = stats.ttest_ind(treatment, control, alternative=alternative)
        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": pvalue < 0.05,
        }

    @staticmethod
    def mann_whitney(
        control: List[float] | np.ndarray,
        treatment: List[float] | np.ndarray,
        alternative: str = "two-sided",
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U test (non-parametric).

        Args:
            control: Control group metric values
            treatment: Treatment group metric values
            alternative: "two-sided", "less", or "greater"

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool}
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, using simple test")
            return StatisticalTest.t_test(control, treatment, alternative)

        statistic, pvalue = stats.mannwhitneyu(
            treatment, control, alternative=alternative
        )
        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": pvalue < 0.05,
        }

    @staticmethod
    def proportion_test(
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
    ) -> Dict[str, Any]:
        """
        Z-test for difference in proportions.

        Args:
            control_successes: Successes in control
            control_total: Total in control
            treatment_successes: Successes in treatment
            treatment_total: Total in treatment

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool, "lift": float}
        """
        p1 = control_successes / control_total if control_total > 0 else 0
        p2 = treatment_successes / treatment_total if treatment_total > 0 else 0

        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))

        if se == 0:
            return {"statistic": 0, "pvalue": 1, "significant": False, "lift": 0}

        # Z-statistic
        statistic = (p2 - p1) / se

        # P-value
        from scipy import stats
        pvalue = 2 * (1 - stats.norm.cdf(abs(statistic)))

        # Lift
        lift = (p2 - p1) / p1 if p1 > 0 else 0

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": pvalue < 0.05,
            "lift": float(lift),
        }


# ============================================================================
# Metrics Computation
# ============================================================================

@dataclass
class MetricDefinition:
    """Definition of a metric to compute in A/B test."""

    name: str
    higher_is_better: bool = True
    function: Optional[Callable] = None
    description: str = ""


class ABMetrics:
    """Compute metrics for A/B testing."""

    @staticmethod
    def compute_metrics(
        predictions: List[str],
        ground_truth: List[str],
        user_interactions: Optional[List[bool]] = None,
    ) -> Dict[str, float]:
        """
        Compute metrics for a single query.

        Args:
            predictions: Ordered list of recommended items
            ground_truth: Relevant items
            user_interactions: Whether user actually clicked/visited

        Returns:
            Dictionary of metric values
        """
        results = {}

        # Recall@K
        for k in [5, 10, 20]:
            pred_k = set(predictions[:k])
            true_set = set(ground_truth)
            results[f"recall@{k}"] = len(pred_k & true_set) / len(true_set) if true_set else 0

        # NDCG@K
        for k in [5, 10]:
            relevance = {item: 1.0 for item in ground_truth}
            dcg = 0.0
            for i, item in enumerate(predictions[:k]):
                rel = relevance.get(item, 0.0)
                dcg += (2**rel - 1) / np.log2(i + 2)

            sorted_rel = sorted(relevance.values(), reverse=True)[:k]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted_rel))
            results[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0

        # Click-through rate (if interactions available)
        if user_interactions is not None:
            results["ctr"] = np.mean(user_interactions)
            results["clicks"] = np.sum(user_interactions)

        # Coverage
        results["num_recommended"] = len(predictions)

        return results


# ============================================================================
# A/B Test Runner
# ============================================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B test."""

    split_ratio: float = 0.5
    salt: str = "ab_test"
    significance_level: float = 0.05
    min_sample_size: int = 100


class ABTest:
    """
    Run and analyze A/B tests for recommendation systems.

    Example:
        ab_test = ABTest()

        # Define recommenders
        def baseline(query):
            return ["poi1", "poi2", "poi3"]

        def new_model(query):
            return ["poi4", "poi2", "poi1"]

        # Run experiment
        results = ab_test.run_experiment(
            control_fn=baseline,
            treatment_fn=new_model,
            test_queries=[
                {"query_id": "q1", "text": "museum", "ground_truth": ["poi1"]},
                {"query_id": "q2", "text": "park", "ground_truth": ["poi2"]},
            ],
            metrics=["recall@10", "ndcg@10"],
        )

        # Print results
        print(results.summary())
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        self.config = config or ABTestConfig()
        self.splitter = TrafficSplitter(
            split_ratio=self.config.split_ratio,
            salt=self.config.salt,
        )

    def run_experiment(
        self,
        control_fn: Callable[[Dict], List[str]],
        treatment_fn: Callable[[Dict], List[str]],
        test_queries: List[Dict],
        metrics: List[str] | None = None,
    ) -> "ABTestResult":
        """
        Run A/B experiment.

        Args:
            control_fn: Function that takes query dict and returns recommendations
            treatment_fn: Function that takes query dict and returns recommendations
            test_queries: List of query dictionaries with "query_id" and "ground_truth"
            metrics: List of metric names to compute

        Returns:
            ABTestResult with analysis
        """
        # Split traffic
        query_ids = [q.get("query_id", f"q{i}") for i, q in enumerate(test_queries)]
        groups = self.splitter.split_traffic(query_ids)

        # Create query lookup
        query_lookup = {q.get("query_id", f"q{i}"): q for i, q in enumerate(test_queries)}

        # Run experiment
        control_results = []
        treatment_results = []

        for qid in groups["control"]:
            query = query_lookup[qid]
            recs = control_fn(query)
            metrics = ABMetrics.compute_metrics(
                recs,
                query.get("ground_truth", []),
                query.get("interactions"),
            )
            control_results.append({"query_id": qid, **metrics})

        for qid in groups["treatment"]:
            query = query_lookup[qid]
            recs = treatment_fn(query)
            metrics_val = ABMetrics.compute_metrics(
                recs,
                query.get("ground_truth", []),
                query.get("interactions"),
            )
            treatment_results.append({"query_id": qid, **metrics_val})

        # Analyze results
        return ABTestResult(
            control_results=control_results,
            treatment_results=treatment_results,
            metrics=metrics or ["recall@10", "ndcg@10"],
            config=self.config,
        )


@dataclass
class ABTestResult:
    """Results of an A/B test."""

    control_results: List[Dict]
    treatment_results: List[Dict]
    metrics: List[str]
    config: ABTestConfig

    def analyze(self) -> Dict[str, Any]:
        """Perform statistical analysis on results."""
        analysis = {}

        control_df = pd.DataFrame(self.control_results)
        treatment_df = pd.DataFrame(self.treatment_results)

        for metric in self.metrics:
            if metric not in control_df.columns:
                continue

            control_values = control_df[metric].dropna().values
            treatment_values = treatment_df[metric].dropna().values

            # Descriptive stats
            analysis[metric] = {
                "control_mean": float(np.mean(control_values)),
                "control_std": float(np.std(control_values)),
                "treatment_mean": float(np.mean(treatment_values)),
                "treatment_std": float(np.std(treatment_values)),
                "relative_lift": float(
                    (np.mean(treatment_values) - np.mean(control_values))
                    / np.mean(control_values) if np.mean(control_values) > 0 else 0
                ),
            }

            # Statistical test
            test_result = StatisticalTest.t_test(control_values, treatment_values)
            analysis[metric]["test"] = test_result

        return analysis

    def summary(self) -> str:
        """Generate human-readable summary."""
        analysis = self.analyze()

        lines = [
            "=" * 70,
            "A/B Test Results",
            "=" * 70,
            f"Split Ratio: {self.config.split_ratio}",
            f"Control Queries: {len(self.control_results)}",
            f"Treatment Queries: {len(self.treatment_results)}",
            f"Significance Level: {self.config.significance_level}",
            "",
        ]

        for metric, results in analysis.items():
            lines.append(f"Metric: {metric}")
            lines.append(f"  Control:     {results['control_mean']:.4f} ± {results['control_std']:.4f}")
            lines.append(f"  Treatment:   {results['treatment_mean']:.4f} ± {results['treatment_std']:.4f}")
            lines.append(f"  Lift:        {results['relative_lift']:+.2%}")

            test = results["test"]
            significance = "***" if test["significant"] else "ns"
            lines.append(f"  t-test:      t={test['statistic']:.4f}, p={test['pvalue']:.4f} {significance}")
            lines.append("")

        lines.append("=" * 70)
        lines.append("Legend: *** p < 0.05 (significant), ns = not significant")
        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as DataFrame."""
        analysis = self.analyze()

        rows = []
        for metric, results in analysis.items():
            rows.append({
                "metric": metric,
                "control_mean": results["control_mean"],
                "treatment_mean": results["treatment_mean"],
                "relative_lift": results["relative_lift"],
                "pvalue": results["test"]["pvalue"],
                "significant": results["test"]["significant"],
            })

        return pd.DataFrame(rows)

    def plot(self, output_path: Optional[str] = None) -> None:
        """Generate visualization of results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
            return

        analysis = self.analyze()
        metrics = [m for m in self.metrics if m in analysis]

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            results = analysis[metric]

            # Bar plot
            x = ["Control", "Treatment"]
            y = [results["control_mean"], results["treatment_mean"]]
            yerr = [results["control_std"], results["treatment_std"]]

            bars = ax.bar(x, y, yerr=yerr, alpha=0.7, capsize=5)

            # Color based on significance
            if results["test"]["significant"]:
                bars[1].set_color("green")
            else:
                bars[1].set_color("gray")

            ax.set_ylabel(metric)
            ax.set_title(f"{metric}\nLift: {results['relative_lift']:+.1%}")

            # Add significance indicator
            if results["test"]["significant"]:
                ax.text(1, y[1] + yerr[1], "***", ha="center", fontsize=16)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Plot saved to {output_path}")
        else:
            plt.show()


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B Testing Framework")
    parser.add_argument("--control", required=True, help="Path to control results JSON")
    parser.add_argument("--treatment", required=True, help="Path to treatment results JSON")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--plot", help="Path to save plot PNG")
    args = parser.parse_args()

    # Load results
    import json

    with open(args.control) as f:
        control = json.load(f)
    with open(args.treatment) as f:
        treatment = json.load(f)

    # Analyze
    result = ABTestResult(
        control_results=control,
        treatment_results=treatment,
        metrics=["recall@10", "ndcg@10"],
        config=ABTestConfig(),
    )

    # Print summary
    print(result.summary())

    # Save results
    if args.output:
        df = result.to_dataframe()
        df.to_json(args.output, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Generate plot
    if args.plot:
        result.plot(args.plot)


if __name__ == "__main__":
    main()
