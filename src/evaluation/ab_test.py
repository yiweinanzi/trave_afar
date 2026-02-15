"""
A/B Testing Framework for Recommendation Systems.

Implements:
- Traffic splitting (hash-based, consistent hashing, layered, whitelist)
- Metric computation (CTR, CVR, dwell time, NDCG, Recall, Funnel)
- Statistical significance testing (t-test, Mann-Whitney U, Chi-square, Bootstrap)
- Visualization and reporting
- Experiment management with result export

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
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Traffic Splitting Strategies
# ============================================================================

class SplitStrategy:
    """Base class for traffic splitting strategies."""

    def assign(self, identifier: str, ratio: float, salt: str) -> str:
        """Assign an identifier to a group."""
        raise NotImplementedError


class HashBasedSplit(SplitStrategy):
    """Hash-based splitting using MD5."""

    def __init__(self, hash_fn: str = "md5"):
        self.hash_fn = hash_fn

    def _hash(self, identifier: str, salt: str) -> int:
        """Compute hash value for identifier."""
        hash_input = f"{salt}:{identifier}"

        if self.hash_fn == "md5":
            hash_bytes = hashlib.md5(hash_input.encode()).digest()
        elif self.hash_fn == "sha256":
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        else:
            hash_bytes = hashlib.md5(hash_input.encode()).digest()

        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def assign(self, identifier: str, ratio: float, salt: str) -> str:
        """Assign identifier to group based on hash."""
        hash_value = self._hash(identifier, salt)
        hash_ratio = hash_value / (2**32)
        return "treatment" if hash_ratio < ratio else "control"


class ConsistentHashSplit(SplitStrategy):
    """Consistent hashing for better distribution stability."""

    def __init__(self, buckets: int = 1000):
        self.buckets = buckets

    def _hash(self, identifier: str, salt: str) -> int:
        """Compute consistent hash value."""
        hash_input = f"{salt}:{identifier}"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def assign(self, identifier: str, ratio: float, salt: str) -> str:
        """Assign identifier to group."""
        bucket = self._hash(identifier, salt) % self.buckets
        threshold = int(self.buckets * ratio)
        return "treatment" if bucket < threshold else "control"


class LayeredSplit(SplitStrategy):
    """Layered splitting for orthogonal experiments."""

    def __init__(self, num_layers: int = 10):
        self.num_layers = num_layers

    def _hash(self, identifier: str, salt: str, layer: int) -> int:
        """Compute hash for specific layer."""
        hash_input = f"{salt}:layer{layer}:{identifier}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def assign(self, identifier: str, ratio: float, salt: str, layer: int = 0) -> str:
        """Assign identifier to group for a specific layer."""
        hash_value = self._hash(identifier, salt, layer)
        hash_ratio = hash_value / (2**32)
        return "treatment" if hash_ratio < ratio else "control"

    def assign_all_layers(self, identifier: str, ratio: float, salt: str) -> Dict[int, str]:
        """Assign identifier across all layers."""
        return {
            layer: self.assign(identifier, ratio, salt, layer)
            for layer in range(self.num_layers)
        }


class TrafficSplitter:
    """
    Split users/items into control and treatment groups.

    Uses deterministic hashing for consistent assignment.
    Supports multiple splitting strategies and whitelists.
    """

    STRATEGIES = {
        "hash": HashBasedSplit,
        "consistent": ConsistentHashSplit,
        "layered": LayeredSplit,
    }

    def __init__(
        self,
        split_ratio: float = 0.5,
        salt: str = "default",
        strategy: str = "hash",
        whitelist: Optional[List[str]] = None,
        whitelist_group: str = "treatment",
        **strategy_kwargs
    ):
        """
        Args:
            split_ratio: Proportion for treatment group (0-1)
            salt: Salt for hash to enable independent experiments
            strategy: Splitting strategy ("hash", "consistent", "layered")
            whitelist: List of identifiers that always go to a specific group
            whitelist_group: Which group whitelist identifiers go to
            **strategy_kwargs: Additional arguments for strategy
        """
        self.split_ratio = split_ratio
        self.salt = salt
        self.strategy_name = strategy
        self.whitelist = set(whitelist or [])
        self.whitelist_group = whitelist_group

        strategy_class = self.STRATEGIES.get(strategy, HashBasedSplit)
        self.strategy = strategy_class(**strategy_kwargs)

    def _hash(self, identifier: str) -> int:
        """Compute hash value for identifier (legacy)."""
        if hasattr(self.strategy, '_hash'):
            return self.strategy._hash(identifier, self.salt)
        # Fallback
        hash_input = f"{self.salt}:{identifier}"
        hash_bytes = hashlib.md5(hash_input.encode()).digest()
        return int.from_bytes(hash_bytes[:4], byteorder='big')

    def assign_group(
        self,
        identifier: str,
        **kwargs
    ) -> str:
        """
        Assign an identifier to a group.

        Args:
            identifier: User ID, item ID, or query ID
            **kwargs: Additional arguments (e.g., layer for layered strategy)

        Returns:
            "control" or "treatment"
        """
        # Check whitelist first
        if identifier in self.whitelist:
            return self.whitelist_group

        return self.strategy.assign(identifier, self.split_ratio, self.salt, **kwargs)

    def assign_all_layers(self, identifier: str) -> Dict[int, str]:
        """Assign identifier across all layers (for layered strategy)."""
        # Check whitelist first
        if identifier in self.whitelist:
            return {i: self.whitelist_group for i in range(10)}

        if isinstance(self.strategy, LayeredSplit):
            return self.strategy.assign_all_layers(identifier, self.split_ratio, self.salt)
        return {0: self.assign_group(identifier)}

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

    def split_traffic_multiple(
        self,
        identifiers: List[str],
        num_groups: int = 3,
    ) -> Dict[str, List[str]]:
        """
        Split identifiers into multiple groups.

        Args:
            identifiers: List of user/query IDs
            num_groups: Number of groups (e.g., 3 for A/B/C test)

        Returns:
            {"group_0": [...], "group_1": [...], "group_2": [...]}
        """
        groups = {f"group_{i}": [] for i in range(num_groups)}

        for identifier in identifiers:
            # Skip whitelist users from multi-group tests
            if identifier in self.whitelist:
                continue

            hash_value = self._hash(identifier)
            bucket = hash_value % num_groups
            groups[f"group_{bucket}"].append(identifier)

        return groups

    def add_to_whitelist(self, identifiers: Union[str, List[str]]) -> None:
        """Add identifiers to the whitelist."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        self.whitelist.update(identifiers)

    def remove_from_whitelist(self, identifiers: Union[str, List[str]]) -> None:
        """Remove identifiers from the whitelist."""
        if isinstance(identifiers, str):
            identifiers = [identifiers]
        self.whitelist.difference_update(identifiers)


# ============================================================================
# Statistical Testing
# ============================================================================

class StatisticalTest:
    """Statistical significance tests with confidence intervals."""

    @staticmethod
    def t_test(
        control: List[float] | np.ndarray,
        treatment: List[float] | np.ndarray,
        alternative: str = "two-sided",
        confidence: float = 0.95,
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Two-sample t-test for difference in means.

        Args:
            control: Control group metric values
            treatment: Treatment group metric values
            alternative: "two-sided", "less", or "greater"
            confidence: Confidence level for interval

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool,
             "ci_lower": float, "ci_upper": float, "effect_size": float}
        """
        control = np.array(control, dtype=float)
        treatment = np.array(treatment, dtype=float)

        n1, n2 = len(control), len(treatment)
        if n1 == 0 or n2 == 0:
            return {
                "statistic": 0.0,
                "pvalue": 1.0,
                "significant": False,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "effect_size": 0.0,
                "mean_diff": 0.0,
            }

        mean1, mean2 = np.mean(control), np.mean(treatment)
        var1 = np.var(control, ddof=1) if n1 > 1 else 0.0
        var2 = np.var(treatment, ddof=1) if n2 > 1 else 0.0

        # Welch's t-test (unequal variances)
        diff = mean2 - mean1
        se = np.sqrt(var1/n1 + var2/n2)

        # 零方差边界场景：两组完全常数时 se=0
        # diff!=0 时应视为显著差异，避免误判 p=1.0
        if se <= 1e-15:
            if abs(diff) <= 1e-15:
                statistic = 0.0
                pvalue = 1.0
                ci_lower = 0.0
                ci_upper = 0.0
                effect_size = 0.0
            else:
                statistic = math.copysign(float("inf"), diff)
                pvalue = 0.0
                epsilon = max(abs(diff) * 1e-12, 1e-12)
                ci_lower = diff - epsilon
                ci_upper = diff + epsilon
                effect_size = math.copysign(float("inf"), diff)

            return {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
                "significant": bool(pvalue < significance_level),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "effect_size": float(effect_size),
                "mean_diff": float(diff),
            }

        # Degrees of freedom (Welch-Satterwaite)
        numerator = (var1 / n1 + var2 / n2) ** 2
        term1 = ((var1 / n1) ** 2) / (n1 - 1) if n1 > 1 and var1 > 0 else 0.0
        term2 = ((var2 / n2) ** 2) / (n2 - 1) if n2 > 1 and var2 > 0 else 0.0
        denominator = term1 + term2
        if denominator > 0:
            df = numerator / denominator
        else:
            df = max(n1 + n2 - 2, 1)
        statistic = diff / se

        try:
            from scipy import stats
            pvalue = stats.t.sf(abs(statistic), df) * 2  # Two-sided
            if alternative == "greater":
                pvalue = stats.t.sf(statistic, df)
            elif alternative == "less":
                pvalue = stats.t.cdf(statistic, df)

            # Confidence interval
            t_crit = stats.t.ppf((1 + confidence) / 2, df)
            ci_lower = diff - t_crit * se
            ci_upper = diff + t_crit * se

            # Cohen's d effect size
            pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            effect_size = diff / pooled_sd if pooled_sd > 0 else 0

        except ImportError:
            # Approximate p-value (assuming normal)
            from math import erfc, sqrt
            if alternative == "greater":
                pvalue = 0.5 * erfc(statistic / sqrt(2))
            elif alternative == "less":
                pvalue = 0.5 * erfc(-statistic / sqrt(2))
            else:
                pvalue = erfc(abs(statistic) / sqrt(2))
            ci_lower = ci_upper = diff
            effect_size = diff / (np.sqrt(var1 + var2) + 1e-10)

        significant = pvalue < significance_level

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": bool(significant),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "effect_size": float(effect_size),
            "mean_diff": float(diff),
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
            {"statistic": float, "pvalue": float, "significant": bool,
             "rank_biserial": float}
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, using simple test")
            return StatisticalTest.t_test(control, treatment, alternative)

        control = np.array(control, dtype=float)
        treatment = np.array(treatment, dtype=float)

        statistic, pvalue = stats.mannwhitneyu(
            treatment, control, alternative=alternative
        )

        # Rank-biserial correlation (effect size)
        n1, n2 = len(control), len(treatment)
        u_stat = min(statistic, n1 * n2 - statistic)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2) if (n1 * n2) > 0 else 0

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": bool(pvalue < 0.05),
            "rank_biserial": float(rank_biserial),
        }

    @staticmethod
    def proportion_test(
        control_successes: int,
        control_total: int,
        treatment_successes: int,
        treatment_total: int,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Z-test for difference in proportions.

        Args:
            control_successes: Successes in control
            control_total: Total in control
            treatment_successes: Successes in treatment
            treatment_total: Total in treatment
            confidence: Confidence level for interval

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool,
             "lift": float, "ci_lower": float, "ci_upper": float}
        """
        if control_total == 0 or treatment_total == 0:
            return {"statistic": 0.0, "pvalue": 1.0, "significant": False,
                    "lift": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        p1 = control_successes / control_total
        p2 = treatment_successes / treatment_total

        # Pooled proportion
        p_pooled = (control_successes + treatment_successes) / (control_total + treatment_total)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))

        if se == 0:
            return {"statistic": 0.0, "pvalue": 1.0, "significant": False,
                    "lift": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

        # Z-statistic
        diff = p2 - p1
        statistic = diff / se

        try:
            from scipy import stats
            pvalue = 2 * (1 - stats.norm.cdf(abs(statistic)))

            # Confidence interval (Wald)
            se_unpooled = np.sqrt(p1*(1-p1)/control_total + p2*(1-p2)/treatment_total)
            z_crit = stats.norm.ppf((1 + confidence) / 2)
            ci_lower = diff - z_crit * se_unpooled
            ci_upper = diff + z_crit * se_unpooled
        except ImportError:
            from math import erfc, sqrt
            pvalue = erfc(abs(statistic) / sqrt(2))
            ci_lower = ci_upper = diff

        # Lift (避免浮点累计误差导致 0.4999999999)
        if p1 > 0 and treatment_total > 0:
            numerator = treatment_successes * control_total - control_successes * treatment_total
            denominator = control_successes * treatment_total
            lift = numerator / denominator if denominator != 0 else 0.0
        else:
            lift = 0.0

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": bool(pvalue < 0.05),
            "lift": float(lift),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    @staticmethod
    def chi_square_test(
        control_successes: int,
        control_failures: int,
        treatment_successes: int,
        treatment_failures: int,
    ) -> Dict[str, Any]:
        """
        Chi-square test for independence.

        Args:
            control_successes: Successes in control
            control_failures: Failures in control
            treatment_successes: Successes in treatment
            treatment_failures: Failures in treatment

        Returns:
            {"statistic": float, "pvalue": float, "significant": bool,
             "cramer_v": float}
        """
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available")
            return {"statistic": 0.0, "pvalue": 1.0, "significant": False, "cramer_v": 0.0}

        # Contingency table
        observed = np.array([
            [control_successes, control_failures],
            [treatment_successes, treatment_failures]
        ])

        # Chi-square test
        statistic, pvalue, dof, expected = stats.chi2_contingency(observed)

        # Cramer's V (effect size)
        n = observed.sum()
        min_dim = min(observed.shape) - 1
        cramer_v = np.sqrt(statistic / (n * min_dim)) if n > 0 and min_dim > 0 else 0

        return {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "significant": bool(pvalue < 0.05),
            "cramer_v": float(cramer_v),
        }

    @staticmethod
    def bootstrap_ci(
        control: List[float] | np.ndarray,
        treatment: List[float] | np.ndarray,
        metric: str = "mean",
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Bootstrap confidence interval for metric difference.

        Args:
            control: Control group metric values
            treatment: Treatment group metric values
            metric: "mean", "median", or custom function
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            {"ci_lower": float, "ci_upper": float, "diff_mean": float}
        """
        control = np.array(control, dtype=float)
        treatment = np.array(treatment, dtype=float)

        if metric == "mean":
            def fn(x): return np.mean(x)
        elif metric == "median":
            def fn(x): return np.median(x)
        else:
            fn = metric

        # Bootstrap
        rng = np.random.default_rng(42)
        diffs = []

        for _ in range(n_bootstrap):
            boot_control = rng.choice(control, size=len(control), replace=True)
            boot_treatment = rng.choice(treatment, size=len(treatment), replace=True)
            diffs.append(fn(boot_treatment) - fn(boot_control))

        diffs = np.array(diffs)
        alpha = 1 - confidence
        ci_lower = np.percentile(diffs, 100 * alpha / 2)
        ci_upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        return {
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "diff_mean": float(np.mean(diffs)),
            "diff_std": float(np.std(diffs)),
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


class ConversionFunnel:
    """
    Conversion funnel analysis for A/B testing.

    Tracks user progression through defined stages:
    - Impressions
    - Clicks
    - Engagement (dwell time threshold)
    - Conversion (booking, purchase, etc.)
    """

    def __init__(
        self,
        stages: List[str] = None,
        dwell_threshold: float = 30.0,
    ):
        """
        Args:
            stages: List of funnel stage names
            dwell_threshold: Seconds for engagement stage
        """
        self.stages = stages or ["impression", "click", "engagement", "conversion"]
        self.dwell_threshold = dwell_threshold

    def analyze_funnel(
        self,
        events: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze conversion funnel from user events.

        Args:
            events: List of event dictionaries with keys:
                - user_id: User identifier
                - item_id: Item identifier
                - clicked: bool (optional)
                - dwell_time: float (optional)
                - converted: bool (optional)

        Returns:
            Funnel metrics including:
                - counts: Users at each stage
                - dropoff: Users lost between stages
                - rates: Conversion rate between stages
                - overall: Overall funnel conversion rate
        """
        # Initialize counters
        funnel_counts = {stage: 0 for stage in self.stages}

        # Track unique users at each stage
        users_seen = set()
        users_clicked = set()
        users_engaged = set()
        users_converted = set()

        for event in events:
            user_id = event.get("user_id")
            if user_id is None:
                continue

            # Impression
            users_seen.add(user_id)
            funnel_counts["impression"] = len(users_seen)

            # Click
            if event.get("clicked", False):
                users_clicked.add(user_id)
                funnel_counts["click"] = len(users_clicked)

            # Engagement (dwell time)
            dwell = event.get("dwell_time", 0)
            if dwell >= self.dwell_threshold:
                users_engaged.add(user_id)
                funnel_counts["engagement"] = len(users_engaged)

            # Conversion
            if event.get("converted", False):
                users_converted.add(user_id)
                funnel_counts["conversion"] = len(users_converted)

        # Calculate dropoff
        dropoff = {}
        prev_count = None
        for stage, count in funnel_counts.items():
            if prev_count is not None:
                dropoff[stage] = max(0, prev_count - count)
            prev_count = count

        # Calculate conversion rates between stages
        rates = {}
        prev_count = None
        for stage, count in funnel_counts.items():
            if prev_count is not None and prev_count > 0:
                rates[f"{stage}_rate"] = count / prev_count
            prev_count = count

        # Overall conversion rate
        overall = {}
        if "impression" in funnel_counts and funnel_counts["impression"] > 0:
            overall["impression_to_conversion"] = (
                funnel_counts.get("conversion", 0) / funnel_counts["impression"]
            )

        return {
            "counts": funnel_counts,
            "dropoff": dropoff,
            "rates": rates,
            "overall": overall,
        }

    def compare_funnels(
        self,
        control_events: List[Dict],
        treatment_events: List[Dict],
    ) -> Dict[str, Any]:
        """
        Compare funnels between control and treatment groups.

        Args:
            control_events: Events from control group
            treatment_events: Events from treatment group

        Returns:
            Comparison with statistical tests for each stage
        """
        control_funnel = self.analyze_funnel(control_events)
        treatment_funnel = self.analyze_funnel(treatment_events)

        comparison = {
            "control": control_funnel,
            "treatment": treatment_funnel,
        }

        # Compare each stage
        for stage in self.stages:
            control_count = control_funnel["counts"].get(stage, 0)
            treatment_count = treatment_funnel["counts"].get(stage, 0)

            # Chi-square test for stage difference
            control_total = control_funnel["counts"].get("impression", 1)
            treatment_total = treatment_funnel["counts"].get("impression", 1)

            control_failures = control_total - control_count
            treatment_failures = treatment_total - treatment_count

            chi_result = StatisticalTest.chi_square_test(
                control_successes=control_count,
                control_failures=max(0, control_failures),
                treatment_successes=treatment_count,
                treatment_failures=max(0, treatment_failures),
            )

            comparison[f"{stage}_test"] = chi_result

        return comparison


class ABMetrics:
    """Compute metrics for A/B testing including CTR, CVR, dwell time."""

    @staticmethod
    def compute_metrics(
        predictions: List[str],
        ground_truth: List[str],
        user_interactions: Optional[List[bool]] = None,
        dwell_times: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Compute metrics for a single query.

        Args:
            predictions: Ordered list of recommended items
            ground_truth: Relevant items
            user_interactions: Whether user actually clicked/visited each item
            dwell_times: Time spent on each item (seconds)

        Returns:
            Dictionary of metric values
        """
        results = {}

        # Recall@K
        for k in [5, 10, 20]:
            pred_k = set(predictions[:k])
            true_set = set(ground_truth)
            results[f"recall@{k}"] = len(pred_k & true_set) / len(true_set) if true_set else 0.0

        # NDCG@K
        for k in [5, 10]:
            relevance = {item: 1.0 for item in ground_truth}
            dcg = 0.0
            for i, item in enumerate(predictions[:k]):
                rel = relevance.get(item, 0.0)
                dcg += (2**rel - 1) / np.log2(i + 2)

            sorted_rel = sorted(relevance.values(), reverse=True)[:k]
            idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(sorted_rel))
            results[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0

        # Hit Rate@K
        for k in [1, 5, 10]:
            pred_k = set(predictions[:k])
            true_set = set(ground_truth)
            results[f"hit_rate@{k}"] = 1.0 if (pred_k & true_set) else 0.0

        # Click-through rate (if interactions available)
        if user_interactions is not None:
            results["ctr"] = float(np.mean(user_interactions))
            results["clicks"] = int(np.sum(user_interactions))

        # Average dwell time (if available)
        if dwell_times is not None and len(dwell_times) > 0:
            results["avg_dwell_time"] = float(np.mean(dwell_times))
            results["total_dwell_time"] = float(np.sum(dwell_times))

        # Coverage
        results["num_recommended"] = len(predictions)

        return results

    @staticmethod
    def calculate_ctr(clicks: int, impressions: int) -> float:
        """Calculate Click-Through Rate."""
        return clicks / impressions if impressions > 0 else 0.0

    @staticmethod
    def calculate_cvr(conversions: int, clicks: int) -> float:
        """Calculate Conversion Rate."""
        return conversions / clicks if clicks > 0 else 0.0

    @staticmethod
    def calculate_dwell_metrics(dwell_times: List[float]) -> Dict[str, float]:
        """Calculate dwell time statistics."""
        if not dwell_times:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}

        return {
            "mean": float(np.mean(dwell_times)),
            "median": float(np.median(dwell_times)),
            "std": float(np.std(dwell_times)),
        }

    @staticmethod
    def calculate_group_metrics(group_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregated metrics for a group.

        Args:
            group_data: List of per-query result dictionaries

        Returns:
            Aggregated metrics dictionary
        """
        if not group_data:
            return {}

        df = pd.DataFrame(group_data)
        metrics = {}

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col == "query_id":
                continue
            values = df[col].dropna()
            if len(values) > 0:
                metrics[f"{col}_mean"] = float(values.mean())
                metrics[f"{col}_std"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
                metrics[f"{col}_median"] = float(values.median())

        # Count metrics
        metrics["total_queries"] = len(group_data)

        return metrics

    @staticmethod
    def confidence_interval(
        values: List[float] | np.ndarray,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for a metric.

        Args:
            values: Sample values
            confidence: Confidence level (0-1)

        Returns:
            (mean, ci_lower, ci_upper)
        """
        values = np.array(values, dtype=float)
        n = len(values)
        if n == 0:
            return 0.0, 0.0, 0.0

        mean = np.mean(values)
        if n == 1:
            # 单样本无法估计方差，退化为零宽置信区间
            return float(mean), float(mean), float(mean)

        std = np.std(values, ddof=1)

        # Standard error
        se = std / np.sqrt(n)

        # Use t-distribution for small samples, normal for large
        try:
            from scipy import stats
            if n < 30:
                t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
            else:
                t_crit = stats.norm.ppf((1 + confidence) / 2)
        except ImportError:
            # Approximate with normal
            from math import sqrt
            t_crit = 1.96 if confidence >= 0.95 else 1.645

        margin = t_crit * se

        return float(mean), float(mean - margin), float(mean + margin)


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
    strategy: str = "hash"
    confidence_level: float = 0.95
    whitelist: Optional[List[str]] = None
    whitelist_group: str = "treatment"


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
            strategy=self.config.strategy,
            whitelist=self.config.whitelist,
            whitelist_group=self.config.whitelist_group,
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
            metrics_val = ABMetrics.compute_metrics(
                recs,
                query.get("ground_truth", []),
                query.get("interactions"),
                query.get("dwell_times"),
            )
            control_results.append({"query_id": qid, **metrics_val})

        for qid in groups["treatment"]:
            query = query_lookup[qid]
            recs = treatment_fn(query)
            metrics_val = ABMetrics.compute_metrics(
                recs,
                query.get("ground_truth", []),
                query.get("interactions"),
                query.get("dwell_times"),
            )
            treatment_results.append({"query_id": qid, **metrics_val})

        # Analyze results
        return ABTestResult(
            control_results=control_results,
            treatment_results=treatment_results,
            metrics=metrics or ["recall@10", "ndcg@10"],
            config=self.config,
        )

    def run_from_logs(
        self,
        control_data: List[Dict],
        treatment_data: List[Dict],
        metrics: List[str],
    ) -> "ABTestResult":
        """
        Run A/B analysis from pre-collected log data.

        Args:
            control_data: List of control group results
            treatment_data: List of treatment group results
            metrics: List of metric names to analyze

        Returns:
            ABTestResult with analysis
        """
        return ABTestResult(
            control_results=control_data,
            treatment_results=treatment_data,
            metrics=metrics,
            config=self.config,
        )


@dataclass
class ABTestResult:
    """Results of an A/B test."""

    control_results: List[Dict]
    treatment_results: List[Dict]
    metrics: List[str]
    config: ABTestConfig
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

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

            if len(control_values) == 0 or len(treatment_values) == 0:
                continue

            # Descriptive stats
            control_mean = float(np.mean(control_values))
            treatment_mean = float(np.mean(treatment_values))
            relative_lift = (treatment_mean - control_mean) / control_mean if control_mean > 0 else 0

            # Confidence intervals
            control_mean, control_ci_lower, control_ci_upper = ABMetrics.confidence_interval(
                control_values, self.config.confidence_level
            )
            treatment_mean, treatment_ci_lower, treatment_ci_upper = ABMetrics.confidence_interval(
                treatment_values, self.config.confidence_level
            )

            analysis[metric] = {
                "control_mean": control_mean,
                "control_std": float(np.std(control_values)),
                "control_median": float(np.median(control_values)),
                "control_ci_lower": control_ci_lower,
                "control_ci_upper": control_ci_upper,
                "control_n": len(control_values),
                "treatment_mean": treatment_mean,
                "treatment_std": float(np.std(treatment_values)),
                "treatment_median": float(np.median(treatment_values)),
                "treatment_ci_lower": treatment_ci_lower,
                "treatment_ci_upper": treatment_ci_upper,
                "treatment_n": len(treatment_values),
                "relative_lift": float(relative_lift),
                "abs_diff": float(treatment_mean - control_mean),
            }

            # Statistical test
            test_result = StatisticalTest.t_test(
                control_values,
                treatment_values,
                confidence=self.config.confidence_level,
                significance_level=self.config.significance_level,
            )
            analysis[metric]["t_test"] = test_result

            # Non-parametric test
            mw_result = StatisticalTest.mann_whitney(control_values, treatment_values)
            analysis[metric]["mann_whitney"] = mw_result

            # Bootstrap CI
            boot_result = StatisticalTest.bootstrap_ci(
                control_values,
                treatment_values,
                confidence=self.config.confidence_level,
            )
            analysis[metric]["bootstrap"] = boot_result

        return analysis

    def summary(self) -> str:
        """Generate human-readable summary."""
        analysis = self.analyze()

        lines = [
            "=" * 80,
            "A/B Test Results Summary",
            "=" * 80,
            f"Timestamp: {self.timestamp}",
            f"Split Ratio: {self.config.split_ratio}",
            f"Control Queries: {len(self.control_results)}",
            f"Treatment Queries: {len(self.treatment_results)}",
            f"Total Queries: {len(self.control_results) + len(self.treatment_results)}",
            f"Significance Level: {self.config.significance_level}",
            f"Confidence Level: {self.config.confidence_level}",
            "",
        ]

        for metric, results in analysis.items():
            lines.append(f"Metric: {metric}")
            lines.append(f"  Control (n={results['control_n']}):")
            lines.append(f"    Mean:   {results['control_mean']:.4f} +/- {results['control_std']:.4f}")
            lines.append(f"    Median: {results['control_median']:.4f}")
            lines.append(f"    95% CI: [{results['control_ci_lower']:.4f}, {results['control_ci_upper']:.4f}]")
            lines.append(f"  Treatment (n={results['treatment_n']}):")
            lines.append(f"    Mean:   {results['treatment_mean']:.4f} +/- {results['treatment_std']:.4f}")
            lines.append(f"    Median: {results['treatment_median']:.4f}")
            lines.append(f"    95% CI: [{results['treatment_ci_lower']:.4f}, {results['treatment_ci_upper']:.4f}]")
            lines.append(f"  Absolute Difference: {results['abs_diff']:.4f}")
            lines.append(f"  Relative Lift: {results['relative_lift']:+.2%}")

            # T-test results
            t_test = results["t_test"]
            sig_indicator = "***" if t_test["significant"] else "ns"
            lines.append(f"  T-test: t={t_test['statistic']:.4f}, p={t_test['pvalue']:.4f} {sig_indicator}")
            lines.append(f"    95% CI: [{t_test['ci_lower']:.4f}, {t_test['ci_upper']:.4f}]")
            lines.append(f"    Effect size (Cohen's d): {t_test['effect_size']:.4f}")

            # Bootstrap results
            boot = results["bootstrap"]
            lines.append(f"  Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

            lines.append("")

        lines.append("=" * 80)
        lines.append("Legend: *** p < 0.05 (significant), ns = not significant")
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as DataFrame."""
        analysis = self.analyze()

        rows = []
        for metric, results in analysis.items():
            rows.append({
                "metric": metric,
                "control_mean": results["control_mean"],
                "control_std": results["control_std"],
                "control_ci_lower": results["control_ci_lower"],
                "control_ci_upper": results["control_ci_upper"],
                "treatment_mean": results["treatment_mean"],
                "treatment_std": results["treatment_std"],
                "treatment_ci_lower": results["treatment_ci_lower"],
                "treatment_ci_upper": results["treatment_ci_upper"],
                "abs_diff": results["abs_diff"],
                "relative_lift": results["relative_lift"],
                "pvalue": results["t_test"]["pvalue"],
                "significant": results["t_test"]["significant"],
                "effect_size": results["t_test"]["effect_size"],
                "ci_lower": results["t_test"]["ci_lower"],
                "ci_upper": results["t_test"]["ci_upper"],
            })

        return pd.DataFrame(rows)

    def generate_report(self, format: str = "markdown") -> str:
        """
        Generate experiment report in specified format.

        Args:
            format: "markdown", "html", or "json"

        Returns:
            Formatted report string
        """
        analysis = self.analyze()

        if format == "json":
            return json.dumps({
                "timestamp": self.timestamp,
                "config": {
                    "split_ratio": self.config.split_ratio,
                    "significance_level": self.config.significance_level,
                    "confidence_level": self.config.confidence_level,
                },
                "results": self._to_serializable(analysis),
            }, indent=2)

        elif format == "html":
            return self._generate_html_report(analysis)

        else:  # markdown
            return self._generate_markdown_report(analysis)

    def _generate_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        lines = [
            "# A/B Test Report",
            "",
            f"**Generated:** {self.timestamp}",
            "",
            "## Experiment Configuration",
            "",
            f"- **Split Ratio:** {self.config.split_ratio}",
            f"- **Significance Level:** {self.config.significance_level}",
            f"- **Confidence Level:** {self.config.confidence_level}",
            f"- **Control Sample Size:** {len(self.control_results)}",
            f"- **Treatment Sample Size:** {len(self.treatment_results)}",
            "",
            "## Results",
            "",
        ]

        # Results table
        lines.append("| Metric | Control Mean | Treatment Mean | Lift | Significant |")
        lines.append("|--------|-------------|----------------|------|------------|")

        for metric, results in analysis.items():
            lift = results["relative_lift"]
            sig = "Yes" if results["t_test"]["significant"] else "No"
            lines.append(
                f"| {metric} | {results['control_mean']:.4f} | "
                f"{results['treatment_mean']:.4f} | {lift:+.1%} | {sig} |"
            )

        lines.append("")

        # Detailed results
        for metric, results in analysis.items():
            lines.append(f"### {metric}")
            lines.append("")
            lines.append("**Control Group:**")
            lines.append(f"- Mean: {results['control_mean']:.4f} +/- {results['control_std']:.4f}")
            lines.append(f"- Median: {results['control_median']:.4f}")
            lines.append(f"- 95% CI: [{results['control_ci_lower']:.4f}, {results['control_ci_upper']:.4f}]")
            lines.append("")
            lines.append("**Treatment Group:**")
            lines.append(f"- Mean: {results['treatment_mean']:.4f} +/- {results['treatment_std']:.4f}")
            lines.append(f"- Median: {results['treatment_median']:.4f}")
            lines.append(f"- 95% CI: [{results['treatment_ci_lower']:.4f}, {results['treatment_ci_upper']:.4f}]")
            lines.append("")
            lines.append("**Statistical Tests:**")
            lines.append(f"- T-test: t={results['t_test']['statistic']:.4f}, p={results['t_test']['pvalue']:.4f}")
            lines.append(f"- Effect size (Cohen's d): {results['t_test']['effect_size']:.4f}")
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>A/B Test Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #555; border-bottom: 1px solid #ddd; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            ".significant { color: green; font-weight: bold; }",
            ".not-significant { color: #888; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>A/B Test Report</h1>",
            f"<p><strong>Generated:</strong> {self.timestamp}</p>",
            "<h2>Experiment Configuration</h2>",
            "<ul>",
            f"<li><strong>Split Ratio:</strong> {self.config.split_ratio}</li>",
            f"<li><strong>Significance Level:</strong> {self.config.significance_level}</li>",
            f"<li><strong>Confidence Level:</strong> {self.config.confidence_level}</li>",
            f"<li><strong>Control Sample Size:</strong> {len(self.control_results)}</li>",
            f"<li><strong>Treatment Sample Size:</strong> {len(self.treatment_results)}</li>",
            "</ul>",
            "<h2>Results</h2>",
            "<table>",
            "<tr><th>Metric</th><th>Control Mean</th><th>Treatment Mean</th><th>Lift</th><th>Significant</th></tr>",
        ]

        for metric, results in analysis.items():
            lift = results["relative_lift"]
            sig_class = "significant" if results["t_test"]["significant"] else "not-significant"
            sig_text = "Yes" if results["t_test"]["significant"] else "No"
            html.append(
                f"<tr><td>{metric}</td>"
                f"<td>{results['control_mean']:.4f}</td>"
                f"<td>{results['treatment_mean']:.4f}</td>"
                f"<td>{lift:+.1%}</td>"
                f"<td class='{sig_class}'>{sig_text}</td></tr>"
            )

        html.extend([
            "</table>",
            "</body>",
            "</html>",
        ])

        return "\n".join(html)

    def _to_serializable(self, value: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        elif isinstance(value, (np.integer, np.floating, np.bool_)):
            return float(value)
        else:
            return value

    def save(self, output_path: str, format: str = "json") -> None:
        """
        Save results to file.

        Args:
            output_path: Path to save results
            format: "json", "markdown", or "html"
        """
        if format == "json":
            analysis = self.analyze()

            # Convert to serializable format
            serializable_results = {
                "timestamp": self.timestamp,
                "config": {
                    "split_ratio": self.config.split_ratio,
                    "salt": self.config.salt,
                    "significance_level": self.config.significance_level,
                    "confidence_level": self.config.confidence_level,
                },
                "sample_sizes": {
                    "control": len(self.control_results),
                    "treatment": len(self.treatment_results),
                },
                "analysis": self._to_serializable(analysis),
            }

            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

        else:
            report = self.generate_report(format=format)
            with open(output_path, 'w') as f:
                f.write(report)

        logger.info(f"Results saved to {output_path}")


# ============================================================================
# Experiment Manager
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    name: str
    description: str = ""
    control_name: str = "Control"
    treatment_name: str = "Treatment"
    split_ratio: float = 0.5
    salt: str = "default"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


class ExperimentManager:
    """Manage multiple A/B experiments."""

    def __init__(self, storage_path: str = "experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.experiments: Dict[str, Any] = {}

    def register(self, config: ExperimentConfig, result: ABTestResult) -> None:
        """Register an experiment result."""
        self.experiments[config.name] = {
            "config": config,
            "result": result,
        }

        # Save to disk
        exp_path = self.storage_path / f"{config.name}.json"
        result.save(str(exp_path))

        # Save config
        config_path = self.storage_path / f"{config.name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "name": config.name,
                "description": config.description,
                "control_name": config.control_name,
                "treatment_name": config.treatment_name,
                "split_ratio": config.split_ratio,
                "salt": config.salt,
                "start_time": config.start_time,
                "end_time": config.end_time,
                "metrics": config.metrics,
                "tags": config.tags,
            }, f, indent=2)

    def load(self, name: str) -> Optional[Dict]:
        """Load an experiment result."""
        exp_path = self.storage_path / f"{name}.json"
        if not exp_path.exists():
            return None

        with open(exp_path) as f:
            return json.load(f)

    def list_experiments(self) -> List[str]:
        """List all registered experiments."""
        return list(self.experiments.keys())

    def compare_experiments(self, names: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        rows = []
        for name in names:
            if name not in self.experiments:
                continue
            result = self.experiments[name]["result"]
            analysis = result.analyze()
            for metric, results in analysis.items():
                rows.append({
                    "experiment": name,
                    "metric": metric,
                    "relative_lift": results["relative_lift"],
                    "pvalue": results["t_test"]["pvalue"],
                    "significant": results["t_test"]["significant"],
                })
        return pd.DataFrame(rows)

    def export_report(
        self,
        name: str,
        output_path: str,
        format: str = "markdown",
    ) -> None:
        """
        Export experiment report.

        Args:
            name: Experiment name
            output_path: Path to save report
            format: "markdown", "html", or "json"
        """
        if name not in self.experiments:
            raise ValueError(f"Experiment {name} not found")

        result = self.experiments[name]["result"]
        result.save(output_path, format=format)

    def get_experiment_summary(self, name: str) -> Dict[str, Any]:
        """Get summary of an experiment."""
        if name not in self.experiments:
            raise ValueError(f"Experiment {name} not found")

        config = self.experiments[name]["config"]
        result = self.experiments[name]["result"]
        analysis = result.analyze()

        summary = {
            "name": config.name,
            "description": config.description,
            "control_name": config.control_name,
            "treatment_name": config.treatment_name,
            "start_time": config.start_time,
            "end_time": config.end_time,
            "sample_sizes": {
                "control": len(result.control_results),
                "treatment": len(result.treatment_results),
            },
            "metrics_summary": {},
        }

        for metric, results in analysis.items():
            summary["metrics_summary"][metric] = {
                "control_mean": results["control_mean"],
                "treatment_mean": results["treatment_mean"],
                "relative_lift": results["relative_lift"],
                "significant": results["t_test"]["significant"],
                "pvalue": results["t_test"]["pvalue"],
            }

        return summary


# ============================================================================
# Online A/B Testing
# ============================================================================

class OnlineABTest:
    """
    Online A/B testing with incremental data collection.

    This class manages ongoing experiments where data arrives incrementally.
    """

    def __init__(
        self,
        config: ABTestConfig,
        max_samples: int = 10000,
    ):
        self.config = config
        self.splitter = TrafficSplitter(
            split_ratio=config.split_ratio,
            salt=config.salt,
            whitelist=config.whitelist,
            whitelist_group=config.whitelist_group,
        )
        self.max_samples = max_samples

        self.control_data: List[Dict] = []
        self.treatment_data: List[Dict] = []
        self.user_assignments: Dict[str, str] = {}

    def assign_user(self, user_id: str) -> str:
        """Assign a user to a group (idempotent)."""
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]

        group = self.splitter.assign_group(user_id)
        self.user_assignments[user_id] = group
        return group

    def record_event(
        self,
        user_id: str,
        event_data: Dict,
    ) -> None:
        """Record an event for a user."""
        group = self.assign_user(user_id)

        record = {"user_id": user_id, **event_data}

        if group == "control":
            self.control_data.append(record)
        else:
            self.treatment_data.append(record)

        # Enforce max samples
        if len(self.control_data) > self.max_samples:
            self.control_data.pop(0)
        if len(self.treatment_data) > self.max_samples:
            self.treatment_data.pop(0)

    def get_current_results(
        self,
        metrics: List[str],
    ) -> ABTestResult:
        """Get current A/B test results."""
        return ABTestResult(
            control_results=self.control_data,
            treatment_results=self.treatment_data,
            metrics=metrics,
            config=self.config,
        )

    def should_stop(self, metric: str, min_effect: float = 0.01) -> Tuple[bool, str]:
        """
        Determine if experiment should stop early.

        Uses sequential testing principles.

        Returns:
            (should_stop, reason)
        """
        if len(self.control_data) < self.config.min_sample_size:
            return False, "Minimum sample size not reached"

        result = self.get_current_results([metric])
        analysis = result.analyze()

        if metric not in analysis:
            return False, "Metric not available"

        metric_results = analysis[metric]

        # Stop if significant with meaningful effect
        if metric_results["t_test"]["significant"]:
            if abs(metric_results["relative_lift"]) >= min_effect:
                return True, f"Significant result: {metric_results['relative_lift']:.2%} lift"

        # Stop if negligible effect (futility)
        ci = metric_results["bootstrap"]
        if abs(ci["ci_lower"]) < min_effect and abs(ci["ci_upper"]) < min_effect:
            return True, "Futility: effect size below threshold"

        return False, "Continue sampling"


# ============================================================================
# Report Templates
# ============================================================================

class ReportTemplate:
    """Templates for generating A/B test reports."""

    @staticmethod
    def executive_summary(result: ABTestResult) -> str:
        """Generate executive summary for stakeholders."""
        analysis = result.analyze()

        lines = [
            "# A/B Test Executive Summary",
            "",
            f"**Date:** {result.timestamp}",
            "",
            "## Key Findings",
            "",
        ]

        # Count significant results
        significant_count = sum(
            1 for m in analysis.values()
            if m.get("t_test", {}).get("significant", False)
        )

        lines.append(f"- **Total Metrics Tested:** {len(analysis)}")
        lines.append(f"- **Significant Improvements:** {significant_count}")
        lines.append("")

        # Top improvements
        improvements = sorted(
            [
                (metric, data)
                for metric, data in analysis.items()
                if data.get("relative_lift", 0) > 0
            ],
            key=lambda x: x[1]["relative_lift"],
            reverse=True
        )[:3]

        if improvements:
            lines.append("## Top Improvements")
            lines.append("")
            for metric, data in improvements:
                lines.append(f"- **{metric}**: +{data['relative_lift']:.1%} lift")
                lines.append(f"  - Control: {data['control_mean']:.4f}")
                lines.append(f"  - Treatment: {data['treatment_mean']:.4f}")
                lines.append(f"  - P-value: {data['t_test']['pvalue']:.4f}")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def technical_report(result: ABTestResult) -> str:
        """Generate detailed technical report."""
        return result.summary()


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B Testing Framework")
    parser.add_argument("--control", required=True, help="Path to control results JSON")
    parser.add_argument("--treatment", required=True, help="Path to treatment results JSON")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--format", default="json", choices=["json", "markdown", "html"],
                       help="Output format")
    parser.add_argument("--plot", help="Path to save plot PNG")
    parser.add_argument("--metrics", nargs="+", default=["recall@10", "ndcg@10"],
                       help="Metrics to analyze")
    args = parser.parse_args()

    # Load results
    with open(args.control) as f:
        control = json.load(f)
    with open(args.treatment) as f:
        treatment = json.load(f)

    # Analyze
    result = ABTestResult(
        control_results=control,
        treatment_results=treatment,
        metrics=args.metrics,
        config=ABTestConfig(),
    )

    # Print summary
    print(result.summary())

    # Save results
    if args.output:
        result.save(args.output, format=args.format)

    # Generate plot (if matplotlib available)
    if args.plot:
        try:
            import matplotlib.pyplot as plt

            analysis = result.analyze()
            fig, ax = plt.subplots(figsize=(10, 6))

            metrics = list(analysis.keys())
            control_means = [analysis[m]["control_mean"] for m in metrics]
            treatment_means = [analysis[m]["treatment_mean"] for m in metrics]

            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width/2, control_means, width, label='Control', alpha=0.8)
            ax.bar(x + width/2, treatment_means, width, label='Treatment', alpha=0.8)

            ax.set_ylabel('Mean Value')
            ax.set_title('A/B Test Results')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()

            plt.tight_layout()
            plt.savefig(args.plot, dpi=300)
            logger.info(f"Plot saved to {args.plot}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")


if __name__ == "__main__":
    main()
