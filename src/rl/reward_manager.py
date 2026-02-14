"""
Reward manager for route planning RL/GRPO.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class RewardWeights:
    pref_weight: float = 1.0
    feasible_penalty: float = 1.5
    overtime_penalty: float = 1.2
    travel_penalty: float = 0.003
    diversity_weight: float = 0.3


class RewardManager:
    """
    Compute route-level reward as a weighted sum of:
    - preference score
    - feasibility constraints (time windows)
    - travel cost penalty
    - category diversity bonus
    """

    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    @staticmethod
    def _build_index(df: pd.DataFrame) -> Dict[str, int]:
        return {str(pid): idx for idx, pid in enumerate(df["poi_id"].astype(str).tolist())}

    @staticmethod
    def _diversity_bonus(categories: Iterable[str]) -> float:
        cats = [c for c in categories if c]
        if not cats:
            return 0.0
        count = {}
        for c in cats:
            count[c] = count.get(c, 0) + 1
        total = float(sum(count.values()))
        probs = [v / total for v in count.values()]
        entropy = -sum(p * math.log(p + 1e-8) for p in probs)
        max_entropy = math.log(len(count) + 1e-8)
        if max_entropy <= 0:
            return 0.0
        return float(entropy / max_entropy)

    def score_route(
        self,
        route_poi_ids: List[str],
        poi_df: pd.DataFrame,
        time_matrix: np.ndarray,
        preference_scores: Optional[Mapping[str, float]] = None,
        start_time_min: int = 480,
        end_time_min: int = 1320,
    ) -> Dict[str, float]:
        if len(route_poi_ids) <= 1:
            return {"total_reward": -1.0, "pref_reward": 0.0, "feasible": 0.0, "travel_penalty": 0.0, "diversity": 0.0}

        idx_map = self._build_index(poi_df)
        pref = preference_scores or {}

        total_pref = 0.0
        total_travel_min = 0.0
        feasible = 1.0
        overtime = 0.0
        categories = []

        current_time = float(start_time_min)
        for i in range(1, len(route_poi_ids)):
            prev_id = str(route_poi_ids[i - 1])
            cur_id = str(route_poi_ids[i])
            if prev_id not in idx_map or cur_id not in idx_map:
                feasible = 0.0
                continue

            prev_idx = idx_map[prev_id]
            cur_idx = idx_map[cur_id]
            travel_min = float(time_matrix[prev_idx, cur_idx]) / 60.0
            total_travel_min += travel_min
            current_time += travel_min

            row = poi_df.iloc[cur_idx]
            open_min = float(row.get("open_min", 0))
            close_min = float(row.get("close_min", 1440))
            stay_min = float(row.get("stay_min", 60))

            if current_time < open_min:
                current_time = open_min
            if current_time > close_min:
                feasible = 0.0

            current_time += stay_min
            total_pref += float(pref.get(cur_id, 0.0))
            categories.append(str(row.get("category", "")))

        if current_time > end_time_min:
            overtime = (current_time - end_time_min) / 60.0

        diversity = self._diversity_bonus(categories)

        pref_reward = self.weights.pref_weight * total_pref
        feasible_penalty = self.weights.feasible_penalty * (1.0 - feasible)
        overtime_penalty = self.weights.overtime_penalty * overtime
        travel_penalty = self.weights.travel_penalty * total_travel_min
        diversity_bonus = self.weights.diversity_weight * diversity

        total_reward = pref_reward + diversity_bonus - feasible_penalty - overtime_penalty - travel_penalty
        return {
            "total_reward": float(total_reward),
            "pref_reward": float(pref_reward),
            "feasible": float(feasible),
            "overtime_penalty": float(overtime_penalty),
            "travel_penalty": float(travel_penalty),
            "diversity": float(diversity_bonus),
        }

