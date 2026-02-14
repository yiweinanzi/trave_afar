"""
Build synthetic SFT/DPO/RL training data from user_events.csv.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.id_mapping import normalize_poi_id


ACTION_WEIGHT = {"click": 1, "fav": 2, "visit": 3}


def load_base_data(poi_csv: str, events_csv: str):
    poi_df = pd.read_csv(poi_csv).copy()
    events_df = pd.read_csv(events_csv).copy()

    poi_df["poi_id"] = poi_df["poi_id"].apply(normalize_poi_id)
    events_df["poi_id"] = events_df["poi_id"].apply(normalize_poi_id)
    events_df = events_df[events_df["poi_id"].isin(set(poi_df["poi_id"]))]
    events_df["event_time"] = pd.to_datetime(events_df["timestamp"], unit="s", errors="coerce")
    events_df = events_df.dropna(subset=["event_time"]).copy()
    return poi_df, events_df


def build_trajectories(events_df: pd.DataFrame, min_len: int = 3) -> List[Dict]:
    events_df = events_df.copy()
    events_df["day"] = events_df["event_time"].dt.date.astype(str)

    trajectories = []
    grouped = events_df.sort_values("event_time").groupby(["user_id", "day"])
    for (user_id, day), group in grouped:
        if len(group) < min_len:
            continue
        seq = group["poi_id"].tolist()
        actions = group["action"].tolist()
        weights = [ACTION_WEIGHT.get(a, 1) for a in actions]
        trajectories.append(
            {
                "user_id": str(user_id),
                "day": day,
                "poi_sequence": seq,
                "actions": actions,
                "event_count": len(seq),
                "engagement_score": float(sum(weights) / len(weights)),
            }
        )
    return trajectories


def synthesize_events(
    events_df: pd.DataFrame,
    poi_ids: List[str],
    target_users: int = 300,
    max_actions: int = 12,
) -> pd.DataFrame:
    """
    Generate extra synthetic interactions for cold-start users.
    """
    if target_users <= 0:
        return events_df

    rng = random.Random(2026)
    newest_ts = int(events_df["timestamp"].max())
    rows = []
    for idx in range(target_users):
        user_id = f"SYNTH_{idx:04d}"
        length = rng.randint(3, max_actions)
        sampled = rng.sample(poi_ids, k=min(length, len(poi_ids)))
        base_ts = newest_ts - rng.randint(1, 90) * 86400
        for step, poi_id in enumerate(sampled):
            action = rng.choices(["click", "fav", "visit"], weights=[0.55, 0.25, 0.20])[0]
            rows.append(
                {
                    "user_id": user_id,
                    "poi_id": poi_id,
                    "timestamp": base_ts + step * rng.randint(1200, 7200),
                    "action": action,
                }
            )
    if not rows:
        return events_df
    synth_df = pd.DataFrame(rows)
    out = pd.concat([events_df[["user_id", "poi_id", "timestamp", "action"]], synth_df], ignore_index=True)
    return out


def _make_rejected_path(chosen: List[str], universe: List[str]) -> List[str]:
    if len(chosen) < 2:
        return chosen
    rng = random.Random()
    rejected = chosen[:]
    rng.shuffle(rejected)
    replace_idx = rng.randrange(len(rejected))
    candidates = [x for x in universe if x not in chosen]
    if candidates:
        rejected[replace_idx] = rng.choice(candidates)
    return rejected


def build_preference_pairs(trajectories: List[Dict], universe: List[str], max_pairs: int = 20000) -> List[Dict]:
    pairs = []
    for traj in trajectories:
        chosen = traj["poi_sequence"]
        if len(chosen) < 3:
            continue
        rejected = _make_rejected_path(chosen, universe)
        pairs.append(
            {
                "user_id": traj["user_id"],
                "day": traj["day"],
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "event_count": traj["event_count"],
                    "engagement_score": traj["engagement_score"],
                },
            }
        )
        if len(pairs) >= max_pairs:
            break
    return pairs


def build_rl_prompts(trajectories: List[Dict], max_samples: int = 40000) -> List[Dict]:
    prompts = []
    for traj in trajectories:
        seq = traj["poi_sequence"]
        for cut in range(2, len(seq)):
            prompts.append(
                {
                    "user_id": traj["user_id"],
                    "day": traj["day"],
                    "state_prefix": seq[:cut],
                    "target_next_poi": seq[cut],
                    "full_target_route": seq,
                }
            )
            if len(prompts) >= max_samples:
                return prompts
    return prompts


def _save_jsonl(items: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run(
    poi_csv: str,
    events_csv: str,
    output_dir: str,
    synth_users: int = 0,
):
    poi_df, events_df = load_base_data(poi_csv, events_csv)
    if synth_users > 0:
        events_df = synthesize_events(events_df, poi_df["poi_id"].tolist(), target_users=synth_users)
        events_df["event_time"] = pd.to_datetime(events_df["timestamp"], unit="s", errors="coerce")

    trajectories = build_trajectories(events_df)
    preference_pairs = build_preference_pairs(trajectories, universe=poi_df["poi_id"].tolist())
    rl_prompts = build_rl_prompts(trajectories)

    out_dir = Path(output_dir)
    _save_jsonl(trajectories, out_dir / "planner_trajectories.jsonl")
    _save_jsonl(preference_pairs, out_dir / "planner_preference_pairs.jsonl")
    _save_jsonl(rl_prompts, out_dir / "planner_rl_prompts.jsonl")

    summary = {
        "poi_count": int(len(poi_df)),
        "event_count": int(len(events_df)),
        "trajectory_count": int(len(trajectories)),
        "preference_pair_count": int(len(preference_pairs)),
        "rl_prompt_count": int(len(rl_prompts)),
        "output_dir": str(out_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Synthesize GoAfar training data")
    parser.add_argument("--poi-csv", default="data/all/poi_expanded.csv")
    parser.add_argument("--events-csv", default="data/all/user_events.csv")
    parser.add_argument("--output-dir", default="outputs/datasets")
    parser.add_argument("--synth-users", type=int, default=300, help="number of synthetic users to append")
    args = parser.parse_args()

    run(
        poi_csv=args.poi_csv,
        events_csv=args.events_csv,
        output_dir=args.output_dir,
        synth_users=args.synth_users,
    )


if __name__ == "__main__":
    main()
