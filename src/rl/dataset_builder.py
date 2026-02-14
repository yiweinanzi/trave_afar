"""
Build GRPO/SFT-ready datasets from synthesized planner data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


def _iter_jsonl(path: str | Path) -> Iterable[Dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_grpo_prompt_samples(
    rl_prompts_path: str,
    output_path: str,
    max_samples: int = 50000,
) -> int:
    """
    Convert planner RL prompts into text-format samples for LLM RL.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as fw:
        for obj in _iter_jsonl(rl_prompts_path):
            prompt = {
                "task": "plan_next_poi",
                "user_id": obj.get("user_id"),
                "day": obj.get("day"),
                "state_prefix": obj.get("state_prefix", []),
                "instruction": "请基于当前已选景点，生成下一步最合理的 poi_id，并保证时间窗可行。",
            }
            sample = {
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "target_next_poi": obj.get("target_next_poi"),
                "full_target_route": obj.get("full_target_route", []),
            }
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break
    return count


def build_sft_samples(
    trajectories_path: str,
    output_path: str,
    max_samples: int = 50000,
) -> int:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as fw:
        for obj in _iter_jsonl(trajectories_path):
            prompt = {
                "task": "plan_route",
                "user_id": obj.get("user_id"),
                "day": obj.get("day"),
                "instruction": "请输出可执行的 POI 序列（JSON list），优先满足偏好并减少绕路。",
            }
            sample = {
                "prompt": json.dumps(prompt, ensure_ascii=False),
                "completion": json.dumps(obj.get("poi_sequence", []), ensure_ascii=False),
            }
            fw.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_samples:
                break
    return count


def main():
    parser = argparse.ArgumentParser(description="Build RL/SFT dataset for planner GRPO")
    parser.add_argument("--rl-prompts", default="outputs/datasets/planner_rl_prompts.jsonl")
    parser.add_argument("--trajectories", default="outputs/datasets/planner_trajectories.jsonl")
    parser.add_argument("--out-rl", default="outputs/datasets/grpo_planner_prompts.jsonl")
    parser.add_argument("--out-sft", default="outputs/datasets/sft_planner_samples.jsonl")
    parser.add_argument("--max-samples", type=int, default=50000)
    args = parser.parse_args()

    rl_n = build_grpo_prompt_samples(args.rl_prompts, args.out_rl, max_samples=args.max_samples)
    sft_n = build_sft_samples(args.trajectories, args.out_sft, max_samples=args.max_samples)
    print(json.dumps({"grpo_samples": rl_n, "sft_samples": sft_n}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
