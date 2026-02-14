#!/usr/bin/env python
"""
Example: Using a GRPO-trained model for route planning inference.

This script demonstrates how to:
1. Load a GRPO-trained model
2. Generate next POI recommendations
3. Integrate with the GoAfar pipeline
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class GRPOPlanner:
    """
    Route planning agent using GRPO-trained model.

    This class provides a simple interface for generating POI recommendations
    using a GRPO-trained policy model.
    """

    def __init__(
        self,
        base_model_path: str = "models/Qwen3-8B",
        adapter_path: str = "outputs/grpo/qwen3-grpo-planner",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the GRPO planner.

        Args:
            base_model_path: Path to base model (Qwen3-8B)
            adapter_path: Path to GRPO-trained LoRA adapters
            device: Device to run inference on
        """
        self.device = device

        print(f"Loading base model from {base_model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        print(f"Loading GRPO adapters from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model.eval()

        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("✓ GRPO planner initialized!")

    def recommend_next_poi(
        self,
        visited_pois: List[str],
        day: int = 1,
        interests: Optional[List[str]] = None,
        province: str = "Unknown",
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Recommend the next POI to visit.

        Args:
            visited_pois: List of already visited POI IDs
            day: Current day of the trip
            interests: User interests (optional)
            province: Province name (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Recommended POI ID (e.g., "S123456")
        """
        # Format prompt
        prompt = self._format_prompt(
            visited_pois=visited_pois,
            day=day,
            interests=interests,
            province=province,
        )

        # Generate
        messages = [
            {"role": "system", "content": "你是一位专业的旅游规划助手。"},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Extract POI ID
        poi_id = self._extract_poi_id(response)

        return poi_id

    def _format_prompt(
        self,
        visited_pois: List[str],
        day: int,
        interests: Optional[List[str]],
        province: str,
    ) -> str:
        """Format the user prompt."""
        parts = []

        if province and province != "Unknown":
            parts.append(f"目的地：{province}")

        parts.append(f"行程天数：第{day}天")

        if interests:
            parts.append(f"兴趣偏好：{', '.join(interests)}")

        if visited_pois:
            # Show last 3 visited POIs
            recent = visited_pois[-3:]
            parts.append(f"已选景点：{', '.join(recent)}")

        parts.append("\n请推荐下一个最合适的景点。")

        return "\n".join(parts)

    def _extract_poi_id(self, text: str) -> str:
        """Extract POI ID from generated text."""
        import json
        import re

        # Try JSON
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data.get("poi_id") or data.get("next_poi", "")
            elif isinstance(data, str):
                return data
        except:
            pass

        # Try S-prefixed pattern
        match = re.search(r'S\d+', text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

        # Try numeric
        match = re.search(r'\d{4,}', text)
        if match:
            return match.group(0)

        return ""

    def plan_route(
        self,
        start_poi: str,
        num_pois: int = 5,
        interests: Optional[List[str]] = None,
        province: str = "Unknown",
    ) -> List[str]:
        """
        Plan a complete route by iteratively recommending POIs.

        Args:
            start_poi: Starting POI ID
            num_pois: Total number of POIs to visit
            interests: User interests
            province: Province name

        Returns:
            List of POI IDs in visitation order
        """
        route = [start_poi]

        for i in range(num_pois - 1):
            print(f"Planning step {i+1}/{num_pois-1}...")

            next_poi = self.recommend_next_poi(
                visited_pois=route,
                day=1,
                interests=interests,
                province=province,
            )

            if not next_poi:
                print("⚠ Model did not generate a valid POI ID")
                break

            if next_poi in route:
                print(f"⚠ Model recommended already visited POI: {next_poi}")
                # Could implement fallback logic here

            route.append(next_poi)
            print(f"  → {next_poi}")

        return route


def main():
    """Example usage of the GRPO planner."""
    print("=" * 60)
    print("GRPO Route Planning Inference Example")
    print("=" * 60)

    # Check if model exists
    adapter_path = "outputs/grpo/qwen3-grpo-planner"
    base_model_path = "models/Qwen3-8B"

    if not Path(adapter_path).exists():
        print(f"\n⚠ Trained model not found at {adapter_path}")
        print("Please train the model first:")
        print("  bash scripts/train_grpo.sh")
        return

    # Initialize planner
    print("\nInitializing GRPO planner...")
    planner = GRPOPlanner(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )

    # Example 1: Single POI recommendation
    print("\n" + "=" * 60)
    print("Example 1: Single POI Recommendation")
    print("=" * 60)

    visited = ["S094645", "S151634"]
    print(f"\nCurrent route: {visited}")
    print("Interests: 自然风光, 历史文化")

    next_poi = planner.recommend_next_poi(
        visited_pois=visited,
        day=1,
        interests=["自然风光", "历史文化"],
        province="四川省",
    )

    print(f"\nRecommended next POI: {next_poi}")

    # Example 2: Complete route planning
    print("\n" + "=" * 60)
    print("Example 2: Complete Route Planning")
    print("=" * 60)

    start_poi = "S094645"
    num_pois = 4

    print(f"\nStarting POI: {start_poi}")
    print(f"Target POIs: {num_pois}")
    print(f"Interests: 自然风光")

    route = planner.plan_route(
        start_poi=start_poi,
        num_pois=num_pois,
        interests=["自然风光"],
        province="四川省",
    )

    print(f"\nPlanned route: {' → '.join(route)}")

    print("\n" + "=" * 60)
    print("Inference examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
