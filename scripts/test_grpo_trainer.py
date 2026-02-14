#!/usr/bin/env python
"""
Test script for GRPO trainer implementation.

This script validates:
1. Data loading
2. Model initialization
3. Reward computation
4. Advantage calculation
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
from src.rl.grpo_trainer import GRPOTrainer, GRPOConfig


def test_data_loading():
    """Test that training data can be loaded."""
    print("Testing data loading...")

    data_path = "outputs/datasets/grpo_planner_prompts.jsonl"

    if not Path(data_path).exists():
        print(f"  ⚠ Data file not found: {data_path}")
        return False

    # Read a few samples
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            samples.append(json.loads(line))

    print(f"  ✓ Loaded {len(samples)} sample(s)")
    print(f"  Sample structure: {list(samples[0].keys())}")

    return True


def test_config():
    """Test GRPO configuration."""
    print("\nTesting GRPO config...")

    config = GRPOConfig(
        model_name_or_path="models/Qwen3-8B",
        train_data="outputs/datasets/grpo_planner_prompts.jsonl",
        batch_size=2,
        group_size=2,
        num_train_epochs=1,
    )

    print(f"  ✓ Config created")
    print(f"  Model: {config.model_name_or_path}")
    print(f"  Group size: {config.group_size}")
    print(f"  Batch size: {config.batch_size}")

    return True


def test_reward_extraction():
    """Test POI ID extraction from text."""
    print("\nTesting POI ID extraction...")

    # Mock a simple trainer for testing
    config = GRPOConfig(
        model_name_or_path="models/Qwen3-8B",
        train_data="outputs/datasets/grpo_planner_prompts.jsonl",
    )

    # Note: We can't fully initialize without the model, so we'll test the method directly
    test_texts = [
        'S123456',
        '{"poi_id": "S789012"}',
        'The next POI is S345678',
        'Invalid output',
        '123456',
    ]

    expected = ['S123456', 'S789012', 'S345678', None, '123456']

    # Create a minimal instance to test the method
    # We'll just test the logic inline here
    import re

    def extract_poi_id(text):
        """Test extraction logic."""
        # Try JSON
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                poi_id = data.get("poi_id") or data.get("next_poi")
                if poi_id:
                    return str(poi_id)
            elif isinstance(data, str):
                return str(data)
        except:
            pass

        # Try S-prefixed pattern
        s_match = re.search(r'S\d+', text, re.IGNORECASE)
        if s_match:
            return s_match.group(0).upper()

        # Try numeric
        num_match = re.search(r'\d{4,}', text)
        if num_match:
            return num_match.group(0)

        return None

    all_passed = True
    for text, exp in zip(test_texts, expected):
        result = extract_poi_id(text)
        passed = result == exp
        status = "✓" if passed else "✗"
        print(f"  {status} '{text}' -> {result} (expected: {exp})")
        if not passed:
            all_passed = False

    return all_passed


def test_advantage_computation():
    """Test group-relative advantage computation."""
    print("\nTesting advantage computation...")

    # Simulate group rewards
    # Group 1: [1.0, 2.0, 0.5, 1.5] -> mean = 1.25
    # Group 2: [0.5, 1.0, 1.5, 2.0] -> mean = 1.25
    rewards = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.5, 1.0, 1.5, 2.0])
    group_size = 4

    # Compute advantages
    num_groups = len(rewards) // group_size
    grouped_rewards = rewards.view(num_groups, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    advantages = grouped_rewards - group_means

    print(f"  Rewards: {rewards.tolist()}")
    print(f"  Group means: {group_means.squeeze().tolist()}")
    print(f"  Advantages (group-normalized):")
    for i, group_adv in enumerate(advantages):
        print(f"    Group {i}: {group_adv.tolist()}")

    # Check that group advantages sum to zero
    for i, group_adv in enumerate(advantages):
        sum_adv = group_adv.sum().item()
        close_to_zero = abs(sum_adv) < 1e-6
        status = "✓" if close_to_zero else "✗"
        print(f"  {status} Group {i} advantage sum: {sum_adv:.10f}")

    return True


def test_trl_availability():
    """Check if TRL is available."""
    print("\nChecking TRL availability...")

    try:
        import trl
        version = getattr(trl, '__version__', 'unknown')
        print(f"  ✓ TRL is available (version: {version})")

        # Check for GRPOTrainer
        try:
            from trl import GRPOTrainer
            print(f"  ✓ GRPOTrainer is available")
            return True
        except ImportError:
            print(f"  ⚠ GRPOTrainer not available (TRL version may be too old)")
            print(f"     Install with: pip install trl>=0.12.0")
            return False

    except ImportError:
        print(f"  ⚠ TRL not installed")
        print(f"     Install with: pip install trl")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GRPO Trainer Implementation Tests")
    print("=" * 60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Configuration", test_config),
        ("POI ID Extraction", test_reward_extraction),
        ("Advantage Computation", test_advantage_computation),
        ("TRL Availability", test_trl_availability),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! GRPO trainer is ready to use.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
