#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: MLflow Experiment Tracking for GoAfar

This script demonstrates various ways to use experiment tracking in GoAfar:

1. Basic tracking with context manager
2. Training script integration
3. Evaluation with metrics
4. Model registration
5. Experiment comparison

Usage:
    python examples/experiment_tracking_example.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def example_basic_tracking():
    """Example 1: Basic experiment tracking with context manager."""
    print("\n=== Example 1: Basic Tracking ===")

    from src.utils.experiment import MLflowExperiment

    with MLflowExperiment("basic_example") as exp:
        # Log parameters
        exp.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "model": "Qwen3-8B",
            "optimizer": "AdamW",
        })

        # Simulate training
        for step in range(0, 100, 10):
            loss = 1.0 - step * 0.01
            accuracy = step * 0.009

            # Log metrics with step
            exp.log_metrics({
                "loss": loss,
                "accuracy": accuracy,
            }, step=step)

        # Log final metrics
        exp.log_metrics({
            "final_loss": 0.1,
            "final_accuracy": 0.95,
        })

        # Log artifacts
        config_path = Path("configs/runtime.yaml")
        if config_path.exists():
            exp.log_artifact(str(config_path))

    print("  Basic tracking example completed!")


def example_decorator_tracking():
    """Example 2: Using decorator for automatic tracking."""
    print("\n=== Example 2: Decorator Tracking ===")

    from src.utils.experiment import track_experiment

    @track_experiment(
        name="decorator_example",
        params={"model": "Qwen3-8B", "epochs": 3}
    )
    def train_model(learning_rate: float, batch_size: int):
        """Simulated training function."""
        # Simulate training
        results = {}
        for step in range(10):
            loss = np.random.uniform(0.1, 1.0)
            results[f"step_{step}_loss"] = loss

        # Return metrics (automatically logged)
        return {
            "final_loss": np.mean(list(results.values())),
            "min_loss": min(results.values()),
        }

    result = train_model(learning_rate=0.001, batch_size=16)
    print(f"  Training result: {result}")


def example_evaluation_tracking():
    """Example 3: Tracking evaluation metrics."""
    print("\n=== Example 3: Evaluation Tracking ===")

    from src.utils.experiment import MLflowExperiment
    from src.evaluation.metrics_advanced import (
        recall_at_k, ndcg_at_k, precision_at_k,
        diversity_score, coverage
    )

    with MLflowExperiment("evaluation_example") as exp:
        # Simulate predictions and ground truth
        predictions = [
            ["poi1", "poi2", "poi3", "poi4", "poi5"],
            ["poi2", "poi3", "poi6", "poi7", "poi8"],
            ["poi1", "poi5", "poi9", "poi10", "poi11"],
        ]
        ground_truth = [
            ["poi2", "poi4"],
            ["poi3", "poi6"],
            ["poi1", "poi9"],
        ]

        # Compute metrics for different K values
        all_metrics = {}
        for k in [5, 10]:
            recalls = [recall_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            ndcgs = [ndcg_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]
            precisions = [precision_at_k(pred, truth, k) for pred, truth in zip(predictions, ground_truth)]

            all_metrics[f"recall@{k}"] = np.mean(recalls)
            all_metrics[f"ndcg@{k}"] = np.mean(ndcgs)
            all_metrics[f"precision@{k}"] = np.mean(precisions)

        # Log metrics
        exp.log_metrics(all_metrics)

        # Add item attributes for diversity
        item_attributes = {
            f"poi{i}": {"category": f"cat{i % 3}"}
            for i in range(1, 12)
        }

        diversity = diversity_score(predictions, item_attributes)
        exp.log_metrics({"diversity": diversity})

    print("  Evaluation tracking completed!")


def example_model_registration():
    """Example 4: Model registration."""
    print("\n=== Example 4: Model Registration ===")

    from src.utils.experiment import ExperimentManager, ModelRegistry

    manager = ExperimentManager()
    registry = ModelRegistry(manager)

    # Register a model
    model_id = registry.register_model(
        model_path="outputs/sft/qwen3-8b-tourism",
        name="tourism_sft",
        version="v1.0",
        tags={
            "task": "recommendation",
            "base_model": "Qwen3-8B",
            "training_data": "tourism_routes",
        },
        model_type="huggingface"
    )

    print(f"  Registered model: {model_id}")


def example_config_loading():
    """Example 5: Loading configuration from YAML."""
    print("\n=== Example 5: Config Loading ===")

    from src.utils.experiment import ExperimentConfig, ExperimentManager

    # Load from config file
    config = ExperimentConfig.from_yaml("configs/runtime.yaml")

    print(f"  MLflow enabled: {config.enabled}")
    print(f"  Backend: {config.backend}")
    print(f"  Tags: {config.tags}")

    # Create manager with config
    manager = ExperimentManager(
        config=config,
    )

    # List available experiments
    exp = manager.create_experiment("config_test")


def example_callback_integration():
    """Example 6: Using MLflow callback with Transformers."""
    print("\n=== Example 6: Callback Integration ===")

    from src.utils.experiment import MLflowCallback

    # Create callback
    callback = MLflowCallback(
        experiment_name="callback_example",
        log_model=True,
        log_artifacts=True,
    )

    print(f"  Created callback: {callback.__class__.__name__}")
    print("  Usage: Add to Trainer callbacks=[callback]")


def main():
    """Run all examples."""
    print("=" * 60)
    print("MLflow Experiment Tracking Examples for GoAfar")
    print("=" * 60)

    try:
        example_basic_tracking()
    except Exception as e:
        print(f"  Error in basic tracking: {e}")

    try:
        example_decorator_tracking()
    except Exception as e:
        print(f"  Error in decorator tracking: {e}")

    try:
        example_evaluation_tracking()
    except Exception as e:
        print(f"  Error in evaluation tracking: {e}")

    try:
        example_model_registration()
    except Exception as e:
        print(f"  Error in model registration: {e}")

    try:
        example_config_loading()
    except Exception as e:
        print(f"  Error in config loading: {e}")

    try:
        example_callback_integration()
    except Exception as e:
        print(f"  Error in callback integration: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
