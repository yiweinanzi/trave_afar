#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Experiment Comparison with MLflow for GoAfar

This script demonstrates how to:
1. Run multiple experiments with different configurations
2. Compare results across experiments
3. Find the best performing configuration
4. Visualize comparison results

Usage:
    python examples/experiment_comparison_example.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def simulate_training_run(
    experiment_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int = 5,
) -> Dict[str, float]:
    """
    Simulate a training run with given hyperparameters.

    Returns a dictionary of final metrics.
    """
    print(f"\nRunning experiment: {experiment_name}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")

    from src.utils.experiment import MLflowExperiment

    with MLflowExperiment(experiment_name) as exp:
        # Log parameters
        exp.log_params({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "optimizer": "AdamW",
            "model": "Qwen3-8B",
        })

        # Simulate training
        final_metrics = {}
        for epoch in range(num_epochs):
            # Simulate metrics based on hyperparameters
            # Higher LR = faster convergence but potentially lower final
            # Larger batch = more stable but slower convergence
            np.random.seed(42 + epoch)

            train_loss = 2.0 / (1 + epoch * learning_rate * 1000) + np.random.normal(0, 0.1)
            train_acc = min(0.99, 0.7 + epoch * 0.05 + np.random.normal(0, 0.02))

            val_loss = train_loss + np.random.normal(0, 0.05)
            val_acc = train_acc - np.random.normal(0, 0.03)

            # Log epoch metrics
            exp.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            final_metrics = {
                "final_train_loss": train_loss,
                "final_train_accuracy": train_acc,
                "final_val_loss": val_loss,
                "final_val_accuracy": val_acc,
            }

        # Log recommendation-specific metrics
        recall_at_10 = np.random.uniform(0.6, 0.9)
        ndcg_at_10 = np.random.uniform(0.7, 0.95)

        exp.log_metrics({
            "recall@10": recall_at_10,
            "ndcg@10": ndcg_at_10,
            "final_val_accuracy": final_metrics["final_val_accuracy"],
        })

        return {
            "recall@10": recall_at_10,
            "ndcg@10": ndcg_at_10,
            **final_metrics,
        }


def compare_experiments_cli():
    """Example: Using CLI to compare experiments."""
    print("\n=== Comparing Experiments via CLI ===")

    import subprocess

    # Simulate having run experiments first
    run_ids = []
    for lr in [0.001, 0.0005, 0.0001]:
        for bs in [16, 32]:
            exp_name = f"comparison_lr{lr}_bs{bs}"
            # Would normally get run_id from mlflow
            run_ids.append(exp_name)

    # Use the CLI to compare (if available)
    try:
        result = subprocess.run(
            ["python", "-m", "src.utils.experiment",
             "compare",
             "--experiments"] + run_ids[:3],
            capture_output=True,
            text=True,
            timeout=5
        )
        print("  Comparison command output:")
        print(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  CLI comparison not available, using programmatic method")


def compare_experiments_programmatic():
    """Example: Programmatic experiment comparison."""
    print("\n=== Programmatic Comparison ===")

    from src.utils.experiment import compare_experiments, get_best_run

    # Simulate different experiment configurations
    configs = [
        {"lr": 0.001, "bs": 16},
        {"lr": 0.001, "bs": 32},
        {"lr": 0.0005, "bs": 16},
        {"lr": 0.0005, "bs": 32},
        {"lr": 0.0001, "bs": 16},
    ]

    # Run experiments
    results = {}
    for i, config in enumerate(configs):
        exp_name = f"exp_lr{config['lr']}_bs{config['bs']}"
        print(f"\n[{i+1}/{len(configs)}] Running {exp_name}")
        results[exp_name] = simulate_training_run(
            experiment_name=exp_name,
            learning_rate=config['lr'],
            batch_size=config['bs'],
        )

    # Create comparison DataFrame
    import pandas as pd

    comparison_rows = []
    for exp_name, metrics in results.items():
        lr = float(exp_name.split('_')[0][2:].replace('lr', ''))
        bs = int(exp_name.split('_')[1][2:].replace('bs', ''))

        comparison_rows.append({
            "experiment": exp_name,
            "learning_rate": lr,
            "batch_size": bs,
            "recall@10": metrics["recall@10"],
            "ndcg@10": metrics["ndcg@10"],
            "val_accuracy": metrics["final_val_accuracy"],
        })

    df = pd.DataFrame(comparison_rows)

    print("\n=== Comparison Table ===")
    print(df.to_string(index=False))

    # Find best by each metric
    print("\n=== Best Results ===")

    for metric in ["recall@10", "ndcg@10", "val_accuracy"]:
        if metric in df.columns:
            if "accuracy" in metric:
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmax()

            best_row = df.loc[best_idx]
            print(f"\nBest {metric}:")
            print(f"  Experiment: {best_row['experiment']}")
            print(f"  Value: {best_row[metric]:.4f}")
            print(f"  Config: lr={best_row['learning_rate']}, bs={int(best_row['batch_size'])}")


def visualize_comparison():
    """Example: Visualizing experiment comparison."""
    print("\n=== Visualization Example ===")

    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt

    # Generate comparison data
    learning_rates = [0.0001, 0.0005, 0.001]
    results_by_lr = {
        0.0001: {"recall": 0.75, "ndcg": 0.82},
        0.0005: {"recall": 0.78, "ndcg": 0.85},
        0.001: {"recall": 0.72, "ndcg": 0.80},
    }

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(learning_rates))
    width = 0.35

    # Recall plot
    ax1 = axes[0]
    recalls = [results_by_lr[lr]["recall"] for lr in learning_rates]
    ax1.bar([i - width/2 for i in x], recalls, width,
              label='Recall@10', alpha=0.8)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall by Learning Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(lr) for lr in learning_rates])
    ax1.legend()

    # NDCG plot
    ax2 = axes[1]
    ndcgs = [results_by_lr[lr]["ndcg"] for lr in learning_rates]
    ax2.bar([i + width/2 for i in x], ndcgs, width,
              label='NDCG@10', alpha=0.8, color='orange')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('NDCG@10')
    ax2.set_title('NDCG by Learning Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(lr) for lr in learning_rates])
    ax2.legend()

    plt.tight_layout()

    # Save figure
    output_path = Path("outputs/experiments/comparison_plot.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"  Saved comparison plot to: {output_path}")

    # Log to experiment
    try:
        from src.utils.experiment import MLflowExperiment
        with MLflowExperiment("hyperparameter_comparison") as exp:
            exp.log_figure(fig, "comparison_plot.png")
            exp.log_metrics({
                "best_recall": max(recalls),
                "best_ndcg": max(ndcgs),
            })
    except Exception as e:
        print(f"  Could not log figure: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Experiment Comparison Examples for GoAfar")
    print("=" * 60)

    # Run comparisons
    compare_experiments_programmatic()

    # Create visualizations
    visualize_comparison()

    print("\n" + "=" * 60)
    print("Comparison examples completed!")
    print("=" * 60)
    print("\nTo view experiments in MLflow UI:")
    print("  1. Start MLflow: mlflow ui")
    print("  2. Navigate to: http://localhost:5000")
    print("  3. Compare experiments in the Experiments tab")


if __name__ == "__main__":
    main()
