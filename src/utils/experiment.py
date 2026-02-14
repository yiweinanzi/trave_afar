"""
MLflow Experiment Tracking for GoAfar.

Implements:
- Experiment parameter logging
- Metric tracking
- Model artifact management
- Experiment comparison

Usage:
    with MLflowExperiment("recommendation_test") as experiment:
        experiment.log_params({
            "model": "Qwen3-8B",
            "learning_rate": 1e-5,
        })
        experiment.log_metrics({"recall@10": 0.75, "ndcg@10": 0.82})
        experiment.log_model("model_path")
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Experiment Context
# ============================================================================

class ExperimentTracker:
    """
    Abstract interface for experiment tracking.

    Supports multiple backends:
    - MLflow (preferred)
    - JSON file (fallback)
    - In-memory (testing)
    """

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        raise NotImplementedError

    def log_model(self, model_path: str, name: str = "model") -> None:
        """Log model artifact."""
        raise NotImplementedError

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        """Log arbitrary artifact."""
        raise NotImplementedError

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for the experiment."""
        raise NotImplementedError

    def finish(self) -> None:
        """End the experiment."""
        pass


class MLflowTracker(ExperimentTracker):
    """MLflow-based experiment tracking."""

    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        try:
            import mlflow
        except ImportError:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

        self.experiment_name = experiment_name
        self._mlflow = mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)
        self.run_id = mlflow.start_run().info.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self._mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model_path: str, name: str = "model") -> None:
        self._mlflow.log_artifact(model_path, artifact_path=name)

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        self._mlflow.log_artifact(file_path, artifact_path=name)

    def set_tag(self, key: str, value: str) -> None:
        self._mlflow.set_tag(key, value)

    def finish(self) -> None:
        self._mlflow.end_run()


class JSONTracker(ExperimentTracker):
    """JSON file-based experiment tracking (fallback)."""

    def __init__(self, experiment_name: str, output_dir: str = "outputs/experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = {
            "experiment_name": experiment_name,
            "start_time": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "artifacts": [],
        }

    def log_params(self, params: Dict[str, Any]) -> None:
        self.data["params"].update(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for key, value in metrics.items():
            if step is None:
                self.data["metrics"][key] = value
            else:
                if key not in self.data["metrics"]:
                    self.data["metrics"][key] = []
                self.data["metrics"][key].append({"step": step, "value": value})

    def log_model(self, model_path: str, name: str = "model") -> None:
        self.data["artifacts"].append({"type": "model", "path": model_path, "name": name})

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        self.data["artifacts"].append({
            "type": "artifact",
            "path": file_path,
            "name": name or Path(file_path).name,
        })

    def set_tag(self, key: str, value: str) -> None:
        self.data["tags"][key] = value

    def finish(self) -> None:
        self.data["end_time"] = datetime.now().isoformat()

        output_path = self.output_dir / f"{self.experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_path, "w") as f:
            json.dump(self.data, f, indent=2)

        logger.info(f"Experiment saved to {output_path}")


# ============================================================================
# Experiment Manager
# ============================================================================

class ExperimentManager:
    """
    Factory and manager for experiment tracking.

    Automatically selects backend based on availability.
    """

    def __init__(
        self,
        default_backend: str = "auto",
        mlflow_tracking_uri: Optional[str] = None,
        json_output_dir: str = "outputs/experiments",
    ):
        """
        Args:
            default_backend: "mlflow", "json", or "auto"
            mlflow_tracking_uri: MLflow server URI
            json_output_dir: Directory for JSON experiments
        """
        self.default_backend = default_backend
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.json_output_dir = json_output_dir

        self._check_backends()

    def _check_backends(self) -> None:
        """Check which backends are available."""
        self.has_mlflow = False
        try:
            import mlflow
            self.has_mlflow = True
        except ImportError:
            pass

    def create_experiment(
        self,
        name: str,
        backend: Optional[str] = None,
    ) -> ExperimentTracker:
        """
        Create a new experiment.

        Args:
            name: Experiment name
            backend: Override default backend

        Returns:
            ExperimentTracker instance
        """
        backend = backend or self.default_backend

        if backend == "auto":
            backend = "mlflow" if self.has_mlflow else "json"

        if backend == "mlflow":
            if not self.has_mlflow:
                logger.warning("MLflow not available, falling back to JSON")
                backend = "json"

        if backend == "mlflow":
            return MLflowTracker(name, self.mlflow_tracking_uri)
        elif backend == "json":
            return JSONTracker(name, self.json_output_dir)
        else:
            raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Context Manager for Experiments
# ============================================================================

class MLflowExperiment:
    """
    Context manager for running experiments.

    Usage:
        with MLflowExperiment("my_test") as exp:
            exp.log_params({"lr": 0.001})
            exp.log_metrics({"accuracy": 0.95})
    """

    def __init__(
        self,
        name: str,
        manager: Optional[ExperimentManager] = None,
        **kwargs,
    ):
        self.name = name
        self.manager = manager or ExperimentManager(**kwargs)
        self.experiment = None

    def __enter__(self) -> ExperimentTracker:
        self.experiment = self.manager.create_experiment(self.name)
        return self.experiment

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.experiment:
            self.experiment.finish()


# ============================================================================
# Model Registry Wrapper
# ============================================================================

class ModelRegistry:
    """
    Interface for model versioning and management.

    Compatible with MLflow Model Registry when available.
    """

    def __init__(self, manager: ExperimentManager):
        self.manager = manager

    def register_model(
        self,
        model_path: str,
        name: str,
        version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register a model.

        Args:
            model_path: Path to model files
            name: Model name
            version: Version string (auto-generated if None)
            tags: Tags for the model

        Returns:
            Model version identifier
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_id = f"{name}:{version}"

        # Log to current experiment
        if self.manager.has_mlflow:
            try:
                import mlflow
                mlflow.log_artifacts(model_path, artifact_path=name)
                mlflow.set_tag("registered_model", model_id)

                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(f"{name}.{key}", value)

                logger.info(f"Model registered: {model_id}")
                return model_id
            except Exception as e:
                logger.warning(f"MLflow registration failed: {e}")

        # Fallback: record in local registry
        registry_path = Path("outputs/models") / "registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry = {}
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

        registry[model_id] = {
            "name": name,
            "version": version,
            "path": model_path,
            "tags": tags or {},
            "registered_at": datetime.now().isoformat(),
        }

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info(f"Model registered (local): {model_id}")
        return model_id

    def load_model(self, name: str, version: Optional[str] = None) -> str:
        """
        Get the path for a registered model.

        Args:
            name: Model name
            version: Specific version (latest if None)

        Returns:
            Path to model files
        """
        # Check local registry
        registry_path = Path("outputs/models") / "registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                registry = json.load(f)

            # Find model
            for model_id, info in registry.items():
                if info["name"] == name:
                    if version is None or info["version"] == version:
                        return info["path"]

        raise ValueError(f"Model not found: {name}:{version or 'latest'}")


# ============================================================================
# Experiment Comparison
# ============================================================================

def compare_experiments(
    experiment_ids: List[str],
    metric_names: List[str],
    backend: str = "mlflow",
) -> pd.DataFrame:
    """
    Compare metrics across multiple experiments.

    Args:
        experiment_ids: List of experiment run IDs
        metric_names: Metrics to compare
        backend: "mlflow" or "json"

    Returns:
        DataFrame with comparison results
    """
    if backend == "mlflow":
        try:
            import mlflow
        except ImportError:
            raise ImportError("MLflow not installed")

        results = []

        for exp_id in experiment_ids:
            run = mlflow.get_run(exp_id)
            row = {"experiment_id": exp_id}

            for metric in metric_names:
                row[metric] = run.data.metrics.get(metric)

            results.append(row)

        return pd.DataFrame(results)

    else:
        # JSON backend - scan experiment directory
        exp_dir = Path("outputs/experiments")
        results = []

        for file_path in exp_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)

            row = {"experiment_id": file_path.stem}

            for metric in metric_names:
                value = data.get("metrics", {}).get(metric)
                # Handle both scalar and list metrics
                if isinstance(value, list) and value:
                    value = value[-1]["value"]
                row[metric] = value

            results.append(row)

        return pd.DataFrame(results)


# ============================================================================
# Decorators for Experiment Tracking
# ============================================================================

def track_experiment(
    name: Optional[str] = None,
    params: Optional[Dict] = None,
):
    """
    Decorator to automatically track function execution as an experiment.

    Usage:
        @track_experiment(name="my_experiment", params={"model": "Qwen3-8B"})
        def train_model(config):
            ...
            return {"accuracy": 0.95}

        result = train_model(config)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            exp_name = name or func.__name__

            with MLflowExperiment(exp_name) as exp:
                if params:
                    exp.log_params(params)

                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    exp.log_metrics(metrics)

            return result

        return wrapper
    return decorator


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Experiment Tracking")
    subparsers = parser.add_subparsers(dest="command")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("--experiments", nargs="+", required=True)
    compare_parser.add_argument("--metrics", nargs="+", default=["recall@10", "ndcg@10"])
    compare_parser.add_argument("--output", help="Save comparison to CSV")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--backend", default="mlflow")

    args = parser.parse_args()

    if args.command == "compare":
        df = compare_experiments(args.experiments, args.metrics)
        print(df)

        if args.output:
            df.to_csv(args.output, index=False)

    elif args.command == "list":
        if args.backend == "mlflow":
            import mlflow
            runs = mlflow.search_runs()
            print(runs[["run_id", "experiment_id", "status", "start_time"]])
        else:
            exp_dir = Path("outputs/experiments")
            for path in exp_dir.glob("*.json"):
                print(path.stem)


if __name__ == "__main__":
    main()
