"""
MLflow Experiment Tracking for GoAfar.

Implements:
- Experiment parameter logging
- Metric tracking
- Model artifact management
- Experiment comparison
- Automatic MLflow server detection
- Model registry support

Usage:
    from src.utils.experiment import ExperimentTracker

    with ExperimentTracker("sft_tourism", tracking_uri="http://localhost:5000") as exp:
        exp.log_params({
            "model": "Qwen3-8B",
            "learning_rate": 1e-5,
        })
        exp.log_metrics({"recall@10": 0.75, "ndcg@10": 0.82}, step=100)
        exp.log_model("outputs/sft/qwen3-8b-tourism")
        exp.log_artifact("configs/training.yaml")

Or using the context manager with auto-cleanup:

    with MLflowExperiment("recommendation_test") as experiment:
        experiment.log_params({"model": "Qwen3-8B"})
        experiment.log_metrics({"recall@10": 0.75})
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    enabled: bool = True
    backend: str = "auto"  # "auto", "mlflow", "json", "none"
    tracking_uri: Optional[str] = None  # MLflow server URI
    experiment_name: Optional[str] = None  # Default experiment name
    json_output_dir: str = "outputs/experiments"
    artifact_location: Optional[str] = None  # Custom artifact location
    auto_flush: bool = True  # Automatically flush metrics after logging
    tags: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, config_path: str = "configs/runtime.yaml") -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            mlflow_config = config_data.get("mlflow", {})
            return cls(**mlflow_config)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return cls()

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("MLFLOW_ENABLED", "true").lower() == "true",
            backend=os.getenv("MLFLOW_BACKEND", "auto"),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME"),
            json_output_dir=os.getenv("MLFLOW_JSON_DIR", "outputs/experiments"),
        )


# ============================================================================
# Experiment Context
# ============================================================================

class ExperimentTracker:
    """
    Abstract interface for experiment tracking.

    Supports multiple backends:
    - MLflow (preferred) - Full-featured experiment tracking
    - JSON file (fallback) - Simple file-based tracking
    - In-memory (testing) - No persistence
    - None - Disabled tracking
    """

    def __init__(self, experiment_name: str, config: Optional[ExperimentConfig] = None):
        """
        Initialize the experiment tracker.

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
        """
        self.experiment_name = experiment_name
        self.config = config or ExperimentConfig()
        self._is_active = False

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        raise NotImplementedError

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        raise NotImplementedError

    def log_model(
        self,
        model_path: str,
        name: str = "model",
        model_type: str = "pytorch",
        **kwargs
    ) -> None:
        """Log model artifact."""
        raise NotImplementedError

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        """Log arbitrary artifact."""
        raise NotImplementedError

    def log_artifacts(self, artifact_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log all artifacts in a directory."""
        raise NotImplementedError

    def log_text(self, text: str, filename: str) -> None:
        """Log text as an artifact file."""
        raise NotImplementedError

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log dictionary as JSON artifact."""
        raise NotImplementedError

    def log_figure(self, figure, filename: str) -> None:
        """Log matplotlib figure."""
        raise NotImplementedError

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag for the experiment."""
        raise NotImplementedError

    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags."""
        for key, value in tags.items():
            self.set_tag(key, value)

    def log_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset metadata in a backend-agnostic way."""
        self.set_tag("dataset_info", json.dumps(dataset_info, ensure_ascii=False))

    def log_training_progress(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training progress with epoch and step."""
        self.set_tag("current_epoch", str(epoch))
        self.log_metrics(metrics, step=step)

    def finish(self) -> None:
        """End the experiment."""
        pass

    def __enter__(self) -> "ExperimentTracker":
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()
        self._is_active = False


class MLflowTracker(ExperimentTracker):
    """MLflow-based experiment tracking with full feature support."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        config: Optional[ExperimentConfig] = None,
    ):
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "MLflow not installed. Install with: pip install mlflow"
            )

        super().__init__(experiment_name, config)

        self.experiment_name = experiment_name
        self._mlflow = mlflow
        self._run_id: Optional[str] = None
        self._active_run = None

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        elif self.config and self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)

        # Set experiment
        mlflow.set_experiment(experiment_name)

    def start_run(self) -> str:
        """Start a new MLflow run and return run ID."""
        if self._active_run is None:
            self._active_run = self._mlflow.start_run()
            self._run_id = self._active_run.info.run_id

            # Set default tags
            if self.config and self.config.tags:
                self.set_tags(self.config.tags)

        return self._run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        # Convert complex types to strings
        for key, value in params.items():
            if isinstance(value, (list, dict, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            self._mlflow.log_param(key, str(value))

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        self._mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model_path: str,
        name: str = "model",
        model_type: str = "pytorch",
        **kwargs
    ) -> None:
        """Log model to MLflow."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        model_path_obj = Path(model_path)

        # HuggingFace 模型优先使用 transformers flavor；失败时降级为普通 artifact。
        if model_type == "huggingface":
            try:
                self._mlflow.transformers.log_model(
                    transformers_model=str(model_path_obj),
                    artifact_path=name,
                    **kwargs,
                )
                return
            except Exception as exc:
                logger.warning(f"Failed to log huggingface model via mlflow transformers flavor: {exc}")

        # PyTorch flavor 需要 torch.nn.Module 对象；仅当显式提供时使用。
        if model_type == "pytorch":
            pytorch_model = kwargs.pop("pytorch_model", None)
            if pytorch_model is not None:
                self._mlflow.pytorch.log_model(
                    pytorch_model=pytorch_model,
                    artifact_path=name,
                    **kwargs,
                )
                return

        # 通用回退：路径存在时按 artifact 目录/文件上传，不再误调用 pytorch flavor。
        if not model_path_obj.exists():
            logger.warning(f"Model path does not exist, skip logging: {model_path_obj}")
            return
        if model_path_obj.is_dir():
            self.log_artifacts(str(model_path_obj), artifact_path=name)
        else:
            self.log_artifact(str(model_path_obj), name)

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        """Log single artifact to MLflow."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        self._mlflow.log_artifact(file_path, artifact_path=name)

    def log_artifacts(self, artifact_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log all artifacts in a directory."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        self._mlflow.log_artifacts(artifact_dir, artifact_path=artifact_path)

    def log_text(self, text: str, filename: str) -> None:
        """Log text as an artifact file."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        self._mlflow.log_text(text, artifact_file=filename)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log dictionary as JSON artifact."""
        self.log_text(json.dumps(data, indent=2, ensure_ascii=False), filename)

    def log_figure(self, figure, filename: str) -> None:
        """Log matplotlib figure."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        try:
            import io
            import matplotlib.pyplot as plt

            buf = io.BytesIO()
            figure.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Save to temp file and log
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                f.write(buf.read())
                temp_path = f.name

            self.log_artifact(temp_path, filename)
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to log figure: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set tag for the run."""
        if not self._is_active and self._active_run is None:
            self.start_run()

        self._mlflow.set_tag(key, value)

    def log_dataset(self, dataset_info: Dict[str, Any]) -> None:
        """Log dataset information as a tag."""
        dataset_json = json.dumps(dataset_info, ensure_ascii=False)
        self.set_tag("dataset_info", dataset_json)

    def log_training_progress(
        self,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Log training progress with epoch and step."""
        self.set_tag("current_epoch", str(epoch))
        self.log_metrics(metrics, step=step)

    def finish(self, status: str = "FINISHED") -> None:
        """End the MLflow run."""
        if self._active_run is not None:
            try:
                self._mlflow.end_run(status=status)
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self._active_run = None


class JSONTracker(ExperimentTracker):
    """JSON file-based experiment tracking (fallback without MLflow)."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "outputs/experiments",
        config: Optional[ExperimentConfig] = None,
    ):
        super().__init__(experiment_name, config)

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
            "status": "running",
        }

        # Add config tags
        if config and config.tags:
            self.data["tags"].update(config.tags)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to JSON."""
        for key, value in params.items():
            if isinstance(value, (list, dict, tuple)):
                value = json.dumps(value, ensure_ascii=False)
            self.data["params"][key] = str(value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to JSON."""
        for key, value in metrics.items():
            if step is None:
                self.data["metrics"][key] = float(value)
            else:
                if key not in self.data["metrics"]:
                    self.data["metrics"][key] = []
                self.data["metrics"][key].append({"step": int(step), "value": float(value)})

    def log_model(
        self,
        model_path: str,
        name: str = "model",
        model_type: str = "pytorch",
        **kwargs
    ) -> None:
        """Log model to JSON registry."""
        self.data["artifacts"].append({
            "type": "model",
            "path": str(model_path),
            "name": name,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
        })

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        """Log artifact to JSON."""
        self.data["artifacts"].append({
            "type": "artifact",
            "path": str(file_path),
            "name": name or Path(file_path).name,
            "timestamp": datetime.now().isoformat(),
        })

    def log_artifacts(self, artifact_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log all artifacts in a directory."""
        artifact_dir = Path(artifact_dir)
        if artifact_dir.is_dir():
            for file_path in artifact_dir.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(artifact_dir)
                    name = str(artifact_path / rel_path) if artifact_path else str(rel_path)
                    self.log_artifact(str(file_path), name)

    def log_text(self, text: str, filename: str) -> None:
        """Log text as an artifact file."""
        output_path = self.output_dir / "artifacts" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        self.log_artifact(str(output_path), filename)

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        """Log dictionary as JSON artifact."""
        self.log_text(json.dumps(data, indent=2, ensure_ascii=False), filename)

    def log_figure(self, figure, filename: str) -> None:
        """Log matplotlib figure."""
        try:
            output_path = self.output_dir / "artifacts" / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, bbox_inches='tight')
            self.log_artifact(str(output_path), filename)
        except Exception as e:
            logger.warning(f"Failed to log figure: {e}")

    def set_tag(self, key: str, value: str) -> None:
        """Set tag."""
        self.data["tags"][key] = value

    def finish(self, status: str = "finished") -> None:
        """Save experiment to JSON file."""
        self.data["end_time"] = datetime.now().isoformat()
        self.data["status"] = status

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"{self.experiment_name}_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        logger.info(f"Experiment saved to {output_path}")


class NullTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""

    def log_params(self, params: Dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def log_model(self, model_path: str, name: str = "model", **kwargs) -> None:
        pass

    def log_artifact(self, file_path: str, name: Optional[str] = None) -> None:
        pass

    def log_artifacts(self, artifact_dir: str, artifact_path: Optional[str] = None) -> None:
        pass

    def log_text(self, text: str, filename: str) -> None:
        pass

    def log_dict(self, data: Dict[str, Any], filename: str) -> None:
        pass

    def log_figure(self, figure, filename: str) -> None:
        pass

    def set_tag(self, key: str, value: str) -> None:
        pass


# ============================================================================
# Experiment Manager
# ============================================================================

class ExperimentManager:
    """
    Factory and manager for experiment tracking.

    Automatically selects backend based on availability and configuration.
    """

    def __init__(
        self,
        default_backend: str = "auto",
        mlflow_tracking_uri: Optional[str] = None,
        json_output_dir: str = "outputs/experiments",
        config: Optional[ExperimentConfig] = None,
    ):
        """
        Args:
            default_backend: "auto", "mlflow", "json", "none"
            mlflow_tracking_uri: MLflow server URI
            json_output_dir: Directory for JSON experiments
            config: Experiment configuration
        """
        self.default_backend = default_backend
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.json_output_dir = json_output_dir
        self.config = config or ExperimentConfig()

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
            if not self.config.enabled:
                backend = "none"
            elif self.has_mlflow:
                backend = "mlflow"
            else:
                backend = "json"

        if backend == "mlflow":
            if not self.has_mlflow:
                logger.warning("MLflow not available, falling back to JSON")
                backend = "json"

        if backend == "mlflow":
            return MLflowTracker(
                name,
                tracking_uri=self.mlflow_tracking_uri,
                config=self.config,
            )
        elif backend == "json":
            return JSONTracker(name, self.json_output_dir, self.config)
        elif backend == "none":
            return NullTracker(name, self.config)
        else:
            raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Context Manager for Experiments
# ============================================================================

class MLflowExperiment:
    """
    Context manager for running experiments with automatic resource management.

    Usage:
        with MLflowExperiment("my_test") as exp:
            exp.log_params({"lr": 0.001})
            exp.log_metrics({"accuracy": 0.95})

    The experiment is automatically finalized when exiting the context.
    """

    def __init__(
        self,
        name: str,
        manager: Optional[ExperimentManager] = None,
        config: Optional[ExperimentConfig] = None,
        **kwargs,
    ):
        self.name = name
        if manager is None:
            manager = ExperimentManager(config=config, **kwargs)
        self.manager = manager
        self.experiment: Optional[ExperimentTracker] = None

    def __enter__(self) -> ExperimentTracker:
        self.experiment = self.manager.create_experiment(self.name)
        if hasattr(self.experiment, 'start_run'):
            self.experiment.start_run()
        return self.experiment

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.experiment:
            status = "FINISHED" if exc_type is None else "FAILED"
            if hasattr(self.experiment, 'finish'):
                self.experiment.finish(status=status)


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
        model_type: str = "pytorch",
    ) -> str:
        """
        Register a model.

        Args:
            model_path: Path to model files
            name: Model name
            version: Version string (auto-generated if None)
            tags: Tags for the model
            model_type: Type of model (pytorch, huggingface, etc.)

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
                mlflow.set_tag("model_type", model_type)

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
            "model_type": model_type,
            "tags": tags or {},
            "registered_at": datetime.now().isoformat(),
        }

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

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
) -> "pd.DataFrame":
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
            import pandas as pd
        except ImportError:
            raise ImportError("MLflow not installed")

        results = []

        for exp_id in experiment_ids:
            run = mlflow.get_run(exp_id)
            row = {"experiment_id": exp_id}

            for metric in metric_names:
                row[metric] = run.data.metrics.get(metric)

            # Add params
            for key, value in run.data.params.items():
                row[f"param.{key}"] = value

            results.append(row)

        return pd.DataFrame(results)

    else:
        # JSON backend - scan experiment directory
        import pandas as pd

        exp_dir = Path("outputs/experiments")
        results = []

        for file_path in exp_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
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


def get_best_run(
    experiment_name: str,
    metric: str,
    mode: str = "max",
) -> Optional[Dict[str, Any]]:
    """
    Get the best run for an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric name
        mode: "max" or "min" - whether higher or lower is better

    Returns:
        Dictionary with run information or None
    """
    try:
        import mlflow

        runs = mlflow.search_runs(experiment_names=[experiment_name])

        if runs.empty:
            return None

        # Filter runs that have the metric
        runs = runs[runs[f"metrics.{metric}"].notna()]

        if runs.empty:
            return None

        # Get best run
        if mode == "max":
            best_idx = runs[f"metrics.{metric}"].idxmax()
        else:
            best_idx = runs[f"metrics.{metric}"].idxmin()

        best_run = runs.loc[best_idx]

        return {
            "run_id": best_run.get("run_id"),
            "metric_value": best_run.get(f"metrics.{metric}"),
            "params": {
                k.replace("params.", ""): v
                for k, v in best_run.items()
                if k.startswith("params.")
            },
        }
    except Exception as e:
        logger.warning(f"Failed to get best run: {e}")
        return None


# ============================================================================
# Decorators for Experiment Tracking
# ============================================================================

def track_experiment(
    name: Optional[str] = None,
    params: Optional[Dict] = None,
    config: Optional[ExperimentConfig] = None,
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

            with MLflowExperiment(exp_name, config=config) as exp:
                if params:
                    exp.log_params(params)

                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                    exp.log_metrics(metrics)

            return result

        return wrapper
    return decorator


# ============================================================================
# Training Callback Integration
# ============================================================================

class MLflowCallback:
    """
    Callback for integrating MLflow tracking with HuggingFace Transformers
    and TRL training.

    Usage:
        from transformers import Trainer

        mlflow_callback = MLflowCallback(experiment_name="sft_tourism")
        trainer = Trainer(..., callbacks=[mlflow_callback])
    """

    def __init__(
        self,
        experiment_name: str = "training",
        config: Optional[ExperimentConfig] = None,
        log_model: bool = True,
        log_artifacts: bool = True,
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.log_model = log_model
        self.log_artifacts = log_artifacts
        self.tracker: Optional[ExperimentTracker] = None
        self._manager: Optional[ExperimentManager] = None

    def on_init_end(self, args, state, control, **kwargs):
        """Initialize MLflow run when training starts."""
        self._manager = ExperimentManager(config=self.config)
        self.tracker = self._manager.create_experiment(self.experiment_name)

        # Log training parameters
        params = {
            "learning_rate": args.learning_rate,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_seq_length": getattr(args, "max_seq_length", None),
            "warmup_ratio": getattr(args, "warmup_ratio", None),
        }
        self.tracker.log_params({k: v for k, v in params.items() if v is not None})

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics during training."""
        if logs and self.tracker:
            metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            step = state.global_step
            self.tracker.log_metrics(metrics, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        """Finalize MLflow run when training ends."""
        if self.tracker:
            # Log final metrics
            if state.log_history:
                final_metrics = {}
                for log in state.log_history[-5:]:  # Last few logs
                    for k, v in log.items():
                        if isinstance(v, (int, float)) and k != "epoch":
                            final_metrics[f"final_{k}"] = v
                self.tracker.log_metrics(final_metrics)

            # Log model and artifacts
            if self.log_model and args.output_dir:
                self.tracker.log_model(args.output_dir)

            if self.log_artifacts:
                # Log training config
                import json
                config_path = Path(args.output_dir) / "training_args.bin"
                if config_path.exists():
                    self.tracker.log_artifact(str(config_path))

            self.tracker.finish()


# ============================================================================
# Utility Functions
# ============================================================================

def setup_mlflow_server(
    backend_store_uri: str = "sqlite:///mlflow.db",
    default_artifact_root: str = "mlflow-artifacts",
    host: str = "0.0.0.0",
    port: int = 5000,
    workers: int = 4,
):
    """
    Start an MLflow server (blocking call).

    Args:
        backend_store_uri: Database URI for backend store
        default_artifact_root: Root directory for artifacts
        host: Server host
        port: Server port
        workers: Number of worker processes
    """
    import subprocess

    cmd = [
        "mlflow", "server",
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", default_artifact_root,
        "--host", host,
        "--port", str(port),
        "--workers", str(workers),
    ]

    logger.info(f"Starting MLflow server: {' '.join(cmd)}")
    subprocess.run(cmd)


def auto_detect_tracking_uri() -> Optional[str]:
    """
    Auto-detect MLflow tracking URI from common configurations.

    Returns:
        Tracking URI or None
    """
    # Check environment variable
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    # Check for local MLflow server
    common_ports = [5000, 5001, 5002]
    for port in common_ports:
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/health", timeout=1)
            if response.status_code == 200:
                uri = f"http://localhost:{port}"
                logger.info(f"Auto-detected MLflow server at {uri}")
                return uri
        except Exception:
            pass

    return None


@contextmanager
def experiment_context(
    name: str,
    tracking_uri: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
):
    """
    Context manager for experiment tracking with auto-setup.

    Usage:
        with experiment_context("my_exp") as exp:
            exp.log_params({"lr": 0.001})
            exp.log_metrics({"loss": 0.5})
    """
    if tracking_uri is None:
        tracking_uri = auto_detect_tracking_uri()

    manager = ExperimentManager(
        default_backend="auto",
        mlflow_tracking_uri=tracking_uri,
        config=config,
    )

    tracker = manager.create_experiment(name)
    if hasattr(tracker, 'start_run'):
        tracker.start_run()

    try:
        yield tracker
    finally:
        tracker.finish()


# ============================================================================
# CLI entry point
# ============================================================================

def main():
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="MLflow Experiment Tracking")
    subparsers = parser.add_subparsers(dest="command")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("--experiments", nargs="+", required=True)
    compare_parser.add_argument("--metrics", nargs="+", default=["recall@10", "ndcg@10"])
    compare_parser.add_argument("--output", help="Save comparison to CSV")
    compare_parser.add_argument("--backend", default="mlflow", choices=["mlflow", "json"])

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--backend", default="mlflow", choices=["mlflow", "json"])
    list_parser.add_argument("--experiment", help="Filter by experiment name")

    # Best run command
    best_parser = subparsers.add_parser("best", help="Get best run")
    best_parser.add_argument("--experiment", required=True, help="Experiment name")
    best_parser.add_argument("--metric", required=True, help="Metric name")
    best_parser.add_argument("--mode", default="max", choices=["max", "min"])

    # Server command
    server_parser = subparsers.add_parser("server", help="Start MLflow server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=5000)
    server_parser.add_argument("--backend-store", default="sqlite:///mlflow.db")
    server_parser.add_argument("--artifact-root", default="mlflow-artifacts")

    args = parser.parse_args()

    if args.command == "compare":
        df = compare_experiments(args.experiments, args.metrics, args.backend)
        print(df)

        if args.output:
            df.to_csv(args.output, index=False)

    elif args.command == "list":
        if args.backend == "mlflow":
            import mlflow
            runs = mlflow.search_runs()
            if args.experiment:
                runs = runs[runs["experiment_name"] == args.experiment]
            print(runs[["run_id", "experiment_id", "status", "start_time"]])
        else:
            exp_dir = Path("outputs/experiments")
            for path in exp_dir.glob("*.json"):
                if args.experiment is None or args.experiment in path.stem:
                    print(path.stem)

    elif args.command == "best":
        best = get_best_run(args.experiment, args.metric, args.mode)
        if best:
            print(f"Best run: {best['run_id']}")
            print(f"{args.metric}: {best['metric_value']}")
            print("Parameters:")
            for k, v in best["params"].items():
                print(f"  {k}: {v}")
        else:
            print("No runs found")

    elif args.command == "server":
        setup_mlflow_server(
            backend_store_uri=args.backend_store,
            default_artifact_root=args.artifact_root,
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()
