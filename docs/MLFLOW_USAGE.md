# MLflow Experiment Tracking - GoAfar User Guide

## Overview

The GoAfar project integrates MLflow for comprehensive experiment tracking, enabling you to monitor, compare, and manage machine learning experiments effectively.

## Features

- **Parameter Logging**: Track hyperparameters, model configurations, and dataset information
- **Metric Tracking**: Record training metrics (loss, accuracy, rewards, etc.) step-by-step
- **Model Artifacts**: Automatically save trained models with versioning
- **Multiple Backends**: MLflow server, JSON file (fallback), or disabled
- **Auto-detection**: Automatically detects running MLflow servers
- **Experiment Comparison**: Compare multiple experiments to find the best configuration

## Quick Start

### 1. Install Dependencies

```bash
pip install mlflow>=2.14.0
```

Or install all GoAfar dependencies:

```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server (Optional)

For full MLflow functionality, start an MLflow server:

```bash
# Start with default settings (SQLite backend)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000

# Or use the built-in command
python -m src.utils.experiment server
```

Then access the UI at `http://localhost:5000`

### 3. Configure Experiment Tracking

Edit `configs/runtime.yaml`:

```yaml
mlflow:
  enabled: true
  backend: auto  # auto, mlflow, json, none
  tracking_uri: null  # auto-detects or specify http://localhost:5000
  experiment_name: null
  json_output_dir: outputs/experiments
  tags:
    project: goafar
    version: "1.0"
```

Or set environment variables:

```bash
export MLFLOW_ENABLED=true
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=my_experiment
```

## Usage in Training Scripts

### SFT Training

```bash
# With MLflow enabled (default)
python -m src.content_generation.train_sft \
    --data outputs/datasets/sft_data.jsonl \
    --output outputs/sft/qwen3-8b-tourism

# Disable MLflow
python -m src.content_generation.train_sft \
    --data outputs/datasets/sft_data.jsonl \
    --output outputs/sft/qwen3-8b-tourism \
    --no-mlflow

# Specify custom MLflow server
python -m src.content_generation.train_sft \
    --data outputs/datasets/sft_data.jsonl \
    --output outputs/sft/qwen3-8b-tourism \
    --mlflow-uri http://remote-server:5000 \
    --mlflow-experiment my_sft_experiment
```

### DPO Training

```bash
# With MLflow enabled
python -m src.content_generation.train_dpo \
    --prefs outputs/datasets/dpo_prefs.csv \
    --output outputs/dpo/qwen3-8b-dpo

# Disable MLflow
python -m src.content_generation.train_dpo \
    --prefs outputs/datasets/dpo_prefs.csv \
    --output outputs/dpo/qwen3-8b-dpo \
    --no-mlflow
```

### GRPO Training

```bash
# Standard GRPO trainer
python -m src.rl.grpo_trainer \
    --model models/Qwen3-8B \
    --data outputs/datasets/grpo_planner_prompts.jsonl \
    --output outputs/grpo/qwen3-8b-planner

# TRL-based GRPO trainer
python -m src.rl.grpo_trainer_trl \
    --model models/Qwen3-8B \
    --data outputs/datasets/grpo_planner_prompts.jsonl \
    --output outputs/grpo/qwen3-grpo-planner
```

## Programmatic Usage

### Basic Experiment Tracking

```python
from src.utils.experiment import MLflowExperiment

# Simple context manager usage
with MLflowExperiment("my_experiment") as exp:
    exp.log_params({
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": "Qwen3-8B",
    })

    for step in range(100):
        # Your training code here
        loss = train_step()

        # Log metrics
        exp.log_metrics({"loss": loss}, step=step)

    # Log model and artifacts
    exp.log_model("outputs/my_model")
    exp.log_artifact("configs/training.yaml")

# Experiment automatically finalized
```

### Advanced Configuration

```python
from src.utils.experiment import (
    ExperimentManager,
    ExperimentConfig,
    MLflowTracker,
)

# Create custom configuration
config = ExperimentConfig(
    enabled=True,
    tracking_uri="http://localhost:5000",
    tags={"team": "recommendation", "phase": "production"},
)

# Create manager
manager = ExperimentManager(
    default_backend="mlflow",
    mlflow_tracking_uri="http://localhost:5000",
    config=config,
)

# Create experiment
tracker = manager.create_experiment("production_training")

# Use tracker
tracker.log_params({"lr": 0.001})
tracker.log_metrics({"accuracy": 0.95})

# Log model
tracker.log_model(
    "outputs/model",
    name="production_model",
    model_type="huggingface"
)

# Finish
tracker.finish(status="FINISHED")
```

### Decorator-based Tracking

```python
from src.utils.experiment import track_experiment

@track_experiment(
    name="model_training",
    params={"model": "Qwen3-8B", "lr": 0.001},
)
def train_model(config):
    # Training code
    return {"accuracy": 0.95, "f1": 0.93}

# Result automatically tracked
result = train_model(config)
```

### HuggingFace Transformers Callback

```python
from transformers import Trainer
from src.utils.experiment import MLflowCallback

# Create callback
mlflow_callback = MLflowCallback(
    experiment_name="sft_tourism",
    log_model=True,
    log_artifacts=True,
)

# Use with Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[mlflow_callback],
)

trainer.train()
```

## CLI Tools

### List Experiments

```bash
# List all MLflow experiments
python -m src.utils.experiment list --backend mlflow

# List JSON experiments
python -m src.utils.experiment list --backend json

# Filter by experiment name
python -m src.utils.experiment list --experiment sft_tourism
```

### Compare Experiments

```bash
# Compare multiple runs
python -m src.utils.experiment compare \
    --experiments run1 run2 run3 \
    --metrics loss accuracy recall@10 \
    --output comparison.csv
```

### Find Best Run

```bash
# Find best run by metric
python -m src.utils.experiment best \
    --experiment sft_tourism \
    --metric recall@10 \
    --mode max  # or min for lower-is-better
```

## Experiment Comparison and Analysis

### Using MLflow UI

1. Navigate to `http://localhost:5000`
2. Select your experiment
3. View runs in a table with metrics comparison
4. Visualize metrics over time
5. Compare parameter configurations

### Programmatic Comparison

```python
from src.utils.experiment import compare_experiments, get_best_run

# Compare multiple runs
df = compare_experiments(
    experiment_ids=["run1", "run2", "run3"],
    metric_names=["loss", "accuracy", "recall@10"],
)
print(df)

# Get best run
best = get_best_run(
    experiment_name="sft_tourism",
    metric="recall@10",
    mode="max",
)
print(f"Best run: {best['run_id']}")
print(f"Best metric value: {best['metric_value']}")
```

## Model Registry

### Register a Model

```python
from src.utils.experiment import ExperimentManager, ModelRegistry

manager = ExperimentManager()
registry = ModelRegistry(manager)

# Register model
model_id = registry.register_model(
    model_path="outputs/sft/qwen3-8b-tourism",
    name="tourism_sft",
    tags={"version": "v1.0", "date": "2024-01-15"},
)
print(f"Registered: {model_id}")
```

### Load a Model

```python
# Load latest version
model_path = registry.load_model("tourism_sft")

# Load specific version
model_path = registry.load_model("tourism_sft", version="v1.0")
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `enabled` | bool | `true` | Enable/disable tracking |
| `backend` | str | `auto` | Backend: auto, mlflow, json, none |
| `tracking_uri` | str | `null` | MLflow server URI |
| `experiment_name` | str | `null` | Default experiment name |
| `json_output_dir` | str | `outputs/experiments` | JSON fallback directory |
| `artifact_location` | str | `mlflow-artifacts` | Artifact storage location |
| `auto_flush` | bool | `true` | Auto-flush metrics after logging |
| `tags` | dict | `{}` | Default experiment tags |

## Troubleshooting

### MLflow Connection Issues

```bash
# Check if MLflow server is running
curl http://localhost:5000/health

# Set explicit tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Use JSON fallback
export MLFLOW_BACKEND=json
```

### Large Artifacts

```python
# Log specific files instead of entire directory
tracker.log_artifact("model/config.json")
tracker.log_artifact("model/pytorch_model.bin")

# Or log as model
tracker.log_model("model", name="checkpoint")
```

### Memory Issues

```python
# Disable artifact logging
tracker.log_model(path, log_artifacts=False)

# Use JSON backend for minimal overhead
config = ExperimentConfig(backend="json")
```

## Best Practices

1. **Always name experiments** meaningfully: `sft_tourism_qwen3_8b_lr2e4`
2. **Log dataset info** with `log_dataset()` for reproducibility
3. **Use tags** to organize experiments by team, phase, or model type
4. **Log artifacts** selectively to avoid storage bloat
5. **Compare runs** before deciding on final configuration
6. **Register best models** to Model Registry for production use

## Integration with Existing Code

To add experiment tracking to custom training code:

```python
from src.utils.experiment import MLflowExperiment

def my_training_function(config):
    with MLflowExperiment("my_custom_training") as exp:
        exp.log_params(config)

        for epoch in range(config.epochs):
            metrics = train_epoch(epoch)
            exp.log_metrics(metrics, step=epoch)

        exp.log_model(config.output_dir)
```

## Visualization with MLflow UI

### Starting the MLflow UI

```bash
# Method 1: Using mlflow server command
mlflow ui \
    --backend-store-uri sqlite:///mlflow.db \
    --port 5000

# Method 2: Using the built-in helper
python -m src.utils.experiment server --port 5000

# Method 3: Remote server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

### Using the MLflow UI

1. **Experiments View**
   - Browse to `http://localhost:5000`
   - See all experiments organized by name
   - Compare runs side by side

2. **Metrics Charting**
   - Select multiple runs to compare
   - View metrics over time (step plots)
   - Customize chart types: line, scatter, bar

3. **Parameter Comparison**
   - View parameters table across runs
   - Identify which parameters affect performance

4. **Artifacts**
   - Download logged models and artifacts
   - View saved figures and plots

### Plotting Custom Metrics

```python
from src.utils.experiment import MLflowExperiment
import matplotlib.pyplot as plt

with MLflowExperiment("plotting_example") as exp:
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curve")

    # Log the figure
    exp.log_figure(fig, "training_curve.png")

    # Or log as artifact
    # fig.savefig("temp.png")
    # exp.log_artifact("temp.png")
```

### Exporting Data from MLflow

```python
import mlflow

# Search runs
runs = mlflow.search_runs(experiment_names=["sft_tourism"])

# Export to CSV
df = runs[["run_id", "params.learning_rate", "metrics.recall@10"]]
df.to_csv("experiment_results.csv", index=False)

# Get detailed metrics for a run
run = mlflow.get_run("run_id_here")
history = mlflow.tracking.MlflowClient().get_metric_history(
    "run_id_here", "loss"
)
```

## References

- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- MLflow Tracking API: https://mlflow.org/docs/latest/python_api/index.html
- TRL Library: https://huggingface.co/docs/trl
