"""
TRL-based GRPO Trainer for Route Planning.

This module provides an alternative implementation using the TRL library's
GRPOTrainer, which offers better performance and more features.

Key features:
- Uses TRL's optimized GRPOTrainer
- Custom reward function for route planning
- Supports distributed training
- Better logging and checkpointing
- Integrated MLflow experiment tracking

Requirements:
    pip install trl>=0.12.0

Reference: https://huggingface.co/docs/trl/main/en/grpo_trainer
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from trl import GRPOTrainer as TrlGRPOTrainer, GRPOConfig as TrlGRPOConfig
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    logger.warning("TRL not installed. Install with: pip install trl")

from .reward_manager import RewardManager, RewardWeights

# 导入实验追踪模块
try:
    from src.utils.experiment import (
        ExperimentConfig,
        ExperimentManager,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class TourismGRPOConfig:
    """Configuration for tourism route planning GRPO training."""

    # Model
    model_name_or_path: str = "models/Qwen3-8B"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True

    # GRPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4  # per device
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 512
    max_completion_length: int = 256
    max_length: int = 2048

    # GRPO-specific
    group_size: int = 4  # Number of completions per prompt
    kl_coef: float = 0.1

    # Training
    num_train_epochs: int = 3
    save_steps: int = 100
    logging_steps: int = 10

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Data
    train_data: str = "outputs/datasets/grpo_planner_prompts.jsonl"

    # Optional POI data for advanced rewards
    poi_data_path: Optional[str] = None
    time_matrix_path: Optional[str] = None

    # Output
    output_dir: str = "outputs/grpo/qwen3-grpo-planner"

    # Reward weights
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # Experiment tracking
    use_mlflow: bool = True  # Enable MLflow tracking
    mlflow_tracking_uri: Optional[str] = None  # MLflow server URI
    mlflow_experiment_name: str = "grpo_planner_trl"  # Experiment name


class TourismRewardFunction:
    """
    Reward function for tourism route planning.

    Computes rewards based on:
    1. Output format validity
    2. Target POI matching (if available)
    3. Route feasibility (time windows)
    4. User preferences
    """

    def __init__(
        self,
        poi_df: Optional[pd.DataFrame] = None,
        time_matrix: Optional[np.ndarray] = None,
        weights: Optional[RewardWeights] = None,
    ):
        self.poi_df = poi_df
        self.time_matrix = time_matrix
        self.reward_manager = RewardManager(weights=weights)

        if poi_df is not None:
            self._build_poi_index()

    def _build_poi_index(self):
        """Build POI ID to index mapping."""
        if self.poi_df is None:
            return
        self.poi_idx_map = {
            str(pid): idx
            for idx, pid in enumerate(self.poi_df["poi_id"].astype(str).tolist())
        }
        logger.info(f"Built POI index with {len(self.poi_idx_map)} entries")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of prompt-completion pairs.

        Args:
            prompts: List of prompt texts
            completions: List of generated completions
            **kwargs: Additional metadata (may contain target_poi, etc.)

        Returns:
            List of reward scores
        """
        rewards = []
        metadata_list = kwargs.get("metadata", [])

        for idx, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                # Extract metadata from prompt if available
                metadata = {}
                if isinstance(metadata_list, list) and idx < len(metadata_list):
                    candidate = metadata_list[idx]
                    if isinstance(candidate, dict):
                        metadata = candidate
                if not metadata:
                    metadata = self._parse_prompt_metadata(prompt)

                # Extract POI ID from completion
                poi_id = self._extract_poi_id(completion)

                # Initialize reward
                reward = 0.0

                # 1. Format validity
                if poi_id and len(poi_id) > 0:
                    reward += 0.5

                    # 2. Target matching
                    target_poi = metadata.get("target_next_poi")
                    if target_poi and poi_id == str(target_poi):
                        reward += 2.0

                    # 3. Database validity
                    if hasattr(self, 'poi_idx_map') and self.poi_idx_map:
                        if poi_id in self.poi_idx_map:
                            reward += 0.5
                        else:
                            reward -= 0.5

                    # 4. Advanced route scoring
                    if self.poi_df is not None and self.time_matrix is not None:
                        route_score = self._score_route(poi_id, metadata)
                        reward += route_score
                else:
                    reward = -1.0

                rewards.append(reward)

            except Exception as e:
                logger.debug(f"Reward computation failed: {e}")
                rewards.append(-1.0)

        return rewards

    def _parse_prompt_metadata(self, prompt: str) -> Dict:
        """Extract metadata from prompt text."""
        try:
            # Look for JSON in prompt
            import re
            json_match = re.search(r'\{[^}]*"task"[^}]*\}', prompt)
            if json_match:
                data = json.loads(json_match.group(0))
                return data
        except:
            pass
        return {}

    def _extract_poi_id(self, text: str) -> Optional[str]:
        """Extract POI ID from text."""
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
        import re
        s_match = re.search(r'S\d+', text, re.IGNORECASE)
        if s_match:
            return s_match.group(0).upper()

        # Try numeric
        num_match = re.search(r'\d{4,}', text)
        if num_match:
            return num_match.group(0)

        return None

    def _score_route(self, poi_id: str, metadata: Dict) -> float:
        """Score route using RewardManager."""
        try:
            state_prefix = metadata.get("state_prefix", [])
            if not state_prefix or not hasattr(self, 'poi_idx_map'):
                return 0.0

            route = state_prefix + [poi_id]
            if poi_id not in self.poi_idx_map:
                return 0.0

            scores = self.reward_manager.score_route(
                route_poi_ids=route,
                poi_df=self.poi_df,
                time_matrix=self.time_matrix,
            )

            # Normalize to [-1, 1]
            total = scores.get("total_reward", 0.0)
            return np.tanh(total / 10.0)

        except Exception as e:
            logger.debug(f"Route scoring failed: {e}")
            return 0.0


class TourismGRPODataset(torch.utils.data.Dataset):
    """Dataset for GRPO training."""

    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))
                if max_samples and len(self.data) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]

        # Format prompt for LLM
        prompt_dict = json.loads(item["prompt"])

        # Create chat format
        messages = [
            {"role": "system", "content": "你是一位专业的旅游规划助手。"},
            {"role": "user", "content": self._format_user_message(prompt_dict)},
        ]

        metadata = dict(prompt_dict)
        if "target_next_poi" in item and item.get("target_next_poi") not in (None, ""):
            metadata["target_next_poi"] = str(item.get("target_next_poi"))
        if "full_target_route" in item and isinstance(item.get("full_target_route"), list):
            metadata["full_target_route"] = item.get("full_target_route")

        return {
            "prompt": json.dumps(messages, ensure_ascii=False),
            "metadata": metadata,
        }

    def _format_user_message(self, prompt_dict: Dict) -> str:
        """Format user message from prompt dictionary."""
        parts = []
        if prompt_dict.get("province"):
            parts.append(f"目的地：{prompt_dict['province']}")
        if prompt_dict.get("day"):
            parts.append(f"日期：{prompt_dict['day']}")
        if prompt_dict.get("interests"):
            parts.append(f"兴趣：{', '.join(prompt_dict['interests'])}")
        if prompt_dict.get("state_prefix"):
            visited = prompt_dict["state_prefix"]
            if visited:
                parts.append(f"已选：{', '.join(visited[-3:])}")

        instruction = prompt_dict.get("instruction", "请推荐下一个景点。")
        return "\n".join(parts) + "\n\n" + instruction


def train_grpo_with_trl(config: TourismGRPOConfig):
    """
    Train route planning model using TRL's GRPOTrainer.

    This is the recommended approach if TRL is available.
    """
    if not TRL_AVAILABLE:
        raise ImportError(
            "TRL is not installed. Install with: pip install trl"
            "\nOr use the alternative implementation in grpo_trainer.py"
        )

    logger.info("Initializing TRL-based GRPO training...")

    # Initialize experiment tracking
    experiment_tracker = None
    if config.use_mlflow and MLFLOW_AVAILABLE:
        logger.info(f"Initializing MLflow experiment tracking: {config.mlflow_experiment_name}")
        exp_config = ExperimentConfig(
            enabled=True,
            tracking_uri=config.mlflow_tracking_uri,
            tags={"task": "grpo_trl", "model": config.model_name_or_path},
        )
        manager = ExperimentManager(
            mlflow_tracking_uri=config.mlflow_tracking_uri,
            config=exp_config,
        )
        experiment_tracker = manager.create_experiment(config.mlflow_experiment_name)
        if hasattr(experiment_tracker, 'start_run'):
            experiment_tracker.start_run()

        # Log configuration
        params = {
            "model_name_or_path": config.model_name_or_path,
            "train_data": config.train_data,
            "output_dir": config.output_dir,
            "group_size": config.group_size,
            "kl_coef": config.kl_coef,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "num_train_epochs": config.num_train_epochs,
            "max_prompt_length": config.max_prompt_length,
            "max_completion_length": config.max_completion_length,
            "use_lora": config.use_lora,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
        }
        experiment_tracker.log_params(params)

    # Load model and tokenizer
    logger.info(f"Loading model from {config.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path or config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load POI data for advanced rewards
    poi_df = None
    time_matrix = None
    if config.poi_data_path and Path(config.poi_data_path).exists():
        logger.info(f"Loading POI data from {config.poi_data_path}")
        poi_df = pd.read_parquet(config.poi_data_path)
        if config.time_matrix_path and Path(config.time_matrix_path).exists():
            logger.info(f"Loading time matrix from {config.time_matrix_path}")
            time_matrix = np.load(config.time_matrix_path)

    # Create reward function
    reward_fn = TourismRewardFunction(
        poi_df=poi_df,
        time_matrix=time_matrix,
        weights=config.reward_weights,
    )

    # Load dataset
    logger.info(f"Loading dataset from {config.train_data}")
    train_dataset = TourismGRPODataset(config.train_data)
    logger.info(f"Loaded {len(train_dataset)} samples")

    # Log dataset info
    if experiment_tracker:
        dataset_info = {
            "train_data": config.train_data,
            "num_samples": len(train_dataset),
        }
        experiment_tracker.log_dataset(dataset_info)

    # Configure LoRA
    lora_config = None
    if config.use_lora:
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

    # Create TRL GRPO config
    trl_config = TrlGRPOConfig(
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_length=config.max_length,
        num_generations=config.group_size,
        kl_coef=config.kl_coef,
        num_train_epochs=config.num_train_epochs,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        output_dir=config.output_dir,
        logging_first_step=True,
        report_to=["tensorboard"],
        # LoRA
        peft_config=lora_config if config.use_lora else None,
    )

    # Create trainer
    trainer = TrlGRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=trl_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Finalize experiment tracking
    if experiment_tracker:
        # Log final metrics
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            final_metrics = {}
            for log in trainer.state.log_history[-5:]:
                for k, v in log.items():
                    if isinstance(v, (int, float)) and k != 'epoch':
                        final_metrics[f'final_{k}'] = v
            experiment_tracker.log_metrics(final_metrics)

        # Log model
        experiment_tracker.log_model(
            config.output_dir,
            name="grpo_trl_model",
            model_type="huggingface"
        )

        experiment_tracker.finish(status="FINISHED")

    logger.info("Training completed successfully!")
    return trainer


# Add missing import
import torch


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train route planning with TRL GRPO"
    )
    parser.add_argument("--model", type=str, default="models/Qwen3-8B")
    parser.add_argument("--data", type=str,
                        default="outputs/datasets/grpo_planner_prompts.jsonl")
    parser.add_argument("--output", type=str,
                        default="outputs/grpo/qwen3-grpo-planner")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--poi-data", type=str,
                        default="data/processed/pois_with_embeddings.parquet")
    parser.add_argument("--time-matrix", type=str,
                        default="data/processed/time_matrix.npy")

    args = parser.parse_args()

    config = TourismGRPOConfig(
        model_name_or_path=args.model,
        train_data=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        poi_data_path=args.poi_data,
        time_matrix_path=args.time_matrix,
    )

    train_grpo_with_trl(config)


if __name__ == "__main__":
    main()
