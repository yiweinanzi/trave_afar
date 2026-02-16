"""
GRPO Trainer for Route Planning using TRL/veRL framework.

This module implements Group Relative Policy Optimization (GRPO) for training
a route planning policy that generates POI sequences satisfying time windows
and user preferences.

Key features:
- Group sampling for advantage estimation (no critic needed)
- KL divergence penalty for policy stability
- Hybrid reward: rule-based (feasibility) + model-based (preference score)
- Compatible with Qwen3-8B as the base policy model
- Supports both TRL (default) and veRL backends
- Integrated MLflow experiment tracking

References:
- GRPO paper: https://arxiv.org/abs/2406.01806 (DeepSeekMath)
- TRL library: https://github.com/huggingface/trl
- veRL framework: https://github.com/volcengine/verl (optional)
"""
from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalPrediction
from tqdm import tqdm

from .dataset_builder import _iter_jsonl
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

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    model_name_or_path: str = "models/Qwen3-8B"
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = True

    # GRPO specific
    group_size: int = 4  # Number of samples per prompt for advantage estimation
    kl_coef: float = 0.1  # KL divergence coefficient
    clip_range: float = 0.2  # PPO-style clipping ratio
    use_grpo_advantage: bool = True  # Use group-relative advantage (no critic)
    use_reference_model: bool = True  # Use a reference policy for KL regularization

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    max_length: int = 2048
    max_prompt_length: int = 512
    max_new_tokens: int = 256

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Reward
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # I/O
    train_data: str = "outputs/datasets/grpo_planner_prompts.jsonl"
    output_dir: str = "outputs/grpo/qwen3-8b-planner"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = True

    # Optional: veRL backend (if installed)
    use_verl: bool = False

    # Experiment tracking
    use_mlflow: bool = True  # Enable MLflow tracking
    mlflow_tracking_uri: Optional[str] = None  # MLflow server URI
    mlflow_experiment_name: str = "grpo_planner"  # Experiment name


class RoutePlanningDataset(torch.utils.data.Dataset):
    """Dataset for route planning GRPO training."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        max_prompt_length: int = 512,
    ):
        self.data = list(_iter_jsonl(data_path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        # Qwen chat template
        self.chat_template = tokenizer.chat_template

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]

        # Parse prompt from stored JSON
        prompt_dict = json.loads(item["prompt"])
        target = item.get("target_next_poi", "")

        # Format as chat message
        messages = [
            {"role": "system", "content": "你是一位专业的旅游规划助手，负责推荐景点并规划行程。"},
            {"role": "user", "content": self._format_user_prompt(prompt_dict)},
        ]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        encodings = self.tokenizer(
            text,
            max_length=self.max_prompt_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "prompt_input_ids": encodings["input_ids"],
            "prompt_attention_mask": encodings["attention_mask"],
            "target_poi": target,
            "metadata": prompt_dict,
        }

    def _format_user_prompt(self, prompt_dict: Dict) -> str:
        """Format user prompt from dictionary."""
        parts = []
        if prompt_dict.get("province"):
            parts.append(f"目的地省份：{prompt_dict['province']}")
        if prompt_dict.get("day"):
            parts.append(f"行程天数：第{prompt_dict['day']}天")
        if prompt_dict.get("interests"):
            parts.append(f"兴趣偏好：{', '.join(prompt_dict['interests'])}")
        if prompt_dict.get("state_prefix"):
            visited = prompt_dict["state_prefix"]
            if visited:
                parts.append(f"已选景点：{', '.join(visited[-3:])}")  # Last 3

        instruction = prompt_dict.get("instruction", "请推荐下一个最合适的景点。")
        return "\n".join(parts) + "\n\n" + instruction


class GRPOTrainer:
    """
    GRPO Trainer for route planning policy.

    Implements Group Relative Policy Optimization without a critic:
    - Sample N responses per prompt
    - Compute advantages using group-relative rewards
    - Optimize policy with KL penalty
    """

    def __init__(self, config: GRPOConfig, poi_df: Optional[pd.DataFrame] = None,
                 time_matrix: Optional[np.ndarray] = None):
        self.config = config
        self.device = torch.device(config.device)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialize experiment tracking
        self.experiment_tracker = None
        if config.use_mlflow and MLFLOW_AVAILABLE:
            logger.info(f"Initializing MLflow experiment tracking: {config.mlflow_experiment_name}")
            exp_config = ExperimentConfig(
                enabled=True,
                tracking_uri=config.mlflow_tracking_uri,
                tags={"task": "grpo", "model": "Qwen3-8B"},
            )
            self.experiment_manager = ExperimentManager(
                mlflow_tracking_uri=config.mlflow_tracking_uri,
                config=exp_config,
            )
            self.experiment_tracker = self.experiment_manager.create_experiment(
                config.mlflow_experiment_name
            )
            if hasattr(self.experiment_tracker, 'start_run'):
                self.experiment_tracker.start_run()

        # Load model and tokenizer
        logger.info(f"Loading model from {config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path or config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float16 if config.fp16 else torch.float32,
            device_map="auto" if config.device == "cuda" else None,
        )

        # Apply LoRA if configured
        if config.use_lora:
            self._setup_lora()

        # Initialize reference policy used by KL regularization.
        self.ref_model: Optional[PreTrainedModel] = None
        self._reference_mode = "none"
        self._init_reference_policy()

        # Load dataset
        logger.info(f"Loading training data from {config.train_data}")
        self.train_dataset = RoutePlanningDataset(
            config.train_data,
            self.tokenizer,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )
        logger.info(f"Loaded {len(self.train_dataset)} training samples")

        # Log configuration after dataset is ready.
        if self.experiment_tracker:
            self._log_config_to_experiment()

        # Reward manager
        self.reward_manager = RewardManager(weights=config.reward_weights)

        # Optional: POI data for advanced reward computation
        self.poi_df = poi_df
        self.time_matrix = time_matrix

        if poi_df is not None:
            logger.info(f"POI data loaded: {len(poi_df)} POIs")
            self._build_poi_index()
        else:
            self.poi_idx_map = None
            logger.warning("No POI data provided, using basic reward computation")

        # Training state
        self.global_step = 0
        self.best_reward = -float("inf")

    def _log_config_to_experiment(self):
        """Log configuration parameters to experiment tracker."""
        if not self.experiment_tracker:
            return

        params = {
            "model_name_or_path": self.config.model_name_or_path,
            "train_data": self.config.train_data,
            "output_dir": self.config.output_dir,
            "group_size": self.config.group_size,
            "kl_coef": self.config.kl_coef,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "num_train_epochs": self.config.num_train_epochs,
            "max_length": self.config.max_length,
            "max_prompt_length": self.config.max_prompt_length,
            "max_new_tokens": self.config.max_new_tokens,
            "use_lora": self.config.use_lora,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "device": self.config.device,
        }
        self.experiment_tracker.log_params(params)

        # Log dataset info
        dataset = getattr(self, "train_dataset", None)
        dataset_info = {
            "train_data": self.config.train_data,
            "num_samples": len(dataset) if dataset is not None else 0,
        }
        self.experiment_tracker.log_dataset(dataset_info)

    def _setup_lora(self):
        """Apply LoRA adapters to the model."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            logger.info("LoRA adapters applied successfully")
        except ImportError:
            logger.warning("peft not installed, skipping LoRA")

    def _init_reference_policy(self) -> None:
        """Initialize a stable reference policy for KL regularization."""
        if self.config.kl_coef <= 0:
            self._reference_mode = "none"
            return

        if not self.config.use_reference_model:
            logger.warning("Reference policy disabled; KL regularization will be ineffective.")
            self._reference_mode = "self_detached"
            return

        # Preferred path for LoRA: temporarily disable adapters and use base model as reference.
        if self.config.use_lora and hasattr(self.model, "disable_adapter"):
            self._reference_mode = "lora_base"
            logger.info("KL reference policy: base model with LoRA adapters disabled")
            return

        # Non-LoRA full-model training on CUDA would require a second large model copy.
        if self.config.device == "cuda":
            logger.warning(
                "Skipping frozen reference model on CUDA to avoid OOM; "
                "KL regularization will be ineffective in this mode."
            )
            self._reference_mode = "self_detached"
            return

        try:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                trust_remote_code=self.config.trust_remote_code,
                torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
                device_map=None,
            )
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)
            self._reference_mode = "frozen_copy"
            logger.info("KL reference policy: frozen model copy initialized")
        except Exception as exc:
            logger.warning(
                f"Failed to initialize frozen reference model ({exc}); "
                "KL regularization will be ineffective."
            )
            self._reference_mode = "self_detached"

    def _build_poi_index(self):
        """Build POI ID to index mapping for fast lookup."""
        if self.poi_df is None:
            return
        self.poi_idx_map = {
            str(pid): idx
            for idx, pid in enumerate(self.poi_df["poi_id"].astype(str).tolist())
        }
        logger.info(f"Built POI index with {len(self.poi_idx_map)} entries")

    def _generate_responses(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        num_responses: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses per prompt.

        Returns:
            (input_ids, attention_mask) including prompt and generated tokens
        """
        outputs = []

        with torch.no_grad():
            for _ in range(num_responses):
                generated = self.model.generate(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                outputs.append(generated)

        # Stack: (batch * num_responses, seq_len)
        return torch.cat(outputs, dim=0)

    def _compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: Optional[PreTrainedModel] = None,
    ) -> torch.Tensor:
        """Compute log probabilities for the given sequences."""
        policy_model = model if model is not None else self.model
        outputs = policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding
        mask = attention_mask[..., 1:].bool()
        per_token_logps = per_token_logps * mask

        return per_token_logps.sum(dim=-1)  # (batch,)

    def _compute_reference_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reference policy log-probs used by KL regularization.

        Modes:
        - lora_base: disable LoRA adapters to get base-policy log-probs
        - frozen_copy: use a frozen reference model
        - self_detached: degrade gracefully (no effective KL signal)
        """
        if self._reference_mode == "lora_base":
            disable_ctx = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else nullcontext()
            was_training = self.model.training
            with torch.no_grad(), disable_ctx:
                self.model.eval()
                ref = self._compute_log_probs(input_ids, attention_mask, model=self.model)
            if was_training:
                self.model.train()
            return ref

        if self.ref_model is not None:
            with torch.no_grad():
                return self._compute_log_probs(input_ids, attention_mask, model=self.ref_model)

        # Fallback path: keep training running, but KL term becomes ~0.
        with torch.no_grad():
            return self._compute_log_probs(input_ids, attention_mask, model=self.model).detach()

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int,
    ) -> torch.Tensor:
        """
        Compute group-relative advantages.

        For each group of N responses:
        - Compute group mean reward
        - Advantage = reward - group_mean
        - Normalize across batch

        This eliminates the need for a value network.
        """
        batch_size = rewards.shape[0]
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if batch_size % group_size != 0:
            raise ValueError(
                f"Reward batch size ({batch_size}) must be divisible by group_size ({group_size})"
            )
        num_groups = batch_size // group_size

        # Reshape: (num_groups, group_size)
        grouped_rewards = rewards.view(num_groups, group_size)

        # Group mean as baseline
        group_means = grouped_rewards.mean(dim=1, keepdim=True)

        # Advantages
        advantages = grouped_rewards - group_means

        # Normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten back
        return advantages.view(batch_size)

    def _decode_generated_only_texts(
        self,
        generated_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> List[str]:
        """Decode generated continuations only (exclude prompt tokens)."""
        prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
        texts: List[str] = []
        for row_ids, prompt_len in zip(generated_ids, prompt_lengths):
            continuation_ids = row_ids[int(prompt_len):]
            texts.append(self.tokenizer.decode(continuation_ids, skip_special_tokens=True))
        return texts

    @staticmethod
    def _expand_reward_metadata(
        metadata_list: List[Dict],
        target_poi_list: List[str],
        group_size: int,
    ) -> List[Dict]:
        """
        Expand per-prompt metadata to per-sample metadata with target labels attached.
        """
        expanded: List[Dict] = []
        for meta, target_poi in zip(metadata_list, target_poi_list):
            merged = dict(meta)
            if target_poi is not None and target_poi != "":
                merged["target_next_poi"] = str(target_poi)
            for _ in range(group_size):
                expanded.append(dict(merged))
        return expanded

    def _compute_rewards(
        self,
        generated_texts: List[str],
        metadata_list: List[Dict],
    ) -> List[float]:
        """
        Compute rewards for generated responses.

        Reward components:
        1. Correctness: Is the output a valid POI ID format? (0 or 1)
        2. Target Match: Does it match the target POI? (+2.0)
        3. Format Validity: Is it in valid POI ID format? (+0.5)
        4. Advanced (if POI data available): feasibility, preference, diversity
        """
        rewards = []

        for text, meta in zip(generated_texts, metadata_list):
            try:
                # Extract POI ID from text
                poi_id = self._extract_poi_id(text)

                # Initialize reward
                reward = 0.0

                # 1. Format validity reward
                if poi_id and len(poi_id) > 0:
                    reward += 0.5  # Valid POI ID format

                    # 2. Target matching reward (if available)
                    target_poi = meta.get("target_next_poi")
                    if target_poi and poi_id == str(target_poi):
                        reward += 2.0  # Perfect match

                    # 3. Check if POI exists in database
                    if self.poi_idx_map and poi_id in self.poi_idx_map:
                        reward += 0.5  # POI exists in database
                    elif self.poi_idx_map:
                        reward -= 0.5  # Invalid POI (doesn't exist)

                    # 4. Advanced rewards (if full data available)
                    if self.poi_df is not None and self.time_matrix is not None:
                        adv_reward = self._compute_advanced_reward(
                            poi_id, meta, text
                        )
                        reward += adv_reward
                else:
                    # Invalid output format
                    reward = -1.0

                rewards.append(reward)

            except Exception as e:
                logger.debug(f"Failed to compute reward: {e}")
                rewards.append(-1.0)

        return rewards

    def _compute_advanced_reward(
        self,
        poi_id: str,
        metadata: Dict,
        generated_text: str,
    ) -> float:
        """
        Compute advanced reward using POI data and time matrix.

        This requires poi_df and time_matrix to be loaded.
        """
        if self.poi_df is None or self.time_matrix is None:
            return 0.0

        try:
            state_prefix = metadata.get("state_prefix", [])
            if not state_prefix or poi_id not in self.poi_idx_map:
                return 0.0

            # Build route for reward computation
            route = state_prefix + [poi_id]

            # Use RewardManager for comprehensive scoring
            scores = self.reward_manager.score_route(
                route_poi_ids=route,
                poi_df=self.poi_df,
                time_matrix=self.time_matrix,
                start_time_min=480,  # 8:00 AM
                end_time_min=1320,   # 10:00 PM
            )

            # Normalize reward to [-1, 1] range
            # Total reward can be negative for infeasible routes
            total_reward = scores.get("total_reward", 0.0)

            # Scale to reasonable bonus range
            return np.tanh(total_reward / 10.0)  # Squash to [-1, 1]

        except Exception as e:
            logger.debug(f"Advanced reward computation failed: {e}")
            return 0.0

    def _extract_poi_id(self, text: str) -> Optional[str]:
        """
        Extract POI ID from generated text.

        Handles formats:
        - S-prefixed IDs: S123456
        - JSON: {"poi_id": "S123456"}
        - Plain text: "S123456" or "123456"
        """
        # Try JSON parsing first
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

        # Try S-prefixed ID pattern (S123456)
        import re
        s_match = re.search(r'S\d+', text, re.IGNORECASE)
        if s_match:
            return s_match.group(0).upper()

        # Try plain numeric ID
        num_match = re.search(r'\d{4,}', text)  # At least 4 digits
        if num_match:
            return num_match.group(0)

        return None

    def train(self):
        """Run GRPO training loop."""
        logger.info("Starting GRPO training...")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        num_epochs = self.config.num_train_epochs

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
                # Move to device
                prompt_ids = batch["prompt_input_ids"].to(self.device)
                prompt_mask = batch["prompt_attention_mask"].to(self.device)

                # Generate group of responses
                group_size = self.config.group_size
                expanded_prompt_ids = prompt_ids.repeat_interleave(group_size, dim=0)
                expanded_prompt_mask = prompt_mask.repeat_interleave(group_size, dim=0)
                all_responses = self._generate_responses(
                    expanded_prompt_ids,
                    expanded_prompt_mask,
                    num_responses=1,  # Already repeated
                )

                # Compute reference log probs (before update)
                ref_log_probs = self._compute_reference_log_probs(
                    all_responses,
                    all_responses != self.tokenizer.pad_token_id,
                )

                # Decode generated text for reward computation
                generated_texts = self._decode_generated_only_texts(
                    all_responses,
                    expanded_prompt_mask,
                )
                reward_metadata = self._expand_reward_metadata(
                    batch["metadata"],
                    batch["target_poi"],
                    group_size,
                )

                # Compute rewards
                rewards = torch.tensor(
                    self._compute_rewards(generated_texts, reward_metadata),
                    dtype=torch.float32,
                    device=self.device,
                )

                # Compute advantages (GRPO)
                advantages = self._compute_advantages(rewards, group_size)

                # Compute current log probs
                current_log_probs = self._compute_log_probs(
                    all_responses,
                    all_responses != self.tokenizer.pad_token_id,
                )

                # Policy loss: -log_prob * advantage
                policy_loss = -(current_log_probs * advantages).mean()

                # KL penalty (against reference)
                kl_penalty = (current_log_probs - ref_log_probs).mean()
                kl_loss = self.config.kl_coef * kl_penalty

                # Total loss
                loss = policy_loss + kl_loss

                # Backward
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                did_step = False
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    did_step = True
                    mean_reward = rewards.mean().item()
                    self.best_reward = max(self.best_reward, mean_reward)

                # Logging
                if did_step and self.global_step % self.config.logging_steps == 0:
                    metrics = {
                        "loss": loss.item(),
                        "policy_loss": policy_loss.item(),
                        "kl_penalty": kl_penalty.item(),
                        "mean_reward": rewards.mean().item(),
                        "std_reward": rewards.std().item(),
                    }
                    logger.info(
                        f"Step {self.global_step}: "
                        f"loss={metrics['loss']:.4f}, "
                        f"policy={metrics['policy_loss']:.4f}, "
                        f"kl={metrics['kl_penalty']:.4f}, "
                        f"mean_reward={metrics['mean_reward']:.4f}"
                    )

                    # Log to experiment tracker
                    if self.experiment_tracker:
                        self.experiment_tracker.log_metrics(metrics, step=self.global_step)

                # Save checkpoint
                if did_step and self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()

        logger.info("Training completed!")
        # 始终保存最终权重，避免 save_steps 未触发时输出目录不存在
        self._save_checkpoint()

        # Finalize experiment tracking
        if self.experiment_tracker:
            # Log final metrics
            final_metrics = {
                "final_best_reward": self.best_reward,
                "final_global_step": self.global_step,
            }
            self.experiment_tracker.log_metrics(final_metrics)

            # Log model
            self.experiment_tracker.log_model(
                self.config.output_dir,
                name="grpo_model_final",
                model_type="huggingface"
            )

            self.experiment_tracker.finish(status="FINISHED")

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for dataloader."""
        prompt_ids = [item["prompt_input_ids"] for item in batch]
        prompt_mask = [item["prompt_attention_mask"] for item in batch]
        target_poi = [item.get("target_poi", "") for item in batch]

        return {
            "prompt_input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(ids) for ids in prompt_ids],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            ),
            "prompt_attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(mask) for mask in prompt_mask],
                batch_first=True,
                padding_value=0,
            ),
            "metadata": [item["metadata"] for item in batch],
            "target_poi": target_poi,
        }

    def _save_checkpoint(self):
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if self.config.use_lora:
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir / "checkpoint")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)

        # Save config
        config_path = output_dir / "grpo_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "global_step": self.global_step,
                "best_reward": self.best_reward,
            }, f, indent=2)

        logger.info(f"Checkpoint saved to {output_dir}")

        # Log to experiment tracker (only on first checkpoint)
        if self.experiment_tracker and self.global_step == self.config.save_steps:
            self.experiment_tracker.log_model(
                str(output_dir),
                name="grpo_model",
                model_type="huggingface"
            )
            self.experiment_tracker.log_artifact(str(config_path))


def main():
    """CLI entry point for GRPO training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train route planning policy with GRPO")
    parser.add_argument("--model", type=str, default="models/Qwen3-8B",
                        help="Base model path")
    parser.add_argument("--data", type=str,
                        default="outputs/datasets/grpo_planner_prompts.jsonl",
                        help="Training data path")
    parser.add_argument("--output", type=str,
                        default="outputs/grpo/qwen3-8b-planner",
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--group-size", type=int, default=4,
                        help="Number of samples per prompt for GRPO")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum generated tokens per sampled response")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log metrics every N optimization steps")
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Use LoRA for efficient training")
    parser.add_argument("--no-lora", dest="use_lora", action="store_false",
                        help="Disable LoRA and train full model")

    # Optional: POI data for advanced reward computation
    parser.add_argument("--poi-data", type=str, default="data/processed/pois_with_embeddings.parquet",
                        help="Path to POI data for advanced reward computation")
    parser.add_argument("--time-matrix", type=str, default="data/processed/time_matrix.npy",
                        help="Path to time matrix for feasibility checking")

    args = parser.parse_args()

    config = GRPOConfig(
        model_name_or_path=args.model,
        train_data=args.data,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        group_size=args.group_size,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        logging_steps=args.logging_steps,
        use_lora=args.use_lora,
    )

    # Load optional POI data for advanced rewards
    poi_df = None
    time_matrix = None

    if args.poi_data and Path(args.poi_data).exists():
        try:
            logger.info(f"Loading POI data from {args.poi_data}")
            poi_df = pd.read_parquet(args.poi_data)
            logger.info(f"Loaded {len(poi_df)} POIs")

            if args.time_matrix and Path(args.time_matrix).exists():
                logger.info(f"Loading time matrix from {args.time_matrix}")
                time_matrix = np.load(args.time_matrix)
                logger.info(f"Loaded time matrix with shape {time_matrix.shape}")
        except Exception as e:
            logger.warning(f"Failed to load POI data: {e}. Using basic rewards.")

    trainer = GRPOTrainer(config, poi_df=poi_df, time_matrix=time_matrix)
    trainer.train()


if __name__ == "__main__":
    main()
