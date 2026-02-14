"""
DPO训练脚本
使用TRL的DPOTrainer对文案生成模型进行偏好对齐训练

参考: open_resource/trl-main/examples/scripts/dpo.py
"""
import os
import sys
import json
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = Path(os.getenv("GOAFAR_MODEL_CACHE", str(PROJECT_ROOT / "models")))


def _resolve_qwen_model_path() -> str:
    env_model_dir = os.getenv("GOAFAR_QWEN_MODEL_DIR", "")
    candidates = []
    if env_model_dir:
        candidates.append(Path(env_model_dir))
    candidates.append(MODEL_CACHE_DIR / "models--Qwen--Qwen3-8B")
    candidates.append(MODEL_CACHE_DIR / "Qwen3-8B")

    for path in candidates:
        if path.exists():
            return str(path)
    return "Qwen/Qwen3-8B"

def prepare_preference_data(prefs_csv: str = 'outputs/datasets/dpo_prefs.csv') -> Dataset:
    """
    准备偏好数据

    Args:
        prefs_csv: 偏好数据CSV文件路径

    Returns:
        Dataset: HuggingFace Dataset格式，包含prompt, chosen, rejected字段
    """
    # 检查文件是否存在
    if not os.path.exists(prefs_csv):
        raise FileNotFoundError(
            f"偏好数据文件不存在: {prefs_csv}\n"
            f"请先运行数据集构建脚本生成DPO训练数据。"
        )

    # 加载数据
    df = pd.read_csv(prefs_csv)
    print(f"✓ 加载偏好数据: {len(df)} 条")

    # 验证必需的列
    required_columns = ['prompt', 'chosen', 'rejected']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"数据集缺少必需的列: {missing_columns}\n"
            f"当前列: {list(df.columns)}\n"
            f"必需列: {required_columns}"
        )

    # 清理数据 - 移除空值
    df = df.dropna(subset=required_columns)
    print(f"✓ 清理后数据: {len(df)} 条")

    # 转换为HuggingFace Dataset格式
    dataset = Dataset.from_pandas(df)

    return dataset

def train_dpo(
    model_name: str = None,
    prefs_csv: str = 'outputs/datasets/dpo_prefs.csv',
    output_dir: str = 'outputs/dpo/qwen3-8b-dpo',
    use_lora: bool = True,
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    learning_rate: float = 1e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    beta: float = 0.1,
    max_length: int = 512,
    max_prompt_length: int = 256,
    use_gpu: bool = True
) -> DPOTrainer:
    """
    训练DPO模型

    Args:
        model_name: 基础模型名称
        prefs_csv: 偏好数据CSV文件
        output_dir: 输出目录
        use_lora: 是否使用LoRA
        use_qlora: 是否使用QLoRA (4-bit量化)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 学习率
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每设备批次大小
        gradient_accumulation_steps: 梯度累积步数
        beta: DPO温度参数
        max_length: 最大序列长度
        max_prompt_length: 最大prompt长度
        use_gpu: 是否使用GPU

    Returns:
        DPOTrainer: 训练好的训练器
    """
    print("="*80)
    print("DPO训练 - GoAfar推荐系统偏好对齐")
    print("="*80)

    # 检查GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 确定模型路径
    if model_name is None:
        model_name = _resolve_qwen_model_path()
        if Path(model_name).exists():
            print(f"使用本地模型: {model_name}")
        else:
            print(f"使用HuggingFace模型: {model_name}")

    # 准备数据
    print(f"\n加载数据集: {prefs_csv}")
    dataset = prepare_preference_data(prefs_csv)

    # QLoRA配置 (4-bit量化)
    bnb_config = None
    if use_qlora and use_gpu and torch.cuda.is_available():
        print(f"\n配置QLoRA (4-bit量化)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 加载tokenizer
    print(f"\n加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    print(f"\n加载模型: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(MODEL_CACHE_DIR)
    }

    if use_gpu and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if use_qlora:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # LoRA配置
    peft_config = None
    ref_model = None

    if use_lora:
        print(f"\n配置LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
            task_type="CAUSAL_LM"
        )

        if use_qlora:
            # QLoRA模式: 准备模型用于量化训练
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            # QLoRA模式下不需要显式参考模型
            ref_model = None
        else:
            # 标准LoRA模式: 不需要参考模型，DPOTrainer会自动处理
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            ref_model = None
    else:
        # 全量微调模式: 需要参考模型
        print("\n全量微调模式 (加载参考模型...)")
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    # DPO配置
    dpo_config = DPOConfig(
        beta=beta,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=use_gpu and torch.cuda.is_available() and not use_qlora,
        fp16=use_gpu and torch.cuda.is_available() and use_qlora,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        output_dir=output_dir,
        report_to="none",
        remove_unused_columns=False
    )

    # 初始化DPO训练器
    print("\n初始化DPO训练器...")
    print(f"  Beta: {beta}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {per_device_train_batch_size}")
    print(f"  Gradient Accumulation: {gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Max Length: {max_length}")
    print(f"  Max Prompt Length: {max_prompt_length}")

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length
    )

    # 训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练配置
    config = {
        "model_name": model_name,
        "prefs_csv": prefs_csv,
        "use_lora": use_lora,
        "use_qlora": use_qlora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "beta": beta,
        "max_length": max_length,
        "max_prompt_length": max_prompt_length
    }
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ DPO训练完成！")
    print(f"模型保存位置: {output_dir}")
    print(f"训练配置保存位置: {config_path}")
    print("="*80)

    return trainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='DPO训练脚本 - GoAfar推荐系统',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default=None,
                        help='基础模型名称（默认使用本地Qwen3-8B）')
    parser.add_argument('--prefs', type=str, default='outputs/datasets/dpo_prefs.csv',
                        help='偏好数据CSV文件')
    parser.add_argument('--output', type=str, default='outputs/dpo/qwen3-8b-dpo',
                        help='输出目录')
    parser.add_argument('--use-lora', action='store_true', default=True,
                        help='使用LoRA')
    parser.add_argument('--no-lora', action='store_true',
                        help='不使用LoRA (全量微调)')
    parser.add_argument('--use-qlora', action='store_true', default=True,
                        help='使用QLoRA (4-bit量化，节省显存)')
    parser.add_argument('--no-qlora', action='store_true',
                        help='不使用QLoRA (标准LoRA)')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32,
                        help='LoRA alpha (通常设为r的2倍)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='每设备批次大小')
    parser.add_argument('--grad-accum', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta参数 (温度参数，越小越偏好)')
    parser.add_argument('--max-length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--max-prompt-length', type=int, default=256,
                        help='最大prompt长度')
    parser.add_argument('--no-gpu', action='store_true',
                        help='不使用GPU')

    args = parser.parse_args()

    # 处理互斥参数
    use_lora = args.use_lora and not args.no_lora
    use_qlora = args.use_qlora and not args.no_qlora and use_lora

    trainer = train_dpo(
        model_name=args.model,
        prefs_csv=args.prefs,
        output_dir=args.output,
        use_lora=use_lora,
        use_qlora=use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        use_gpu=not args.no_gpu
    )
