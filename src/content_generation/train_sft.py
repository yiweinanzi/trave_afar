"""
SFT训练脚本 - 使用QLoRA进行高效微调
使用TRL的SFTTrainer对Qwen3-8B进行监督微调

参考: open_resource/trl-main/examples/scripts/sft.py

任务目标：
1. 意图理解：从用户查询中提取结构化信息
2. 文案生成：生成高质量旅游路线标题和描述
3. POI推荐：根据用户意图推荐合适的POI列表

QLoRA配置：
- 4-bit量化 (NF4)
- LoRA适配器 (r=16, alpha=16)
- 双数量化以进一步减少显存

实验追踪：
- 支持MLflow自动追踪训练参数和指标
- 自动记录模型和训练配置
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# 添加src到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入实验追踪模块
try:
    from src.utils.experiment import (
        MLflowExperiment,
        ExperimentConfig,
        ExperimentManager,
        MLflowCallback,
    )
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow tracking not available. Install with: pip install mlflow")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = Path(os.getenv("GOAFAR_MODEL_CACHE", str(PROJECT_ROOT / "models")))


def _resolve_model_path(model_name: str = None) -> str:
    """解析模型路径"""
    if model_name and Path(model_name).exists():
        return model_name

    env_model_dir = os.getenv("GOAFAR_QWEN_MODEL_DIR", "")
    candidates = []
    if env_model_dir:
        candidates.append(Path(env_model_dir))
    candidates.append(MODEL_CACHE_DIR / "Qwen3-8B")
    candidates.append(MODEL_CACHE_DIR / "models--Qwen--Qwen3-8B")

    for path in candidates:
        if path.exists():
            return str(path)

    return model_name or str(MODEL_CACHE_DIR / "Qwen3-8B")


def load_sft_data(data_path: str, allow_sample_data: bool = False) -> Dataset:
    """
    加载SFT训练数据

    支持的格式：
    1. JSONL: {"prompt": "...", "response": "..."} 或 {"prompt": "...", "completion": "..."}
    2. CSV: prompt, response列
    """
    data_path = Path(data_path)

    if not data_path.exists():
        if allow_sample_data:
            print(f"数据文件不存在: {data_path}")
            print("创建示例数据...")
            return create_sample_data()
        raise FileNotFoundError(
            f"SFT训练数据不存在: {data_path}. "
            "如需使用示例数据，请显式设置 allow_sample_data=True。"
        )

    if data_path.suffix == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"不支持的数据格式: {data_path.suffix}")

    # 确保必要字段存在（支持response或completion字段）
    if "prompt" not in df.columns:
        raise ValueError("数据必须包含 prompt 列")
    if "response" not in df.columns and "completion" not in df.columns:
        raise ValueError("数据必须包含 response 或 completion 列")

    # 统一使用response字段，并兼容 response/completion 混合存在的情况
    if "response" not in df.columns:
        df["response"] = None

    if "completion" in df.columns:
        response_missing = df["response"].isna() | (df["response"].astype(str).str.strip() == "")
        df.loc[response_missing, "response"] = df.loc[response_missing, "completion"]

    # 清理字段，移除空样本，避免出现字符串 "nan" 参与训练
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)
    df.loc[df["prompt"].str.lower() == "nan", "prompt"] = ""
    df.loc[df["response"].str.lower() == "nan", "response"] = ""

    valid_mask = (df["prompt"].str.strip() != "") & (df["response"].str.strip() != "")
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"⚠️ 清理无效SFT样本: {dropped} 条")
    df = df.loc[valid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError("清理后无可用SFT样本，请检查数据集字段内容")

    print(f"✓ 加载SFT数据: {len(df)} 条")
    return Dataset.from_pandas(df)


def create_sample_data() -> Dataset:
    """创建示例SFT数据"""
    sample_data = [
        {
            "prompt": "用户想去新疆看雪山和草原，计划3天",
            "response": json.dumps({
                "province": "新疆",
                "cities": ["阿勒泰", "伊犁"],
                "interests": ["雪山", "草原"],
                "activities": ["拍照", "徒步"],
                "duration_days": 3,
                "season": "夏",
                "style": "摄影游"
            }, ensure_ascii=False)
        },
        {
            "prompt": "给新疆喀纳斯湖禾木村3天游写标题和描述",
            "response": json.dumps({
                "title": "喀纳斯秘境3日｜禾木村落星空下，探寻阿尔泰山的秋日童话",
                "description": "深入喀纳斯湖畔，漫步禾木村原始图瓦木屋，捕捉神态各异的晨雾与晚霞。三天的行程让你充分感受北疆阿勒泰地区的壮美风光，从碧蓝的湖水到金黄的白桦林，每一处都是摄影师的天堂。"
            }, ensure_ascii=False)
        },
        {
            "prompt": "用户想去西藏看布达拉宫，计划2天",
            "response": json.dumps({
                "province": "西藏",
                "cities": ["拉萨"],
                "interests": ["历史文化", "建筑"],
                "activities": ["参观", "朝拜"],
                "duration_days": 2,
                "season": "夏",
                "style": "文化游"
            }, ensure_ascii=False)
        },
        {
            "prompt": "推荐云南丽江古城和大理古镇的3日游",
            "response": json.dumps({
                "province": "云南",
                "cities": ["丽江", "大理"],
                "interests": ["古城", "民族文化"],
                "activities": ["逛街", "拍照"],
                "duration_days": 3,
                "season": "春",
                "style": "休闲游"
            }, ensure_ascii=False)
        },
        {
            "prompt": "用户想去四川看熊猫和吃火锅",
            "response": json.dumps({
                "province": "四川",
                "cities": ["成都"],
                "interests": ["动物", "美食"],
                "activities": ["观赏", "品尝"],
                "duration_days": 2,
                "season": "全年",
                "style": "休闲游"
            }, ensure_ascii=False)
        }
    ]

    df = pd.DataFrame(sample_data)
    print(f"✓ 创建示例SFT数据: {len(df)} 条")
    return Dataset.from_pandas(df)


def train_sft(
    model_name: str = None,
    data_path: str = "outputs/datasets/sft_data.jsonl",
    output_dir: str = "outputs/sft/qwen3-8b-tourism",
    use_qlora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_grad_norm: float = 0.3,
    max_seq_length: int = 512,
    use_gpu: bool = True,
    allow_sample_data: bool = False,
    # 实验追踪参数
    use_mlflow: bool = True,
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: str = "sft_tourism",
):
    """
    使用QLoRA训练SFT模型

    Args:
        model_name: 基础模型名称或路径
        data_path: 训练数据路径
        output_dir: 输出目录
        use_qlora: 是否使用QLoRA (4-bit量化 + LoRA)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 学习率
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每设备批次大小
        gradient_accumulation_steps: 梯度累积步数
        max_grad_norm: 梯度裁剪阈值
        max_seq_length: 最大序列长度
        use_gpu: 是否使用GPU
        use_mlflow: 是否启用MLflow实验追踪
        mlflow_tracking_uri: MLflow服务器地址
        mlflow_experiment_name: MLflow实验名称
    """
    print("=" * 80)
    print("SFT训练 - 旅游推荐任务监督微调 (QLoRA)")
    print("=" * 80)

    # 初始化实验追踪
    experiment_tracker = None
    if use_mlflow and MLFLOW_AVAILABLE:
        print(f"初始化MLflow实验追踪: {mlflow_experiment_name}")
        config = ExperimentConfig(
            enabled=True,
            tracking_uri=mlflow_tracking_uri,
            tags={"task": "sft", "model": "Qwen3-8B"},
        )
        manager = ExperimentManager(
            mlflow_tracking_uri=mlflow_tracking_uri,
            config=config,
        )
        experiment_tracker = manager.create_experiment(mlflow_experiment_name)
        if hasattr(experiment_tracker, 'start_run'):
            experiment_tracker.start_run()

    # 检查GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 确定模型路径
    model_path = _resolve_model_path(model_name)
    if Path(model_path).exists():
        print(f"使用本地模型: {model_path}")
    else:
        print(f"使用HuggingFace模型: {model_path}")

    # 加载数据
    dataset = load_sft_data(data_path, allow_sample_data=allow_sample_data)

    # 记录数据集信息
    if experiment_tracker:
        dataset_info = {
            "data_path": data_path,
            "num_samples": len(dataset),
            "max_seq_length": max_seq_length,
        }
        experiment_tracker.log_dataset(dataset_info)

    # 加载tokenizer
    print(f"\n加载tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 记录训练参数
    if experiment_tracker:
        params = {
            "model_path": model_path,
            "data_path": data_path,
            "output_dir": output_dir,
            "use_qlora": use_qlora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "max_seq_length": max_seq_length,
            "device": device,
        }
        experiment_tracker.log_params(params)

    # QLoRA: 4-bit量化配置
    bnb_config = None
    if use_qlora and use_gpu and torch.cuda.is_available():
        print("\n配置QLoRA: 4-bit量化 (NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # 双数量化，进一步减少显存
        )

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": str(MODEL_CACHE_DIR)
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    elif use_gpu and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs
    )

    # LoRA配置（与是否启用4-bit量化解耦）
    peft_config = None
    print(f"\n配置LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 仅在4-bit量化实际启用时执行k-bit准备
    # LoRA适配器由SFTTrainer根据peft_config注入
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # SFT配置
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=use_gpu and torch.cuda.is_available() and bnb_config is None,
        fp16=False,
        packing=False,
        report_to="none",
        dataset_text_field="text",
    )

    # 数据预处理：组合prompt和response
    def format_prompts(examples):
        """格式化训练样本"""
        texts = []
        for prompt, response in zip(examples["prompt"], examples["response"]):
            # 使用tokenizer原生chat template，避免手工模板与模型配置不一致
            messages = [
                {"role": "system", "content": "你是一位专业的旅游规划助手。"},
                {"role": "user", "content": str(prompt)},
                {"role": "assistant", "content": str(response)},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

    # 初始化SFT训练器
    print("\n初始化SFT训练器...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 训练
    print("\n开始训练...")
    trainer.train()

    # 保存模型
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 记录最终指标和模型
    if experiment_tracker:
        # 记录训练日志中的最终指标
        if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
            final_metrics = {}
            for log in trainer.state.log_history[-5:]:  # 最后几条日志
                for k, v in log.items():
                    if isinstance(v, (int, float)) and k != 'epoch':
                        final_metrics[f'final_{k}'] = v
            experiment_tracker.log_metrics(final_metrics)

        # 记录模型
        experiment_tracker.log_model(
            output_dir,
            name="sft_model",
            model_type="huggingface"
        )

        # 记录训练配置
        config_path = Path(output_dir) / "training_config.json"
        training_config = {
            "model_path": model_path,
            "data_path": data_path,
            "output_dir": output_dir,
            "use_qlora": use_qlora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_grad_norm": max_grad_norm,
            "max_seq_length": max_seq_length,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        experiment_tracker.log_artifact(str(config_path))

        # 结束实验
        experiment_tracker.finish(status="FINISHED")

    print("\n✅ SFT训练完成！")
    print(f"模型保存位置: {output_dir}")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SFT训练脚本 - 使用QLoRA进行高效微调')
    parser.add_argument('--model', type=str, default=None, help='基础模型名称（默认使用本地Qwen3-8B）')
    parser.add_argument('--data', type=str, default='outputs/datasets/sft_data.jsonl', help='训练数据路径')
    parser.add_argument('--output', type=str, default='outputs/sft/qwen3-8b-tourism', help='输出目录')
    parser.add_argument('--use-qlora', action='store_true', default=True, help='使用QLoRA (4-bit量化 + LoRA)')
    parser.add_argument('--no-qlora', action='store_false', dest='use_qlora', help='不使用QLoRA')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--grad-accum', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--max-grad-norm', type=float, default=0.3, help='梯度裁剪阈值')
    parser.add_argument('--max-length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--no-gpu', action='store_true', help='不使用GPU')
    parser.add_argument('--allow-sample-data', action='store_true', help='缺失训练数据时允许回退到内置示例数据')

    args = parser.parse_args()

    trainer = train_sft(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        max_seq_length=args.max_length,
        use_gpu=not args.no_gpu,
        allow_sample_data=args.allow_sample_data,
    )
