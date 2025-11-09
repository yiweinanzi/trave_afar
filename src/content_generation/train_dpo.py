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

def prepare_preference_data(prefs_csv='outputs/dpo/prefs.csv'):
    """
    准备偏好数据
    
    Args:
        prefs_csv: 偏好数据CSV文件路径
    
    Returns:
        Dataset: HuggingFace Dataset
    """
    os.makedirs(os.path.dirname(prefs_csv), exist_ok=True)
    
    # 如果文件不存在，创建示例数据
    if not os.path.exists(prefs_csv):
        print(f"偏好数据文件不存在，创建示例数据: {prefs_csv}")
        sample_data = [
            {
                'prompt': '给"古城+夜景+步行少"行程写标题',
                'chosen': '西安古城轻走｜夜景串游 4h 不卡点',
                'rejected': '某地城市旅游路线推荐 标题一'
            },
            {
                'prompt': '给"湖泊+拍照+轻松"行程写标题',
                'chosen': '天山南北｜喀纳斯湖-赛里木湖，镜面天空',
                'rejected': '新疆旅游路线推荐'
            },
            {
                'prompt': '给"雪山+徒步+深度游"行程写标题',
                'chosen': '雪域高原｜珠峰大本营-纳木错，触摸世界之巅',
                'rejected': '西藏旅游路线'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_csv(prefs_csv, index=False, encoding='utf-8')
        print(f"✓ 创建示例数据: {len(df)} 条")
    
    # 加载数据
    df = pd.read_csv(prefs_csv)
    print(f"✓ 加载偏好数据: {len(df)} 条")
    
    # 转换为HuggingFace Dataset格式
    dataset = Dataset.from_pandas(df)
    
    return dataset

def train_dpo(
    model_name=None,
    prefs_csv='outputs/dpo/prefs.csv',
    output_dir='outputs/dpo/run',
    use_lora=True,
    lora_r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    learning_rate=1e-5,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    beta=0.1,
    max_length=256,
    use_gpu=True
):
    """
    训练DPO模型
    
    Args:
        model_name: 基础模型名称
        prefs_csv: 偏好数据CSV文件
        output_dir: 输出目录
        use_lora: 是否使用LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        learning_rate: 学习率
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每设备批次大小
        gradient_accumulation_steps: 梯度累积步数
        beta: DPO温度参数
        max_length: 最大序列长度
        use_gpu: 是否使用GPU
    """
    print("="*80)
    print("DPO训练 - 文案生成偏好对齐")
    print("="*80)
    
    # 检查GPU
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # 确定模型路径
    if model_name is None:
        # 检查本地模型路径
        local_model_path = "/root/autodl-tmp/goafar_project/models/models--Qwen--Qwen3-8B"
        if os.path.exists(local_model_path):
            model_name = local_model_path
            print(f"使用本地模型: {model_name}")
        else:
            model_name = "Qwen/Qwen3-8B"
            print(f"使用HuggingFace模型: {model_name}")
    
    # 准备数据
    dataset = prepare_preference_data(prefs_csv)
    
    # 加载tokenizer
    print(f"\n加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="/root/autodl-tmp/goafar_project/models"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"\n加载模型: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "cache_dir": "/root/autodl-tmp/goafar_project/models"
    }
    
    if use_gpu and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # 准备参考模型（DPO需要）
    if use_lora:
        ref_model = None  # LoRA模式下不需要显式参考模型
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    
    # LoRA配置
    peft_config = None
    if use_lora:
        print(f"\n配置LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 准备模型用于LoRA训练
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # 数据预处理
    def preprocess_function(examples):
        """预处理偏好数据"""
        prompts = examples['prompt']
        chosen = examples['chosen']
        rejected = examples['rejected']
        
        # 构建prompt-chosen和prompt-rejected对
        chosen_texts = [f"{p}\n{chosen[i]}" for i, p in enumerate(prompts)]
        rejected_texts = [f"{p}\n{rejected[i]}" for i, p in enumerate(prompts)]
        
        # Tokenize
        chosen_tokens = tokenizer(
            chosen_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        rejected_tokens = tokenizer(
            rejected_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"]
        }
    
    # 预处理数据集
    print("\n预处理数据集...")
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        eval_strategy="no",
        bf16=use_gpu and torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        report_to="none"
    )
    
    # DPO配置
    dpo_config = DPOConfig(
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        loss_type="sigmoid"
    )
    
    # 初始化DPO训练器
    print("\n初始化DPO训练器...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2
    )
    
    # 训练
    print("\n开始训练...")
    trainer.train()
    
    # 保存模型
    print(f"\n保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n✅ DPO训练完成！")
    print(f"模型保存位置: {output_dir}")
    
    return trainer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DPO训练脚本')
    parser.add_argument('--model', type=str, default=None, help='基础模型名称（默认使用本地Qwen3-8B）')
    parser.add_argument('--prefs', type=str, default='outputs/dpo/prefs.csv', help='偏好数据CSV文件')
    parser.add_argument('--output', type=str, default='outputs/dpo/run', help='输出目录')
    parser.add_argument('--use-lora', action='store_true', default=True, help='使用LoRA')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--grad-accum', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO温度参数')
    parser.add_argument('--max-length', type=int, default=256, help='最大序列长度')
    parser.add_argument('--no-gpu', action='store_true', help='不使用GPU')
    
    args = parser.parse_args()
    
    trainer = train_dpo(
        model_name=args.model,
        prefs_csv=args.prefs,
        output_dir=args.output,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        beta=args.beta,
        max_length=args.max_length,
        use_gpu=not args.no_gpu
    )

