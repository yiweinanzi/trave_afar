#!/usr/bin/env python3
"""
DPO模型评测脚本

评测Direct Preference Optimization训练后模型的偏好对齐效果：
1. 偏好对准确率 (chosen vs rejected)
2. 奖励模型分数对比
3. 文案质量对比
"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from jinja2 import Template
from tqdm import tqdm


def load_preference_data(csv_path: str) -> List[Dict]:
    """加载DPO偏好数据"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


class DPOModelEvaluator:
    """DPO模型评测器"""

    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.tokenizer = None
        self.ref_model = None

    def load_model(self, load_ref_model: bool = False):
        """加载模型"""
        print(f"加载DPO模型: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # 加载基础模型
        base_model_path = self.model_path
        if (Path(self.model_path) / "adapter_config.json").exists():
            # LoRA模型，需要加载base
            import yaml
            config_path = Path(self.model_path) / "adapter_config.json"
            with open(config_path) as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-8B")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"trust_remote_code": True}
        if self.use_gpu:
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16

        # 加载训练后的模型
        if (Path(self.model_path) / "adapter_config.json").exists():
            base = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        self.model.eval()

        # 加载参考模型（用于对比）
        if load_ref_model:
            print(f"加载参考模型: {base_model_path}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, **model_kwargs
            )
            self.ref_model.eval()

        print(f"✓ 模型已加载到 {self.device}")

    def compute_logprob(self, prompt: str, response: str) -> float:
        """计算给定响应对数的平均log概率"""
        chat_template = Template(
            "<|im_start|>system\n你是一位专业的旅游规划助手。<|im_end|>\n"
            "<|im_start|>user\n{{ prompt }}<|im_end|>\n"
            "<|im_start|>assistant\n{{ response }}<|im_end|>"
        )

        text = chat_template.render(prompt=prompt, response=response)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits

        # 计算response部分的log概率
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        # 只计算response部分
        start_idx = len(prompt_tokens) + 4  # 考虑特殊token
        end_idx = start_idx + len(response_tokens)

        if end_idx > logits.shape[1]:
            end_idx = logits.shape[1]

        response_logits = logits[0, start_idx:end_idx, :]
        response_ids = inputs["input_ids"][0, start_idx:end_idx]

        log_probs = torch.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze()

        return float(token_log_probs.mean().item())

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """生成文本"""
        chat_template = Template(
            "<|im_start|>system\n你是一位专业的旅游规划助手。<|im_end|>\n"
            "<|im_start|>user\n{{ prompt }}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        text = chat_template.render(prompt=prompt)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response


class RewardModel:
    """简化的奖励模型，用于评估响应质量"""

    @staticmethod
    def score_response(prompt: str, response: str, source: str = "") -> float:
        """
        计算响应质量分数

        评分维度：
        - 相关性：是否回答了问题
        - 完整性：是否包含必要信息
        - 可读性：格式是否清晰
        """
        score = 0.0

        # 相关性：响应不为空
        if response and len(response) > 5:
            score += 0.3

        # 完整性：包含足够的内容
        if len(response) > 20:
            score += 0.2

        # 格式：是否结构化
        if "第" in response or "、" in response or "," in response:
            score += 0.2

        # 质量关键词
        quality_words = ["推荐", "精华", "深度", "体验", "经典", "必游", "精选"]
        if any(word in response for word in quality_words):
            score += 0.15

        # 避免低质量词
        bad_words = ["不知道", "无法", "错误", "抱歉"]
        if any(word in response for word in bad_words):
            score -= 0.3

        return max(0.0, min(1.0, score))


def evaluate_dpo_model(
    model_path: str,
    test_data_path: str = "outputs/datasets/dpo_prefs.csv",
    max_samples: int = 100
) -> Dict:
    """
    评测DPO模型

    Returns:
        Dict: 评测指标
    """
    print("=" * 80)
    print("DPO模型评测 - 偏好对齐效果")
    print("=" * 80)

    # 加载数据
    print(f"\n加载测试数据: {test_data_path}")
    data = load_preference_data(test_data_path)
    data = data[:max_samples]

    print(f"  测试样本: {len(data)} 对")

    # 统计来源
    sources = {}
    for item in data:
        source = item.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1

    print(f"  数据来源: {sources}")

    # 加载模型
    evaluator = DPOModelEvaluator(model_path)
    evaluator.load_model()

    reward_model = RewardModel()

    # 评测指标
    metrics = {
        "chosen_accuracy": 0.0,  # 模型是否偏好chosen
        "chosen_reward_higher_rate": 0.0,  # chosen奖励分数是否更高
        "avg_reward_diff": 0.0,
        "by_source": {}
    }

    chosen_correct = 0
    chosen_reward_higher = 0
    reward_diffs = []
    source_metrics = {}

    print("\n评测中...")
    for item in tqdm(data):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        source = item.get("source", "unknown")

        # 计算log概率
        try:
            chosen_logprob = evaluator.compute_logprob(prompt, chosen)
            rejected_logprob = evaluator.compute_logprob(prompt, rejected)

            # 模型应该给chosen更高的概率
            if chosen_logprob > rejected_logprob:
                chosen_correct += 1

            # 计算奖励分数
            chosen_reward = reward_model.score_response(prompt, chosen, source)
            rejected_reward = reward_model.score_response(prompt, rejected, source)

            if chosen_reward > rejected_reward:
                chosen_reward_higher += 1

            reward_diffs.append(chosen_reward - rejected_reward)

            # 按来源统计
            if source not in source_metrics:
                source_metrics[source] = {
                    "correct": 0, "total": 0,
                    "reward_correct": 0, "reward_diffs": []
                }

            source_metrics[source]["total"] += 1
            if chosen_logprob > rejected_logprob:
                source_metrics[source]["correct"] += 1
            if chosen_reward > rejected_reward:
                source_metrics[source]["reward_correct"] += 1
            source_metrics[source]["reward_diffs"].append(chosen_reward - rejected_reward)

        except Exception as e:
            print(f"\n警告: 评测失败 - {e}")
            continue

    n = len(data)
    if n > 0:
        metrics["chosen_accuracy"] = chosen_correct / n
        metrics["chosen_reward_higher_rate"] = chosen_reward_higher / n
        metrics["avg_reward_diff"] = np.mean(reward_diffs)

        # 按来源统计
        for source, stats in source_metrics.items():
            if stats["total"] > 0:
                metrics["by_source"][source] = {
                    "accuracy": stats["correct"] / stats["total"],
                    "reward_accuracy": stats["reward_correct"] / stats["total"],
                    "avg_reward_diff": np.mean(stats["reward_diffs"])
                }

    return metrics


def print_dpo_report(results: Dict):
    """打印DPO评测报告"""
    print("\n" + "=" * 80)
    print("DPO评测报告")
    print("=" * 80)

    print("\n【整体指标】")
    print(f"  偏好对齐准确率: {results['chosen_accuracy']:.2%}")
    print(f"  奖励分数准确率: {results['chosen_reward_higher_rate']:.2%}")
    print(f"  平均奖励差: {results['avg_reward_diff']:.4f}")

    if results.get("by_source"):
        print("\n【按来源统计】")
        for source, metrics in results["by_source"].items():
            print(f"\n  {source}:")
            print(f"    偏好准确率: {metrics['accuracy']:.2%}")
            print(f"    奖励准确率: {metrics['reward_accuracy']:.2%}")
            print(f"    奖励差: {metrics['avg_reward_diff']:.4f}")

    # 分析
    print("\n【分析】")
    if results['chosen_accuracy'] >= 0.7:
        print("  ✓ 模型偏好对齐效果良好，DPO训练成功")
    elif results['chosen_accuracy'] >= 0.5:
        print("  ⚠ 模型有一定的偏好对齐效果，但仍有改进空间")
    else:
        print("  ✗ 模型偏好对齐效果不佳，需要检查训练配置")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='DPO模型评测')
    parser.add_argument('--model', type=str, default='outputs/dpo/qwen3-8b-dpo',
                        help='DPO模型路径')
    parser.add_argument('--test-data', type=str, default='outputs/datasets/dpo_prefs.csv',
                        help='测试偏好数据路径')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='最大评测样本数')
    parser.add_argument('--output', type=str, default='outputs/evaluation/dpo_eval.json',
                        help='评测结果输出路径')
    args = parser.parse_args()

    results = evaluate_dpo_model(
        model_path=args.model,
        test_data_path=args.test_data,
        max_samples=args.max_samples
    )

    # 打印报告
    print_dpo_report(results)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
