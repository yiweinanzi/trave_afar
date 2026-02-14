#!/usr/bin/env python3
"""
GRPO模型评测脚本

评测Group Relative Policy Optimization训练后模型的路线规划效果：
1. 下一步POI预测准确率
2. 完整路线可行性
3. 奖励分数对比
4. 与基准模型对比
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from tqdm import tqdm


def load_grpo_prompts(jsonl_path: str, max_samples: int = None) -> List[Dict]:
    """加载GRPO提示数据"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break
    return data


def load_poi_data(poi_csv: str = "data/all/poi_expanded.csv") -> pd.DataFrame:
    """加载POI数据"""
    return pd.read_csv(poi_csv, low_memory=False)


class GRPOModelEvaluator:
    """GRPO模型评测器"""

    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载GRPO训练后的模型"""
        print(f"加载GRPO模型: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model_path = self.model_path
        is_lora = False

        if (Path(self.model_path) / "adapter_config.json").exists():
            is_lora = True
            with open(Path(self.model_path) / "adapter_config.json") as f:
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

        if is_lora:
            base = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
            self.model = PeftModel.from_pretrained(base, self.model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)

        self.model.eval()
        print(f"✓ 模型已加载到 {self.device}")

    def predict_next_poi(self, state_prefix: List[str], instruction: str = "") -> str:
        """预测下一个POI"""
        # 构建提示
        prompt = f"当前已选景点: {', '.join(state_prefix)}\n"
        if instruction:
            prompt += f"{instruction}\n"
        prompt += "请预测下一个最合适的景点POI ID:"

        chat_template = Template(
            "<|im_start|>system\n你是一个旅游路线规划助手。<|im_end|>\n"
            "<|im_start|>user\n{{ prompt }}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        text = chat_template.render(prompt=prompt)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # 提取POI ID
        # 响应可能是 "S123456" 或 "推荐: S123456"
        for token in response.split():
            token = token.strip(" ,.:;，。：；")
            if token.startswith("S") and len(token) >= 4:
                return token
            if token.isdigit() and len(token) >= 4:
                return token

        return response.split()[0] if response.split() else ""


class RouteRewardCalculator:
    """路线奖励计算器"""

    def __init__(self, poi_df: pd.DataFrame):
        self.poi_df = poi_df
        self.poi_set = set(poi_df['poi_id'].astype(str).tolist())

    def calculate_reward(
        self,
        predicted_route: List[str],
        target_route: List[str],
        state_prefix: List[str]
    ) -> Dict:
        """
        计算路线奖励

        Args:
            predicted_route: 模型预测的完整路线
            target_route: 目标路线
            state_prefix: 当前已选POI前缀

        Returns:
            奖励指标字典
        """
        metrics = {
            "next_poi_correct": 0.0,
            "partial_overlap": 0.0,
            "route_similarity": 0.0,
            "novelty_bonus": 0.0,
            "total_reward": 0.0
        }

        if not predicted_route:
            return metrics

        # 1. 下一个POI是否正确
        if len(predicted_route) > len(state_prefix) and len(target_route) > len(state_prefix):
            pred_next = predicted_route[len(state_prefix)]
            true_next = target_route[len(state_prefix)]
            if pred_next == true_next:
                metrics["next_poi_correct"] = 1.0

        # 2. 路线重叠度
        pred_set = set(predicted_route)
        true_set = set(target_route)

        if true_set:
            overlap = len(pred_set & true_set)
            metrics["partial_overlap"] = overlap / len(true_set)

        # 3. 序列相似度（考虑顺序）
        min_len = min(len(predicted_route), len(target_route))
        matches = sum(1 for i in range(min_len) if predicted_route[i] == target_route[i])
        metrics["route_similarity"] = matches / max_len if min_len > 0 else 0.0

        # 4. 新颖性奖励（预测出目标中没有的POI）
        novel_pois = pred_set - true_set
        valid_novel = novel_pois & self.poi_set
        metrics["novelty_bonus"] = len(valid_novel) * 0.1

        # 总奖励
        metrics["total_reward"] = (
            2.0 * metrics["next_poi_correct"] +
            1.0 * metrics["partial_overlap"] +
            0.5 * metrics["route_similarity"] +
            metrics["novelty_bonus"]
        )

        return metrics


def evaluate_grpo_model(
    model_path: str,
    test_data_path: str = "outputs/datasets/grpo_planner_prompts.jsonl",
    poi_csv: str = "data/all/poi_expanded.csv",
    max_samples: int = 100,
    use_baseline: bool = False
) -> Dict:
    """
    评测GRPO模型

    Args:
        model_path: GRPO模型路径
        test_data_path: 测试数据路径
        poi_csv: POI数据路径
        max_samples: 最大评测样本数
        use_baseline: 是否使用基准策略对比

    Returns:
        评测指标字典
    """
    print("=" * 80)
    print("GRPO模型评测 - 路线规划效果")
    print("=" * 80)

    # 加载数据
    print(f"\n加载测试数据: {test_data_path}")
    test_data = load_grpo_prompts(test_data_path, max_samples)
    print(f"  测试样本: {len(test_data)}")

    print(f"\n加载POI数据: {poi_csv}")
    poi_df = load_poi_data(poi_csv)
    print(f"  POI数量: {len(poi_df)}")

    # 加载模型
    if model_path and Path(model_path).exists():
        print(f"\n加载GRPO模型: {model_path}")
        evaluator = GRPOModelEvaluator(model_path)
        evaluator.load_model()
    else:
        print(f"\n使用基准策略（无模型）")
        evaluator = None

    reward_calculator = RouteRewardCalculator(poi_df)

    # 评测指标
    metrics = {
        "next_poi_accuracy": 0.0,
        "avg_reward": 0.0,
        "avg_partial_overlap": 0.0,
        "avg_route_similarity": 0.0,
        "valid_predictions": 0,
        "total_samples": 0
    }

    all_rewards = []
    overlaps = []
    similarities = []
    next_correct = 0

    print("\n评测中...")
    for item in tqdm(test_data):
        prompt_data = json.loads(item.get("prompt", "{}"))
        state_prefix = prompt_data.get("state_prefix", [])
        target_next = item.get("target_next_poi", "")
        full_route = item.get("full_target_route", state_prefix + [target_next])

        if not state_prefix:
            continue

        metrics["total_samples"] += 1

        # 预测下一个POI
        if evaluator:
            instruction = prompt_data.get("instruction", "")
            predicted_poi = evaluator.predict_next_poi(state_prefix, instruction)
        else:
            # 基准策略：随机选择
            all_pois = poi_df['poi_id'].astype(str).tolist()
            predicted_poi = np.random.choice(all_pois) if all_pois else ""

        # 构建预测路线
        predicted_route = state_prefix + [predicted_poi]

        # 计算奖励
        reward_metrics = reward_calculator.calculate_reward(
            predicted_route, full_route, state_prefix
        )

        all_rewards.append(reward_metrics["total_reward"])
        overlaps.append(reward_metrics["partial_overlap"])
        similarities.append(reward_metrics["route_similarity"])
        next_correct += reward_metrics["next_poi_correct"]

        if predicted_poi:
            metrics["valid_predictions"] += 1

    # 计算平均指标
    n = metrics["total_samples"]
    if n > 0:
        metrics["next_poi_accuracy"] = next_correct / n
        metrics["avg_reward"] = np.mean(all_rewards)
        metrics["avg_partial_overlap"] = np.mean(overlaps)
        metrics["avg_route_similarity"] = np.mean(similarities)

    return metrics


def print_grpo_report(results: Dict):
    """打印GRPO评测报告"""
    print("\n" + "=" * 80)
    print("GRPO评测报告")
    print("=" * 80)

    print("\n【路线规划指标】")
    print(f"  下一POI准确率: {results['next_poi_accuracy']:.2%}")
    print(f"  平均奖励: {results['avg_reward']:.4f}")
    print(f"  平均路线重叠度: {results['avg_partial_overlap']:.2%}")
    print(f"  平均路线相似度: {results['avg_route_similarity']:.2%}")
    print(f"  有效预测率: {results['valid_predictions']}/{results['total_samples']}")

    # 分析
    print("\n【分析】")
    if results['next_poi_accuracy'] >= 0.5:
        print("  ✓ 模型路线规划能力良好")
    elif results['next_poi_accuracy'] >= 0.3:
        print("  ⚠ 模型有一定规划能力，但需继续训练")
    else:
        print("  ✗ 模型规划能力较弱，建议检查训练配置")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='GRPO模型评测')
    parser.add_argument('--model', type=str, default='outputs/grpo/qwen3-8b-grpo',
                        help='GRPO模型路径（留空使用基准策略）')
    parser.add_argument('--test-data', type=str, default='outputs/datasets/grpo_planner_prompts.jsonl',
                        help='GRPO测试数据路径')
    parser.add_argument('--poi-csv', type=str, default='data/all/poi_expanded.csv',
                        help='POI数据路径')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='最大评测样本数')
    parser.add_argument('--output', type=str, default='outputs/evaluation/grpo_eval.json',
                        help='评测结果输出路径')
    args = parser.parse_args()

    results = evaluate_grpo_model(
        model_path=args.model,
        test_data_path=args.test_data,
        poi_csv=args.poi_csv,
        max_samples=args.max_samples
    )

    # 打印报告
    print_grpo_report(results)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
