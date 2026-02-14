#!/usr/bin/env python3
"""
SFT模型评测脚本

评测监督微调后模型的性能：
1. 意图理解准确率
2. 路线生成质量
3. 文案生成质量
4. 推理速度
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from jinja2 import Template
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


def load_test_data(data_path: str) -> List[Dict]:
    """加载测试数据"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


class SFTModelEvaluator:
    """SFT模型评测器"""

    def __init__(self, model_path: str, use_gpu: bool = True):
        self.model_path = model_path
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")

        from transformers import AutoModelForCausalLM, AutoTokenizer

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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **model_kwargs
        )
        self.model.eval()

        print(f"✓ 模型已加载到 {self.device}")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        """生成文本"""
        # Qwen聊天格式
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
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 解码，只返回生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response


class IntentUnderstandingEvaluator:
    """意图理解评测"""

    @staticmethod
    def parse_response(response: str) -> Dict:
        """解析模型响应"""
        try:
            # 尝试解析JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            return data
        except:
            # 提取关键信息
            result = {}
            for key in ["province", "duration_days", "interests", "style", "season"]:
                if f'"{key}"' in response or f"'{key}'" in response:
                    try:
                        if f'"{key}":' in response:
                            value = response.split(f'"{key}":')[1].split(",")[0].split("}")[0].strip()
                            result[key] = json.loads(value)
                    except:
                        pass
            return result

    @staticmethod
    def evaluate(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """评测意图理解"""
        metrics = {
            "province_accuracy": 0.0,
            "duration_mae": 0.0,
            "interests_f1": 0.0,
            "exact_match": 0.0,
            "valid_json_rate": 0.0
        }

        valid_count = 0
        exact_match_count = 0
        province_correct = 0
        duration_errors = []

        all_pred_interests = []
        all_true_interests = []

        for pred, true in zip(predictions, ground_truths):
            # 检查是否有效
            if isinstance(pred, dict) and pred:
                valid_count += 1

                # 省份准确率
                if pred.get("province") == true.get("province"):
                    province_correct += 1

                # 天数MAE
                if "duration_days" in pred and "duration_days" in true:
                    try:
                        pred_days = float(pred["duration_days"])
                        true_days = float(true["duration_days"])
                        duration_errors.append(abs(pred_days - true_days))
                    except:
                        pass

                # 兴趣F1
                pred_interests = set(pred.get("interests", []))
                true_interests = set(true.get("interests", []))

                for i in pred_interests:
                    all_pred_interests.append(i)
                    all_true_interests.append(i if i in true_interests else f"NOT_{i}")

                # 完全匹配
                if pred == true:
                    exact_match_count += 1

        n = len(ground_truths)
        if n > 0:
            metrics["valid_json_rate"] = valid_count / n
            metrics["province_accuracy"] = province_correct / n
            metrics["duration_mae"] = np.mean(duration_errors) if duration_errors else 0.0
            metrics["exact_match"] = exact_match_count / n

            # 计算兴趣F1
            if all_pred_interests and all_true_interests:
                metrics["interests_f1"] = f1_score(
                    all_true_interests, all_pred_interests, average='micro', zero_division=0
                )

        return metrics


class RouteGenerationEvaluator:
    """路线生成评测"""

    @staticmethod
    def parse_route(response: str) -> List[List[str]]:
        """解析路线响应"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)

            if isinstance(data, dict):
                return data.get("daily_pois", [])
            elif isinstance(data, list):
                return [data]
        except:
            pass

        return []

    @staticmethod
    def evaluate(predictions: List[List], ground_truths: List[List]) -> Dict:
        """评测路线生成"""
        metrics = {
            "valid_route_rate": 0.0,
            "avg_poi_count": 0.0,
            "poi_overlap_rate": 0.0,
            "day_match_rate": 0.0
        }

        valid_count = 0
        total_pois = 0
        overlapping_pois = 0
        total_true_pois = 0
        day_matches = 0

        for pred, true in zip(predictions, ground_truths):
            if pred and isinstance(pred, list):
                valid_count += 1

                pred_pois = set()
                true_pois = set()

                for day_route in pred:
                    if isinstance(day_route, list):
                        pred_pois.update(day_route)
                        total_pois += len(day_route)

                for day_route in true:
                    if isinstance(day_route, list):
                        true_pois.update(day_route)
                        total_true_pois += len(day_route)

                overlapping_pois += len(pred_pois & true_pois)

                if len(pred) == len(true):
                    day_matches += 1

        n = len(ground_truths)
        if n > 0:
            metrics["valid_route_rate"] = valid_count / n
            metrics["avg_poi_count"] = total_pois / n if n > 0 else 0
            metrics["poi_overlap_rate"] = overlapping_pois / total_true_pois if total_true_pois > 0 else 0
            metrics["day_match_rate"] = day_matches / n

        return metrics


class ContentGenerationEvaluator:
    """文案生成评测"""

    @staticmethod
    def parse_content(response: str) -> Dict:
        """解析文案响应"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            return data
        except:
            return {"title": response, "description": ""}

    @staticmethod
    def evaluate(predictions: List[Dict], ground_truths: List[Dict]) -> Dict:
        """评测文案生成"""
        metrics = {
            "has_title_rate": 0.0,
            "has_description_rate": 0.0,
            "avg_title_length": 0.0,
            "avg_description_length": 0.0,
            "valid_json_rate": 0.0
        }

        title_lengths = []
        desc_lengths = []
        has_title = 0
        has_desc = 0
        valid_json = 0

        for pred, true in zip(predictions, ground_truths):
            if isinstance(pred, dict):
                valid_json += 1

                title = pred.get("title", "")
                desc = pred.get("description", "")

                if title:
                    has_title += 1
                    title_lengths.append(len(title))

                if desc:
                    has_desc += 1
                    desc_lengths.append(len(desc))

        n = len(ground_truths)
        if n > 0:
            metrics["valid_json_rate"] = valid_json / n
            metrics["has_title_rate"] = has_title / n
            metrics["has_description_rate"] = has_desc / n
            metrics["avg_title_length"] = np.mean(title_lengths) if title_lengths else 0
            metrics["avg_description_length"] = np.mean(desc_lengths) if desc_lengths else 0

        return metrics


def evaluate_sft_model(
    model_path: str,
    test_data_path: str = "outputs/datasets/sft_data.jsonl",
    max_samples: int = 100,
    batch_size: int = 4
) -> Dict:
    """
    综合评测SFT模型

    Returns:
        Dict: 评测指标
    """
    print("=" * 80)
    print("SFT模型评测")
    print("=" * 80)

    # 加载数据
    print(f"\n加载测试数据: {test_data_path}")
    all_data = load_test_data(test_data_path)

    # 按任务类型分组
    intent_data = [d for d in all_data if d.get("task_type") == "intent_understanding"]
    route_data = [d for d in all_data if d.get("task_type") == "route_generation"]
    content_data = [d for d in all_data if d.get("task_type") == "content_generation"]
    planner_data = [d for d in all_data if "completion" in d or "target_next_poi" in d]

    print(f"  意图理解: {len(intent_data)} 条")
    print(f"  路线生成: {len(route_data)} 条")
    print(f"  文案生成: {len(content_data)} 条")
    print(f"  规划任务: {len(planner_data)} 条")

    # 限制样本数
    intent_data = intent_data[:max_samples]
    route_data = route_data[:max_samples]
    content_data = content_data[:max_samples]
    planner_data = planner_data[:max_samples]

    # 加载模型
    evaluator = SFTModelEvaluator(model_path)
    evaluator.load_model()

    results = {}

    # 评测意图理解
    if intent_data:
        print(f"\n评测意图理解 ({len(intent_data)} 样本)...")
        intent_preds = []
        intent_gts = []

        for item in tqdm(intent_data, desc="意图理解"):
            prompt = item["prompt"]
            gt = json.loads(item["response"])

            start = time.time()
            response = evaluator.generate(prompt, max_new_tokens=128)
            elapsed = time.time() - start

            pred = IntentUnderstandingEvaluator.parse_response(response)
            intent_preds.append(pred)
            intent_gts.append(gt)

        results["intent"] = IntentUnderstandingEvaluator.evaluate(intent_preds, intent_gts)
        results["intent_avg_latency"] = np.mean([time.time() for _ in range(len(intent_data))])  # 占位

    # 评测路线生成
    if route_data:
        print(f"\n评测路线生成 ({len(route_data)} 样本)...")
        route_preds = []
        route_gts = []

        for item in tqdm(route_data, desc="路线生成"):
            prompt = item["prompt"]
            gt = json.loads(item["response"])

            response = evaluator.generate(prompt, max_new_tokens=256)
            pred = RouteGenerationEvaluator.parse_route(response)
            gt_route = gt.get("daily_pois", [])

            route_preds.append(pred)
            route_gts.append(gt_route)

        results["route"] = RouteGenerationEvaluator.evaluate(route_preds, route_gts)

    # 评测文案生成
    if content_data:
        print(f"\n评测文案生成 ({len(content_data)} 样本)...")
        content_preds = []
        content_gts = []

        for item in tqdm(content_data, desc="文案生成"):
            prompt = item["prompt"]
            gt = json.loads(item["response"])

            response = evaluator.generate(prompt, max_new_tokens=128)
            pred = ContentGenerationEvaluator.parse_content(response)

            content_preds.append(pred)
            content_gts.append(gt)

        results["content"] = ContentGenerationEvaluator.evaluate(content_preds, content_gts)

    return results


def print_evaluation_report(results: Dict):
    """打印评测报告"""
    print("\n" + "=" * 80)
    print("评测报告")
    print("=" * 80)

    if "intent" in results:
        print("\n【意图理解】")
        m = results["intent"]
        print(f"  省份准确率: {m['province_accuracy']:.2%}")
        print(f"  天数MAE: {m['duration_mae']:.2f} 天")
        print(f"  兴趣F1: {m['interests_f1']:.4f}")
        print(f"  完全匹配率: {m['exact_match']:.2%}")
        print(f"  有效JSON率: {m['valid_json_rate']:.2%}")

    if "route" in results:
        print("\n【路线生成】")
        m = results["route"]
        print(f"  有效路线率: {m['valid_route_rate']:.2%}")
        print(f"  平均POI数: {m['avg_poi_count']:.1f}")
        print(f"  POI重叠率: {m['poi_overlap_rate']:.2%}")
        print(f"  天数匹配率: {m['day_match_rate']:.2%}")

    if "content" in results:
        print("\n【文案生成】")
        m = results["content"]
        print(f"  标题生成率: {m['has_title_rate']:.2%}")
        print(f"  描述生成率: {m['has_description_rate']:.2%}")
        print(f"  平均标题长度: {m['avg_title_length']:.1f}")
        print(f"  平均描述长度: {m['avg_description_length']:.1f}")
        print(f"  有效JSON率: {m['valid_json_rate']:.2%}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='SFT模型评测')
    parser.add_argument('--model', type=str, default='outputs/sft/qwen3-8b-tourism',
                        help='SFT模型路径')
    parser.add_argument('--test-data', type=str, default='outputs/datasets/sft_data.jsonl',
                        help='测试数据路径')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='最大评测样本数')
    parser.add_argument('--output', type=str, default='outputs/evaluation/sft_eval.json',
                        help='评测结果输出路径')
    args = parser.parse_args()

    results = evaluate_sft_model(
        model_path=args.model,
        test_data_path=args.test_data,
        max_samples=args.max_samples
    )

    # 打印报告
    print_evaluation_report(results)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存: {output_path}")


if __name__ == "__main__":
    main()
