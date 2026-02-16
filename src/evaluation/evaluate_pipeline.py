#!/usr/bin/env python3
"""
端到端评测脚本

评测整个GoAfar推荐系统的性能：
1. 意图理解准确率
2. 召回性能 (Recall@K)
3. 排序性能 (NDCG@K)
4. 路线可行性
5. 端到端响应时间
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入新的PipelineEvaluator
from src.evaluation.pipeline_evaluator import (
    PipelineEvaluator,
    generate_evaluation_report,
    save_evaluation_report
)


def load_test_queries(csv_path: str = None) -> List[Dict]:
    """加载测试查询集"""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        queries = []

        def _split_cell(value) -> List[str]:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            text = str(value).strip()
            if not text:
                return []
            return [part.strip() for part in text.split(",") if part.strip()]

        for _, row in df.iterrows():
            days_val = row.get("days", 3)
            try:
                days = int(float(days_val)) if not pd.isna(days_val) else 3
            except Exception:
                days = 3

            queries.append({
                "query": row.get("query", ""),
                "province": row.get("province", ""),
                "days": days,
                "interests": _split_cell(row.get("interests", "")),
                "ground_truth_pois": _split_cell(row.get("ground_truth_pois", "")),
            })
        return queries

    # 默认测试查询
    return [
        {
            "query": "想去新疆看雪山和草原，拍秋天的景色，计划3天",
            "province": "新疆",
            "days": 3,
            "interests": ["雪山", "草原"],
            "ground_truth_pois": []
        },
        {
            "query": "西藏拉萨5日深度游，布达拉宫和纳木错",
            "province": "西藏",
            "days": 5,
            "interests": ["历史文化", "湖泊"],
            "ground_truth_pois": []
        },
        {
            "query": "云南大理洱海2天骑行，轻松休闲",
            "province": "云南",
            "days": 2,
            "interests": ["湖泊", "休闲"],
            "ground_truth_pois": []
        },
        {
            "query": "推荐四川成都的美食和熊猫基地，2天行程",
            "province": "四川",
            "days": 2,
            "interests": ["美食", "动物"],
            "ground_truth_pois": []
        },
        {
            "query": "北京长城和故宫，1天快速游览",
            "province": "北京",
            "days": 1,
            "interests": ["历史", "建筑"],
            "ground_truth_pois": []
        }
    ]


def evaluate_pipeline(
    queries: List[Dict],
    use_llm: bool = False,
    track_recall: bool = True
) -> Dict:
    """
    端到端评测

    Args:
        queries: 查询列表
        use_llm: 是否使用LLM模式
        track_recall: 是否跟踪召回贡献

    Returns:
        评测指标字典
    """
    print("=" * 80)
    print("GoAfar端到端评测")
    print("=" * 80)

    evaluator = PipelineEvaluator()
    evaluator.initialize(use_llm=use_llm)

    return evaluator.evaluate_batch(
        queries=queries,
        track_recall=track_recall,
        track_conversion=True
    )


def print_pipeline_report(evaluation: Dict):
    """打印评测报告"""
    report = generate_evaluation_report(evaluation, include_details=True)
    print("\n" + report)


def save_evaluation_report(evaluation: Dict, output_path: str):
    """保存评测结果"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='端到端评测')
    parser.add_argument('--queries', type=str, default=None,
                        help='测试查询CSV路径（留空使用默认查询）')
    parser.add_argument('--use-llm', action='store_true',
                        help='使用LLM模式（需要模型）')
    parser.add_argument('--output', type=str, default='outputs/evaluation/pipeline_eval.json',
                        help='评测结果输出路径')
    args = parser.parse_args()

    queries = load_test_queries(args.queries)

    evaluation = evaluate_pipeline(queries, use_llm=args.use_llm)

    print_pipeline_report(evaluation)
    save_evaluation_report(evaluation, args.output)


if __name__ == "__main__":
    main()
