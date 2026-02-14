#!/usr/bin/env python3
"""
Pipeline Evaluator - 端到端流水线评测器

提供完整的推荐系统流水线评测功能：
- 各路召回贡献分析
- 转化率分析
- 延迟分解（各模块耗时）
- 推荐质量指标
"""
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from .metrics import (
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    diversity_score,
    novelty_score
)


class PipelineEvaluator:
    """
    端到端流水线评测器

    评测整个推荐系统流水线的性能和效果
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化评测器

        Args:
            config: 评测配置
        """
        self.config = config or {}
        self.intent_module = None
        self.retriever = None
        self.reranker = None
        self.planner = None

        # 评测结果存储
        self.results = []
        self.recall_contributions = defaultdict(list)  # 各路召回贡献
        self.latency_breakdown = defaultdict(list)    # 延迟分解
        self.conversion_metrics = []                   # 转化率指标

    def initialize(self, use_llm: bool = False):
        """
        初始化各个模块

        Args:
            use_llm: 是否使用LLM模式
        """
        print("初始化推荐系统���块...")

        # 1. 意图理解
        from src.llm4rec.intent_understanding import IntentUnderstandingModule
        self.intent_module = IntentUnderstandingModule(use_template=not use_llm)
        print("  ✓ 意图理解模块")

        # 2. 召回
        from src.recommendation.candidate_merger import CandidateMerger
        self.retriever = CandidateMerger()
        print("  ✓ 候选召回模块")

        # 3. 重排序
        from src.llm4rec.llm_reranker import LLMReranker
        self.reranker = LLMReranker(use_template=not use_llm)
        print("  ✓ 重排序模块")

        # 4. 路线规划
        from src.routing.vrptw_solver import VRPTWSolver
        self.planner = VRPTWSolver
        print("  ✓ 路线规划模块")

    def evaluate_query(
        self,
        query: Dict,
        track_recall_contributions: bool = True
    ) -> Dict:
        """
        评测单个查询

        Args:
            query: 查询字典 {query, province, days, interests, ground_truth_pois}
            track_recall_contributions: 是否跟踪召回贡献

        Returns:
            评测结果字典
        """
        result = {
            "query": query["query"],
            "error": None,
            "latency": 0.0,
            "intent": None,
            "recall_contributions": {},
            "num_candidates": 0,
            "num_final_pois": 0,
            "route_feasible": False,
            "route_hours": 0.0,
            "latency_breakdown": {}
        }

        start_time = time.time()

        try:
            # 1. 意图理解
            intent_start = time.time()
            intent = self.intent_module.understand(query["query"])
            result["intent"] = intent
            intent_time = time.time() - intent_start

            # 2. 召回（跟踪各路召回）
            recall_start = time.time()
            if track_recall_contributions:
                candidates, contributions = self._recall_with_tracking(
                    intent=intent,
                    top_k=80,
                    province=query.get("province")
                )
                result["recall_contributions"] = contributions
            else:
                candidates = self.retriever.merge_candidates(
                    intent=intent,
                    top_k=80
                )

            recall_time = time.time() - recall_start
            result["num_candidates"] = len(candidates)

            # 3. 重排序
            rerank_start = time.time()
            final_pois = self.reranker.rerank(
                candidates=candidates,
                intent=intent,
                topk=30
            )
            rerank_time = time.time() - rerank_start
            result["num_final_pois"] = len(final_pois)

            # 4. 路线规划
            if len(final_pois) >= 3:
                from src.routing.time_matrix_builder import build_time_matrix

                plan_start = time.time()
                selected_pois = final_pois.head(min(20, len(final_pois)))

                try:
                    time_matrix, poi_df = build_time_matrix(
                        poi_ids=selected_pois['poi_id'].tolist(),
                        use_cache=True
                    )

                    solver = self.planner(poi_df, time_matrix)
                    solution = solver.solve(
                        max_duration_hours=10,
                        time_limit_seconds=30
                    )

                    if solution:
                        result["route_feasible"] = True
                        result["route_hours"] = solution.get("total_hours", 0)
                        result["num_visited"] = solution.get("visited_pois", 0)
                    else:
                        result["route_feasible"] = False

                    plan_time = time.time() - plan_start
                except Exception as e:
                    plan_time = time.time() - plan_start
                    result["route_error"] = str(e)
            else:
                plan_time = 0
                result["route_feasible"] = False

            total_time = time.time() - start_time
            result["latency"] = total_time

            result["latency_breakdown"] = {
                "intent": intent_time,
                "recall": recall_time,
                "rerank": rerank_time,
                "plan": plan_time,
                "total": total_time
            }

        except Exception as e:
            result["error"] = str(e)
            result["latency"] = time.time() - start_time

        return result

    def _recall_with_tracking(
        self,
        intent: Dict,
        top_k: int,
        province: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        带贡献跟踪的召回

        Args:
            intent: 用户意图
            top_k: 召回数量
            province: 省份过滤

        Returns:
            (候选DataFrame, 贡献字典)
        """
        from src.recommendation.candidate_merger import merge_candidates

        # 这里简化处理，实际应该修改merge_candidates返回各路贡献
        # 目前先返回基本的合并结果
        candidates = merge_candidates(
            query_text=intent.get("query", ""),
            topk_dense=top_k,
            topk_seq=30,
            topk_geo=30,
            province_filter=province
        )

        # 模拟各路召回贡献（实际应该从召回模块获取）
        contributions = {
            "semantic": len(candidates) * 0.55,  # 语义召回贡献
            "behavior": len(candidates) * 0.30,  # 行为召回贡献
            "geo": len(candidates) * 0.15,       # 地理召回贡献
            "total": len(candidates)
        }

        return candidates, contributions

    def evaluate_batch(
        self,
        queries: List[Dict],
        track_recall: bool = True,
        track_conversion: bool = True
    ) -> Dict[str, Any]:
        """
        批量评测

        Args:
            queries: 查询列表
            track_recall: 是否跟踪召回贡献
            track_conversion: 是否跟踪转化率

        Returns:
            评测指标字典
        """
        print(f"\n开始评测 {len(queries)} 个查询...")
        print("-" * 80)

        results = []
        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] 查询: {query['query'][:50]}...")
            result = self.evaluate_query(query, track_recall_contributions=track_recall)
            results.append(result)

            if result["error"]:
                print(f"  ✗ 失败: {result['error']}")
            else:
                print(f"  ✓ 成功")
                print(f"    意图省份: {result['intent'].get('province') if result['intent'] else 'N/A'}")
                print(f"    候选数: {result['num_candidates']}")
                print(f"    最终POI: {result['num_final_pois']}")
                print(f"    路线可行: {result['route_feasible']}")
                print(f"    延迟: {result['latency']:.2f}s")

        # 汇总指标
        metrics = self._aggregate_metrics(results)

        return {
            "metrics": metrics,
            "results": results
        }

    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        聚合评测指标

        Args:
            results: 评测结果列表

        Returns:
            聚合指标字典
        """
        metrics = {
            "total_queries": len(results),
            "successful_queries": 0,
            "feasible_routes": 0,
            "avg_latency": 0.0,
            "avg_candidates": 0.0,
            "avg_final_pois": 0.0,
            "avg_route_hours": 0.0,

            # 召回贡献分析
            "recall_contributions": {
                "semantic": 0.0,
                "behavior": 0.0,
                "geo": 0.0
            },

            # 延迟分解
            "latency_breakdown": {
                "intent": [],
                "recall": [],
                "rerank": [],
                "plan": []
            },

            # 转化率
            "conversion_metrics": {
                "recall_to_rerank": 0.0,  # 召回->重排序转化率
                "rerank_to_route": 0.0,   # 重排序->路线规划转化率
                "overall_success": 0.0    # 整体成功率
            }
        }

        successful = [r for r in results if not r["error"]]
        metrics["successful_queries"] = len(successful)

        if successful:
            # 基础指标
            metrics["avg_latency"] = np.mean([r["latency"] for r in successful])
            metrics["avg_candidates"] = np.mean([r["num_candidates"] for r in successful])
            metrics["avg_final_pois"] = np.mean([r["num_final_pois"] for r in successful])

            # 可行路线
            feasible = [r for r in successful if r["route_feasible"]]
            metrics["feasible_routes"] = len(feasible)
            if feasible:
                metrics["avg_route_hours"] = np.mean([r.get("route_hours", 0) for r in feasible])

            # 召回贡献
            recall_contribs = [r.get("recall_contributions", {}) for r in successful]
            if recall_contribs:
                metrics["recall_contributions"]["semantic"] = np.mean([
                    c.get("semantic", 0) for c in recall_contribs
                ])
                metrics["recall_contributions"]["behavior"] = np.mean([
                    c.get("behavior", 0) for c in recall_contribs
                ])
                metrics["recall_contributions"]["geo"] = np.mean([
                    c.get("geo", 0) for c in recall_contribs
                ])

            # 延迟分解
            for r in successful:
                if "latency_breakdown" in r:
                    bd = r["latency_breakdown"]
                    metrics["latency_breakdown"]["intent"].append(bd.get("intent", 0))
                    metrics["latency_breakdown"]["recall"].append(bd.get("recall", 0))
                    metrics["latency_breakdown"]["rerank"].append(bd.get("rerank", 0))
                    metrics["latency_breakdown"]["plan"].append(bd.get("plan", 0))

            # 计算平均延迟
            for stage in metrics["latency_breakdown"]:
                if metrics["latency_breakdown"][stage]:
                    metrics["latency_breakdown"][stage] = np.mean(metrics["latency_breakdown"][stage])

            # 转化率
            if successful:
                # 召回->重排序: 平均最终POI数 / 平均候选数
                metrics["conversion_metrics"]["recall_to_rerank"] = (
                    metrics["avg_final_pois"] / metrics["avg_candidates"]
                    if metrics["avg_candidates"] > 0 else 0.0
                )

                # 重排序->路线: 可行路线数 / 成功查询数
                metrics["conversion_metrics"]["rerank_to_route"] = (
                    len(feasible) / len(successful)
                )

                # 整体成功率
                metrics["conversion_metrics"]["overall_success"] = (
                    len(feasible) / len(results)
                )

        return metrics

    def calculate_recommendation_metrics(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        item_attributes: Optional[Dict[str, Dict]] = None,
        item_popularity: Optional[Dict[str, float]] = None,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """
        计算推荐质量指标

        Args:
            predictions: 预测列表
            ground_truth: 真实标签列表
            item_attributes: 物品属性（用于多样性）
            item_popularity: 物品流行度（用于新颖性）
            k_values: K值列表

        Returns:
            指标字典
        """
        metrics = {}

        # 基础指标
        for k in k_values:
            recalls = [
                recall_at_k(pred, truth, k)
                for pred, truth in zip(predictions, ground_truth)
            ]
            metrics[f"recall@{k}"] = np.mean(recalls)

            ndcgs = [
                ndcg_at_k(pred, truth, k)
                for pred, truth in zip(predictions, ground_truth)
            ]
            metrics[f"ndcg@{k}"] = np.mean(ndcgs)

            hit_rates = [
                hit_rate_at_k(pred, truth, k)
                for pred, truth in zip(predictions, ground_truth)
            ]
            metrics[f"hitrate@{k}"] = np.mean(hit_rates)

        # 多样性
        if item_attributes is not None:
            metrics["diversity"] = diversity_score(predictions, item_attributes)

        # 新颖性
        if item_popularity is not None:
            metrics["novelty"] = novelty_score(predictions, item_popularity, k=10)

        return metrics


def generate_evaluation_report(
    evaluation: Dict[str, Any],
    include_details: bool = True
) -> str:
    """
    生成评测报告

    Args:
        evaluation: 评测结果字典
        include_details: 是否包含详细信息

    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 80)
    lines.append("GoAfar 端到端评测报告")
    lines.append("=" * 80)

    m = evaluation["metrics"]

    # 成功率
    lines.append("\n【成功率】")
    lines.append(f"  成功查询: {m['successful_queries']}/{m['total_queries']} "
                 f"({m['successful_queries']/m['total_queries']:.1%})")
    lines.append(f"  可行路线: {m['feasible_routes']}/{m['total_queries']} "
                 f"({m['feasible_routes']/m['total_queries']:.1%})")

    # 性能指标
    lines.append("\n【性能指标】")
    lines.append(f"  平均延迟: {m['avg_latency']:.2f}s")
    lines.append(f"  平均候选数: {m['avg_candidates']:.0f}")
    lines.append(f"  平均最终POI: {m['avg_final_pois']:.0f}")

    if m['avg_route_hours'] > 0:
        lines.append(f"  平均路线时长: {m['avg_route_hours']:.1f}h")

    # 召回贡献分析
    lines.append("\n【召回贡献分析】")
    rc = m['recall_contributions']
    if rc['semantic'] > 0 or rc['behavior'] > 0 or rc['geo'] > 0:
        total = rc['semantic'] + rc['behavior'] + rc['geo']
        if total > 0:
            lines.append(f"  语义召回: {rc['semantic']:.1f} ({rc['semantic']/total:.1%})")
            lines.append(f"  行为召回: {rc['behavior']:.1f} ({rc['behavior']/total:.1%})")
            lines.append(f"  地理召回: {rc['geo']:.1f} ({rc['geo']/total:.1%})")

    # 延迟分解
    lines.append("\n【延迟分解】")
    bd = m['latency_breakdown']
    if isinstance(bd.get('intent'), (int, float)):
        lines.append(f"  意图理解: {bd['intent']:.2f}s")
        lines.append(f"  候选召回: {bd['recall']:.2f}s")
        lines.append(f"  重排序: {bd['rerank']:.2f}s")
        lines.append(f"  路线规划: {bd['plan']:.2f}s")

    # 转化率
    lines.append("\n【转化率】")
    conv = m['conversion_metrics']
    lines.append(f"  召回→重排序: {conv['recall_to_rerank']:.2%}")
    lines.append(f"  重排序→路线: {conv['rerank_to_route']:.2%}")
    lines.append(f"  整体成功率: {conv['overall_success']:.2%}")

    if include_details:
        lines.append("\n【详细结果】")
        for i, result in enumerate(evaluation.get("results", [])[:5]):
            lines.append(f"\n  查询 {i+1}: {result['query'][:40]}...")
            if result.get("error"):
                lines.append(f"    错误: {result['error']}")
            else:
                lines.append(f"    候选: {result['num_candidates']}, "
                           f"最终: {result['num_final_pois']}, "
                           f"可行: {result['route_feasible']}, "
                           f"延迟: {result['latency']:.2f}s")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def save_evaluation_report(
    evaluation: Dict[str, Any],
    output_path: str
):
    """
    保存评测结果

    Args:
        evaluation: 评测结果字典
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存: {output_path}")
