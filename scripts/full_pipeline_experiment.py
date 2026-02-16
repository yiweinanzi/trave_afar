#!/usr/bin/env python3
"""
GoAfar 全链路实验脚本

测试完整的推荐系统流程：
1. 数据准备检查
2. 端到端Pipeline测试
3. 多组件协调测试
4. 性能测试
5. 生成实验报告
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from service.pipeline import RecommendationPipeline
from service.config import load_runtime_config


class PipelineExperiment:
    """全链路实验执行器"""

    def __init__(self):
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "data_check": {},
            "stages": {},
            "performance": {},
            "recommendations": [],
        }
        self.pipeline = None

    @staticmethod
    def _route_stop_to_dict(stop: Any) -> Dict[str, Any]:
        """兼容不同RouteStop字段命名。"""
        name = getattr(stop, "poi_name", None) or getattr(stop, "name", None) or ""
        stay_min = getattr(stop, "stay_min", None)
        if stay_min is None:
            stay_min = getattr(stop, "duration_minutes", 0)
        poi_id = getattr(stop, "poi_id", "") or ""
        return {
            "name": name,
            "duration_minutes": int(stay_min or 0),
            "poi_id": poi_id,
        }

    def check_data_preparation(self) -> bool:
        """1. 数据准备检查"""
        print("\n" + "=" * 60)
        print("阶段 1: 数据准备检查")
        print("=" * 60)

        cfg = load_runtime_config()
        checks_passed = True

        # 检查POI嵌入
        emb_file = cfg.resolve_path(f"{cfg.paths.emb_dir}/poi_emb.npy")
        meta_file = cfg.resolve_path(f"{cfg.paths.emb_dir}/poi_meta.csv")

        print(f"\n检查POI嵌入文件: {emb_file}")
        if emb_file.exists():
            emb_data = np.load(emb_file, allow_pickle=True)
            print(f"  ✓ POI嵌入存在，shape: {emb_data.shape}")
            self.results["data_check"]["poi_emb"] = {
                "exists": True,
                "shape": list(emb_data.shape),
                "size_mb": emb_file.stat().st_size / (1024 * 1024)
            }
        else:
            print(f"  ✗ POI嵌入不存在")
            self.results["data_check"]["poi_emb"] = {"exists": False}
            checks_passed = False

        print(f"\n检查POI元数据: {meta_file}")
        if meta_file.exists():
            import pandas as pd
            meta_df = pd.read_csv(meta_file)
            print(f"  ✓ POI元数据存在，rows: {len(meta_df)}, cols: {list(meta_df.columns)}")
            self.results["data_check"]["poi_meta"] = {
                "exists": True,
                "rows": len(meta_df),
                "columns": list(meta_df.columns)
            }
        else:
            print(f"  ✗ POI元数据不存在")
            self.results["data_check"]["poi_meta"] = {"exists": False}
            checks_passed = False

        # 检查时间矩阵
        time_matrix_file = cfg.resolve_path(f"{cfg.paths.routing_dir}/time_matrix.npy")
        print(f"\n检查时间矩阵: {time_matrix_file}")
        if time_matrix_file.exists():
            try:
                # 时间矩阵可能存储为字典或数组
                data = np.load(time_matrix_file, allow_pickle=True)
                if data.shape == ():
                    # 如果是scalar，则是一个字典
                    matrix = data.item()
                else:
                    # 如果是数组，直接使用
                    matrix = data
                print(f"  ✓ 时间矩阵存在，size: {time_matrix_file.stat().st_size / (1024 * 1024):.2f} MB")
                self.results["data_check"]["time_matrix"] = {
                    "exists": True,
                    "size_mb": time_matrix_file.stat().st_size / (1024 * 1024)
                }
            except Exception as e:
                print(f"  ⚠️ 时间矩阵文件损坏: {e}")
                self.results["data_check"]["time_matrix"] = {"exists": False}
        else:
            print(f"  ✗ 时间矩阵不存在（将在首次运行时自动生成）")
            self.results["data_check"]["time_matrix"] = {"exists": False}

        # 检查POI数据
        poi_file = cfg.resolve_path(cfg.paths.poi_csv)
        print(f"\n检查POI数据: {poi_file}")
        if poi_file.exists():
            import pandas as pd
            poi_df = pd.read_csv(poi_file)
            print(f"  ✓ POI数据存在，rows: {len(poi_df)}, provinces: {poi_df['province'].nunique()}")
            self.results["data_check"]["poi_data"] = {
                "exists": True,
                "rows": len(poi_df),
                "provinces": poi_df['province'].nunique() if 'province' in poi_df.columns else 0
            }
        else:
            print(f"  ✗ POI数据不存在")
            self.results["data_check"]["poi_data"] = {"exists": False}
            checks_passed = False

        return checks_passed

    def initialize_pipeline(self) -> bool:
        """初始化Pipeline"""
        print("\n" + "=" * 60)
        print("阶段 2: 初始化Pipeline")
        print("=" * 60)

        try:
            start_time = time.time()
            cfg = load_runtime_config()
            self.pipeline = RecommendationPipeline(config=cfg)
            self.pipeline.warmup()
            init_time = time.time() - start_time

            print(f"\n✓ Pipeline初始化完成，耗时: {init_time:.2f}s")

            # 检查模型健康状态
            health = self.pipeline.check_model_health()
            print(f"\n模型健康状态:")
            for model, status in health.items():
                icon = "✓" if status else "✗"
                print(f"  {icon} {model}: {status}")

            self.results["stages"]["initialization"] = {
                "success": True,
                "duration_seconds": init_time,
                "model_health": health
            }
            return True
        except Exception as e:
            print(f"\n✗ Pipeline初始化失败: {e}")
            self.results["stages"]["initialization"] = {
                "success": False,
                "error": str(e)
            }
            return False

    def test_full_pipeline(self, request: Dict[str, Any], test_name: str) -> Dict[str, Any]:
        """测试完整推荐流程"""
        print("\n" + "=" * 60)
        print(f"阶段 3: 端到端Pipeline测试 - {test_name}")
        print("=" * 60)
        print(f"\n请求参数:")
        for key, value in request.items():
            print(f"  {key}: {value}")

        result = {
            "test_name": test_name,
            "request": request,
            "stage_times": {},
            "success": False,
            "error": None,
            "response": None
        }

        try:
            # 准备请求
            from schemas.recommendation import RecommendationRequest
            rec_request = RecommendationRequest(
                query_text=request.get("query", ""),
                province=request.get("city", request.get("province", "")),
                max_hours=min(request.get("days", 1) * 10, 10),
                topk_candidates=30,
                use_llm=request.get("use_llm", False),
                return_debug=True
            )

            # 执行推荐
            total_start = time.time()

            # Step 1: 意图理解
            step_start = time.time()
            # 意图理解在pipeline内部执行
            result["stage_times"]["intent_understanding"] = 0  # 内部计时

            # Step 2: 召回
            step_start = time.time()
            # 召回在pipeline内部执行
            result["stage_times"]["recall"] = 0  # 内部计时

            # Step 3: 排序
            step_start = time.time()
            result["stage_times"]["reranking"] = 0  # 内部计时

            # 执行完整推荐
            response = self.pipeline.recommend(rec_request)
            total_time = time.time() - total_start

            result["stage_times"]["total"] = total_time

            if response.success:
                print(f"\n✓ 推荐成功！")
                print(f"  标题: {response.title}")
                print(f"  描述: {response.description[:100]}...")
                print(f"  总时长: {response.total_hours:.1f}小时")
                print(f"  景点数: {response.num_pois}")
                print(f"  省份: {response.province}")

                # 显示路线
                print(f"\n  推荐路线:")
                for i, stop in enumerate(response.route[:5]):  # 只显示前5个
                    stop_data = self._route_stop_to_dict(stop)
                    print(f"    {i+1}. {stop_data['name']} (停留{stop_data['duration_minutes']}分钟)")
                if len(response.route) > 5:
                    print(f"    ... 共{len(response.route)}个景点")

                # Debug信息
                if response.debug:
                    print(f"\n  Debug信息:")
                    print(f"    召回来源: {response.debug.recall_sources}")
                    print(f"    召回候选数: {response.debug.candidate_count_before_rerank}")
                    print(f"    排序后候选数: {response.debug.candidate_count_after_rerank}")
                    print(f"    时间矩阵来源: {response.debug.matrix_provider}")
                    if response.debug.fallback_events:
                        print(f"    降级事件: {response.debug.fallback_events}")

                result["success"] = True
                result["response"] = {
                    "title": response.title,
                    "description": response.description,
                    "total_hours": response.total_hours,
                    "num_pois": response.num_pois,
                    "province": response.province,
                    "route": [self._route_stop_to_dict(stop) for stop in response.route],
                    "debug": {
                        "recall_sources": response.debug.recall_sources,
                        "candidate_count_before_rerank": response.debug.candidate_count_before_rerank,
                        "candidate_count_after_rerank": response.debug.candidate_count_after_rerank,
                        "matrix_provider": response.debug.matrix_provider,
                        "fallback_events": response.debug.fallback_events
                    } if response.debug else None
                }
            else:
                print(f"\n✗ 推荐失败: {response.title}")  # error stored in title
                result["error"] = response.title

        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)

        return result

    def run_multi_component_test(self) -> None:
        """3. 多组件协调测试"""
        print("\n" + "=" * 60)
        print("阶段 3: 多组件协调测试")
        print("=" * 60)

        # 测试用例
        test_cases = [
            {
                "name": "基础模板模式",
                "request": {
                    "query": "想去新疆看7天雪山和草原，拍照",
                    "city": "新疆",
                    "days": 7,
                    "budget": 5000,
                    "group_type": "朋友",
                    "interests": ["自然", "摄影"],
                    "use_llm": False
                }
            },
            {
                "name": "LLM增强模式",
                "request": {
                    "query": "想去云南看古镇和自然风光，5天行程",
                    "city": "云南",
                    "days": 5,
                    "budget": 3000,
                    "group_type": "情侣",
                    "interests": ["文化", "自然"],
                    "use_llm": True
                }
            },
            {
                "name": "甘肃短途游",
                "request": {
                    "query": "甘肃周末两日游，历史文化景点",
                    "city": "甘肃",
                    "days": 2,
                    "budget": 2000,
                    "group_type": "家庭",
                    "interests": ["历史", "文化"],
                    "use_llm": False
                }
            }
        ]

        cfg = load_runtime_config()
        poi_path = cfg.resolve_path(cfg.paths.poi_csv)
        available_provinces = set()
        if poi_path.exists():
            import pandas as pd
            poi_df = pd.read_csv(poi_path)
            if "province" in poi_df.columns:
                available_provinces = set(
                    poi_df["province"].dropna().astype(str).unique().tolist()
                )

        for test_case in test_cases:
            province = test_case["request"].get("city", test_case["request"].get("province", ""))
            if province and available_provinces and province not in available_provinces:
                msg = f"province_not_supported_in_dataset:{province}"
                print(f"\n⚠️ 跳过测试 {test_case['name']}: {msg}")
                self.results["recommendations"].append({
                    "test_name": test_case["name"],
                    "request": test_case["request"],
                    "stage_times": {},
                    "success": False,
                    "error": msg,
                    "response": None,
                    "skipped": True,
                })
                continue
            result = self.test_full_pipeline(test_case["request"], test_case["name"])
            self.results["recommendations"].append(result)

    def run_performance_test(self) -> None:
        """4. 性能测试"""
        print("\n" + "=" * 60)
        print("阶段 4: 性能测试")
        print("=" * 60)

        # 内存测试
        print("\n内存使用情况:")
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"  RSS内存: {mem_info.rss / (1024 ** 2):.2f} MB")
        print(f"  VMS内存: {mem_info.vms / (1024 ** 2):.2f} MB")

        self.results["performance"]["memory"] = {
            "rss_mb": mem_info.rss / (1024 ** 2),
            "vms_mb": mem_info.vms / (1024 ** 2)
        }

        # 多次运行测试延迟
        print("\n延迟测试（3次运行）:")
        latencies = []
        for i in range(3):
            request = {
                "query": "四川成都三日游，美食和熊猫",
                "city": "四川",
                "days": 3,
                "use_llm": False
            }

            from schemas.recommendation import RecommendationRequest
            rec_request = RecommendationRequest(
                query_text=request["query"],
                province=request["city"],
                max_hours=10,
                topk_candidates=30,
                use_llm=False,
                return_debug=True
            )

            start = time.time()
            response = self.pipeline.recommend(rec_request)
            latency = time.time() - start
            latencies.append(latency)
            print(f"  运行 {i+1}: {latency:.2f}s")

        self.results["performance"]["latency"] = {
            "mean": np.mean(latencies),
            "std": np.std(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "samples": latencies
        }

        print(f"\n延迟统计:")
        print(f"  平均: {np.mean(latencies):.2f}s")
        print(f"  标准差: {np.std(latencies):.2f}s")
        print(f"  最小: {np.min(latencies):.2f}s")
        print(f"  最大: {np.max(latencies):.2f}s")

    def generate_report(self) -> None:
        """5. 生成实验报告"""
        print("\n" + "=" * 60)
        print("阶段 5: 生成实验报告")
        print("=" * 60)

        # 准备报告内容
        report_lines = []

        # 标题
        report_lines.append("# GoAfar 全链路实验报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {self.results['timestamp']}")
        report_lines.append("")

        # 1. 数据准备检查
        report_lines.append("## 1. 数据准备检查")
        report_lines.append("")
        for key, value in self.results["data_check"].items():
            status = "✓" if value.get("exists", False) else "✗"
            report_lines.append(f"- **{key}**: {status}")
            if value.get("exists"):
                for k, v in value.items():
                    if k != "exists":
                        report_lines.append(f"  - {k}: {v}")
        report_lines.append("")

        # 2. Pipeline初始化
        report_lines.append("## 2. Pipeline初始化")
        report_lines.append("")
        init_result = self.results["stages"].get("initialization", {})
        if init_result.get("success"):
            report_lines.append(f"- ✓ 初始化成功")
            report_lines.append(f"- 初始化耗时: {init_result['duration_seconds']:.2f}s")
            report_lines.append("")
            report_lines.append("### 模型健康状态")
            report_lines.append("")
            health = init_result.get("model_health", {})
            for model, status in health.items():
                icon = "✓" if status else "✗"
                report_lines.append(f"- {icon} **{model}**: {status}")
        else:
            report_lines.append(f"- ✗ 初始化失败: {init_result.get('error', 'Unknown')}")
        report_lines.append("")

        # 3. 多组件协调测试
        report_lines.append("## 3. 多组件协调测试")
        report_lines.append("")
        for rec in self.results["recommendations"]:
            report_lines.append(f"### {rec['test_name']}")
            report_lines.append("")
            report_lines.append(f"**请求参数**:")
            for key, value in rec['request'].items():
                report_lines.append(f"- {key}: {value}")
            report_lines.append("")

            if rec["success"]:
                resp = rec["response"]
                report_lines.append(f"**推荐结果**:")
                report_lines.append(f"- 标题: {resp['title']}")
                report_lines.append(f"- 描述: {resp['description'][:200]}...")
                report_lines.append(f"- 总时长: {resp['total_hours']:.1f}小时")
                report_lines.append(f"- 景点数: {resp['num_pois']}")
                report_lines.append(f"- 省份: {resp['province']}")

                if resp.get("debug"):
                    dbg = resp["debug"]
                    report_lines.append("")
                    report_lines.append(f"**Debug信息**:")
                    report_lines.append(f"- 召回来源: {dbg['recall_sources']}")
                    report_lines.append(f"- 候选数（召回后）: {dbg['candidate_count_before_rerank']}")
                    report_lines.append(f"- 候选数（排序后）: {dbg['candidate_count_after_rerank']}")
                    report_lines.append(f"- 时间矩阵来源: {dbg['matrix_provider']}")
                    if dbg.get("fallback_events"):
                        report_lines.append(f"- 降级事件: {dbg['fallback_events']}")

                report_lines.append("")
                report_lines.append(f"**推荐路线**:")
                for i, stop in enumerate(resp["route"][:10]):
                    report_lines.append(f"{i+1}. {stop['name']} (停留{stop['duration_minutes']}分钟)")
                if len(resp["route"]) > 10:
                    report_lines.append(f"... 共{len(resp['route'])}个景点")
            else:
                report_lines.append(f"**错误**: {rec.get('error', 'Unknown')}")
            report_lines.append("")

        # 4. 性能测试
        report_lines.append("## 4. 性能测试")
        report_lines.append("")
        perf = self.results["performance"]
        if "memory" in perf:
            report_lines.append("### 内存使用")
            report_lines.append("")
            report_lines.append(f"- RSS: {perf['memory']['rss_mb']:.2f} MB")
            report_lines.append(f"- VMS: {perf['memory']['vms_mb']:.2f} MB")
            report_lines.append("")

        if "latency" in perf:
            report_lines.append("### 延迟统计")
            report_lines.append("")
            lat = perf["latency"]
            report_lines.append(f"- 平均: {lat['mean']:.2f}s")
            report_lines.append(f"- 标准差: {lat['std']:.2f}s")
            report_lines.append(f"- 最小: {lat['min']:.2f}s")
            report_lines.append(f"- 最大: {lat['max']:.2f}s")
            report_lines.append("")
            report_lines.append("### 样本数据")
            report_lines.append("")
            for i, sample in enumerate(lat["samples"]):
                report_lines.append(f"- 运行 {i+1}: {sample:.2f}s")
            report_lines.append("")

        # 5. 问题诊断
        report_lines.append("## 5. 问题诊断")
        report_lines.append("")

        issues = []
        # 检查数据问题
        if not self.results["data_check"].get("poi_emb", {}).get("exists"):
            issues.append("- POI嵌入文件缺失，请运行嵌入构建脚本")
        if not self.results["data_check"].get("poi_meta", {}).get("exists"):
            issues.append("- POI元数据文件缺失")
        if not self.results["data_check"].get("poi_data", {}).get("exists"):
            issues.append("- POI数据文件缺失")

        # 检查模型问题
        if init_result.get("model_health"):
            health = init_result["model_health"]
            if not health.get("embedding", True):
                issues.append("- Embedding模型不可用")
            if not health.get("reranker", True):
                issues.append("- Reranker模型不可用")
            if not health.get("llm", True):
                issues.append("- LLM模型不可用")

        # 检查推荐问题
        failed_tests = [r for r in self.results["recommendations"] if not r["success"]]
        if failed_tests:
            report_lines.append("### 失败的测试")
            for test in failed_tests:
                report_lines.append(f"- **{test['test_name']}**: {test.get('error', 'Unknown')}")
            report_lines.append("")

        # 检查降级事件
        all_fallbacks = set()
        for rec in self.results["recommendations"]:
            if rec["success"] and rec["response"].get("debug", {}).get("fallback_events"):
                for event in rec["response"]["debug"]["fallback_events"]:
                    all_fallbacks.add(event)

        if all_fallbacks:
            report_lines.append("### 降级事件")
            for event in sorted(all_fallbacks):
                report_lines.append(f"- {event}")
            report_lines.append("")

        if not issues:
            report_lines.append("✓ 未发现明显问题，系统运行正常")
        else:
            report_lines.append("### 发现的问题")
            for issue in issues:
                report_lines.append(issue)

        # 保存报告
        report_path = Path("/root/autodl-tmp/goafar_project_broken/docs/FULL_PIPELINE_EXPERIMENT_REPORT.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"\n✓ 报告已保存到: {report_path}")
        print(f"  报告大小: {report_path.stat().st_size / 1024:.2f} KB")

        # 同时保存JSON格式
        json_path = report_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"✓ JSON数据已保存到: {json_path}")

    def run(self) -> None:
        """运行完整实验"""
        print("\n" + "=" * 60)
        print("GoAfar 全链路实验")
        print("=" * 60)

        # 阶段1: 数据检查
        if not self.check_data_preparation():
            print("\n⚠ 数据检查发现问题，但继续执行实验...")

        # 阶段2: 初始化
        if not self.initialize_pipeline():
            print("\n✗ Pipeline初始化失败，实验终止")
            return

        # 阶段3: 多组件测试
        self.run_multi_component_test()

        # 阶段4: 性能测试
        self.run_performance_test()

        # 阶段5: 生成报告
        self.generate_report()

        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)


def main():
    """主函数"""
    experiment = PipelineExperiment()
    experiment.run()


if __name__ == "__main__":
    main()
