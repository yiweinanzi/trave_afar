"""
LLM-enhanced entrypoint (same pipeline, different config override).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schemas.recommendation import RecommendationRequest, model_to_dict
from service.config import load_runtime_config
from service.pipeline import RecommendationPipeline

_LLM_PIPELINES = {}


def _build_llm_pipeline(use_gpu: bool = False) -> RecommendationPipeline:
    key = f"use_gpu={bool(use_gpu)}"
    if key in _LLM_PIPELINES:
        return _LLM_PIPELINES[key]

    cfg = load_runtime_config()
    cfg.llm.enabled = True
    cfg.llm.use_gpu = use_gpu
    cfg.rerank.enabled = True
    cfg.rerank.use_template = False
    pipeline = RecommendationPipeline(config=cfg)
    pipeline.warmup()
    _LLM_PIPELINES[key] = pipeline
    return pipeline


def recommend_with_llm(query_text, use_gpu=False, max_hours=10, topk_candidates=30):
    pipeline = _build_llm_pipeline(use_gpu=use_gpu)
    request = RecommendationRequest(
        query_text=query_text,
        max_hours=max_hours,
        topk_candidates=topk_candidates,
        use_llm=True,
        return_debug=False,
    )
    resp = pipeline.recommend(request)
    if not resp.success:
        return {"error": resp.error}

    return {
        "title": resp.title,
        "description": resp.description,
        "route": [model_to_dict(stop) for stop in resp.route],
        "total_hours": resp.total_hours,
        "num_pois": resp.num_pois,
        "query": resp.query,
        "user_intent": model_to_dict(resp.user_intent) if resp.user_intent else None,
        "province": resp.province,
        "degraded": resp.degraded,
    }


def main():
    scenarios = [
        {"query_text": "想去新疆喀纳斯看3天秋天的景色，拍照", "max_hours": 10},
        {"query_text": "计划西藏拉萨5日游，朝拜布达拉宫，体验藏族文化", "max_hours": 12},
        {"query_text": "云南大理洱海2天骑行，轻松休闲", "max_hours": 8},
    ]
    os.makedirs("outputs/results", exist_ok=True)
    done = 0
    for idx, scenario in enumerate(scenarios, 1):
        result = recommend_with_llm(**scenario, use_gpu=False)
        if "error" in result:
            print(f"✗ 场景 {idx} 失败: {result['error']}")
            continue
        done += 1
        path = f"outputs/results/llm_scenario_{idx}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✓ 结果已保存: {path}")
    print(f"✓ 完成 {done}/{len(scenarios)} 个场景")


if __name__ == "__main__":
    main()
