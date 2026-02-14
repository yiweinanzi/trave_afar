"""
GoAfar unified main entrypoint.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schemas.recommendation import RecommendationRequest, model_to_dict
from service.pipeline import get_pipeline


def recommend_route(
    query_text,
    province=None,
    max_hours=10,
    topk_candidates=20,
    user_id=None,
    use_llm=True,
):
    """
    Backward-compatible wrapper for unified pipeline.
    """
    pipeline = get_pipeline()
    req = RecommendationRequest(
        query_text=query_text,
        province=province,
        max_hours=max_hours,
        topk_candidates=topk_candidates,
        user_id=user_id,
        use_llm=use_llm,
        return_debug=False,
    )
    resp = pipeline.recommend(req)
    if not resp.success:
        return {"error": resp.error}

    return {
        "title": resp.title,
        "description": resp.description,
        "route": [model_to_dict(stop) for stop in resp.route],
        "total_hours": resp.total_hours,
        "num_pois": resp.num_pois,
        "query": resp.query,
        "province": resp.province,
        "user_intent": model_to_dict(resp.user_intent) if resp.user_intent else None,
        "degraded": resp.degraded,
    }


def main():
    scenarios = [
        {"query_text": "想去新疆喀纳斯看秋天的景色，拍照", "province": "新疆", "max_hours": 10},
        {"query_text": "去西藏朝拜布达拉宫，体验藏族文化", "province": "西藏", "max_hours": 8},
        {"query_text": "云南大理洱海骑行，逛古镇", "province": "云南", "max_hours": 6},
    ]

    os.makedirs("outputs/results", exist_ok=True)
    results_all = []
    for idx, scenario in enumerate(scenarios, 1):
        try:
            result = recommend_route(**scenario, use_llm=True)
            if "error" in result:
                print(f"✗ 场景 {idx} 失败: {result['error']}")
                continue
            results_all.append(result)
            output_file = f"outputs/results/scenario_{idx}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✓ 结果已保存: {output_file}")
        except Exception as exc:
            print(f"✗ 场景 {idx} 执行异常: {exc}")

    print(f"✓ 完成 {len(results_all)}/{len(scenarios)} 个场景")


if __name__ == "__main__":
    main()
