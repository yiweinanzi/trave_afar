import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from schemas.recommendation import RecommendationRequest, RecommendationResponse, RouteStop, model_to_dict


def test_request_contract():
    req = RecommendationRequest(query_text="想去新疆看雪山", topk_candidates=20, max_hours=8)
    assert req.query_text
    assert req.topk_candidates == 20
    assert req.max_hours == 8


def test_response_contract():
    resp = RecommendationResponse(
        success=True,
        query="想去新疆看雪山",
        route=[RouteStop(poi_id="1001", poi_name="天山天池", stay_min=120)],
    )
    dumped = model_to_dict(resp)
    assert dumped["success"] is True
    assert isinstance(dumped["route"], list)
    assert dumped["route"][0]["poi_id"] == "1001"
