"""
Pipeline request/response contracts.
Pydantic is preferred; dataclass fallback is provided for lightweight environments.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field

    class IntentInfo(BaseModel):
        original_query: str = ""
        province: Optional[str] = None
        cities: List[str] = Field(default_factory=list)
        interests: List[str] = Field(default_factory=list)
        activities: List[str] = Field(default_factory=list)
        duration_days: Optional[int] = None
        season_preference: Optional[str] = None
        travel_style: Optional[str] = None
        constraints: List[str] = Field(default_factory=list)
        expanded_query: Optional[str] = None
        extra: Dict[str, Any] = Field(default_factory=dict)


    class RouteStop(BaseModel):
        poi_id: str
        poi_name: str
        poi_city: Optional[str] = None
        arrival_time_min: Optional[int] = None
        arrival_time_str: Optional[str] = None
        stay_min: int = 0


    class RecommendationRequest(BaseModel):
        query_text: str = Field(min_length=1, description="User query text")
        province: Optional[str] = None
        max_hours: float = Field(default=10, gt=0, le=24)
        topk_candidates: int = Field(default=30, ge=5, le=200)
        user_id: Optional[str] = None
        use_llm: bool = True
        return_debug: bool = False


    class RecommendationDebug(BaseModel):
        search_query: str = ""
        recall_sources: Dict[str, int] = Field(default_factory=dict)
        fallback_events: List[str] = Field(default_factory=list)
        matrix_provider: str = ""
        candidate_count_before_rerank: int = 0
        candidate_count_after_rerank: int = 0


    class RecommendationResponse(BaseModel):
        success: bool = True
        error: Optional[str] = None
        title: str = ""
        description: str = ""
        route: List[RouteStop] = Field(default_factory=list)
        total_hours: float = 0.0
        num_pois: int = 0
        query: str
        province: Optional[str] = None
        user_intent: Optional[IntentInfo] = None
        degraded: bool = False
        debug: Optional[RecommendationDebug] = None

        @classmethod
        def error_response(cls, query: str, message: str) -> "RecommendationResponse":
            return cls(success=False, error=message, query=query)

except Exception:  # pragma: no cover - fallback mode
    from dataclasses import asdict, dataclass, field

    class _CompatModel:
        model_fields: Dict[str, Any] = {}

        def model_dump(self) -> Dict[str, Any]:
            return asdict(self)


    @dataclass
    class IntentInfo(_CompatModel):
        original_query: str = ""
        province: Optional[str] = None
        cities: List[str] = field(default_factory=list)
        interests: List[str] = field(default_factory=list)
        activities: List[str] = field(default_factory=list)
        duration_days: Optional[int] = None
        season_preference: Optional[str] = None
        travel_style: Optional[str] = None
        constraints: List[str] = field(default_factory=list)
        expanded_query: Optional[str] = None
        extra: Dict[str, Any] = field(default_factory=dict)

    IntentInfo.model_fields = IntentInfo.__annotations__


    @dataclass
    class RouteStop(_CompatModel):
        poi_id: str
        poi_name: str
        poi_city: Optional[str] = None
        arrival_time_min: Optional[int] = None
        arrival_time_str: Optional[str] = None
        stay_min: int = 0

    RouteStop.model_fields = RouteStop.__annotations__


    @dataclass
    class RecommendationRequest(_CompatModel):
        query_text: str
        province: Optional[str] = None
        max_hours: float = 10
        topk_candidates: int = 30
        user_id: Optional[str] = None
        use_llm: bool = True
        return_debug: bool = False

    RecommendationRequest.model_fields = RecommendationRequest.__annotations__


    @dataclass
    class RecommendationDebug(_CompatModel):
        search_query: str = ""
        recall_sources: Dict[str, int] = field(default_factory=dict)
        fallback_events: List[str] = field(default_factory=list)
        matrix_provider: str = ""
        candidate_count_before_rerank: int = 0
        candidate_count_after_rerank: int = 0

    RecommendationDebug.model_fields = RecommendationDebug.__annotations__


    @dataclass
    class RecommendationResponse(_CompatModel):
        query: str
        success: bool = True
        error: Optional[str] = None
        title: str = ""
        description: str = ""
        route: List[RouteStop] = field(default_factory=list)
        total_hours: float = 0.0
        num_pois: int = 0
        province: Optional[str] = None
        user_intent: Optional[IntentInfo] = None
        degraded: bool = False
        debug: Optional[RecommendationDebug] = None

        @classmethod
        def error_response(cls, query: str, message: str) -> "RecommendationResponse":
            return cls(success=False, error=message, query=query)

    RecommendationResponse.model_fields = RecommendationResponse.__annotations__


def model_to_dict(obj: Any) -> Dict[str, Any]:
    """
    Compat helper for pydantic v1/v2 and dataclass fallback objects.
    """
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}
