"""
Unified runtime configuration loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    poi_csv: str = "data/poi.csv"
    user_events_csv: str = "data/user_events.csv"
    emb_dir: str = "outputs/emb"
    routing_dir: str = "outputs/routing"
    cache_dir: str = "outputs/cache"
    results_dir: str = "outputs/results"
    shp_dir: str = "data/external/shengfen"
    yelp_dir: str = "data/external/yelp"
    geolife_dir: str = "data/external/Geolife Trajectories 1.3"


@dataclass
class EmbeddingConfig:
    model_path: str = "models/Xorbits/bge-m3"
    fallback_model: str = "models/Xorbits/bge-m3"
    use_gpu: bool = False
    auto_build_if_missing: bool = True
    backend: str = "auto"
    faiss_index_file: str = "outputs/emb/poi_faiss.index"


@dataclass
class RecallConfig:
    semantic_topk: int = 80
    behavior_topk: int = 60
    geo_topk: int = 40
    final_topk: int = 80
    fusion: str = "rrf"
    rrf_k: int = 60
    dense_weight: float = 0.55
    behavior_weight: float = 0.30
    geo_weight: float = 0.15
    calibrate: str = "minmax"
    behavior_provider: str = "popularity"
    recbole_model_path: str = "outputs/recbole/saved"
    recbole_config: str = "configs/recbole.yaml"
    recbole_use_gpu: bool = True
    adaptive_fusion: bool = False


@dataclass
class RerankConfig:
    enabled: bool = True
    use_template: bool = True
    use_reranker_model: bool = True
    qwen_reranker_path: str = "models/Qwen3-Reranker-4B"
    rerank_topk: int = 20
    topk: int = 30


@dataclass
class PlannerConfig:
    matrix_provider: str = "auto"
    osrm_url: str = "http://router.project-osrm.org"
    avg_speed_kmh: int = 55
    start_time_min: int = 480
    max_duration_hours: int = 10
    solve_time_limit_seconds: int = 25
    matrix_cache_ttl_hours: int = 24


@dataclass
class LLMConfig:
    enabled: bool = False
    mode: str = "template"
    qwen_model: str = "Qwen/Qwen3-8B"
    qwen_embedding_path: str = "models/Qwen3-Embedding-4B"
    qwen_reranker_path: str = "models/Qwen3-Reranker-4B"
    use_lora: bool = False
    lora_path: Optional[str] = None
    use_gpu: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7


@dataclass
class FallbackConfig:
    allow_without_embeddings: bool = True
    min_candidates_for_planning: int = 5
    llm_to_template: bool = True


@dataclass
class RuntimeConfig:
    runtime: Dict[str, Any] = field(default_factory=lambda: {"seed": 2026, "log_level": "INFO"})
    paths: PathsConfig = field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    recall: RecallConfig = field(default_factory=RecallConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    project_root: Path = PROJECT_ROOT

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_runtime_config(config_path: str = "configs/runtime.yaml", overrides: Dict[str, Any] | None = None) -> RuntimeConfig:
    cfg_path = PROJECT_ROOT / config_path
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_update(data, overrides)

    runtime_cfg = RuntimeConfig(
        runtime=data.get("runtime", {"seed": 2026, "log_level": "INFO"}),
        paths=PathsConfig(**data.get("paths", {})),
        embedding=EmbeddingConfig(**data.get("embedding", {})),
        recall=RecallConfig(**data.get("recall", {})),
        rerank=RerankConfig(**data.get("rerank", {})),
        planner=PlannerConfig(**data.get("planner", {})),
        llm=LLMConfig(**data.get("llm", {})),
        fallback=FallbackConfig(**data.get("fallback", {})),
    )
    return runtime_cfg
