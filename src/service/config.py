"""
Unified runtime configuration loader.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config_loader import load_yaml_with_env


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathsConfig:
    poi_csv: str = "data/all/poi_with_coords.csv"
    user_events_csv: str = "data/all/user_events.csv"
    emb_dir: str = "outputs/emb"
    routing_dir: str = "outputs/routing"
    cache_dir: str = "outputs/cache"
    results_dir: str = "outputs/results"
    shp_dir: str = "data/external/shengfen"
    yelp_dir: str = "data/external/yelp"
    geolife_dir: str = "data/external/Geolife Trajectories 1.3"


@dataclass
class EmbeddingConfig:
    model_path: str = "models/Qwen3-Embedding-4B"
    fallback_model: str = "models/Qwen3-Embedding-4B"
    use_gpu: bool = True
    auto_build_if_missing: bool = True
    backend: str = "auto"
    faiss_index_file: str = "outputs/emb/poi_faiss.index"
    batch_size: int = 32
    quantization: str = "none"


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
    qwen_model: str = "models/Qwen3-8B"
    qwen_embedding_path: str = "models/Qwen3-Embedding-4B"
    qwen_reranker_path: str = "models/Qwen3-Reranker-4B"
    use_lora: bool = False
    lora_path: Optional[str] = None
    use_gpu: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    inference_max_length: int = 2048
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class FallbackConfig:
    allow_without_embeddings: bool = True
    min_candidates_for_planning: int = 5
    llm_to_template: bool = True


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    log_dir: str = "logs"
    log_file: Optional[str] = None
    console_output: bool = True
    file_output: bool = True
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    use_colors: bool = True
    max_file_size_mb: int = 10  # yaml config uses this name
    format: str = "text"
    sanitize_secrets: bool = True


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
    log: LogConfig = field(default_factory=LogConfig)
    logging: LogConfig = field(default_factory=LogConfig)  # Alias for backward compatibility
    project_root: Path = PROJECT_ROOT

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.project_root / path


def _coerce_scalar(value: Any) -> Any:
    """将 ${...} 展开后的字符串转换为基础类型。"""
    if not isinstance(value, str):
        return value

    raw = value.strip()
    lowered = raw.lower()

    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"[-+]?\d+", raw):
        return int(raw)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", raw):
        return float(raw)

    return value


def _coerce_recursive(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _coerce_recursive(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_coerce_recursive(item) for item in data]
    return _coerce_scalar(data)


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_runtime_config(config_path: str = "configs/runtime.yaml", overrides: Dict[str, Any] | None = None) -> RuntimeConfig:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / config_path

    data: Dict[str, Any] = {}
    if cfg_path.exists():
        data = load_yaml_with_env(
            config_path=cfg_path,
            env=os.environ.get("GOAFAR_ENV_MODE", "dev"),
            apply_overrides=True,
        )

    if overrides:
        data = _deep_update(data, overrides)

    # Load logging config (support both "logging" and "log" keys)
    logging_data = _coerce_recursive(data.get("logging", data.get("log", {})))
    paths_data = _coerce_recursive(data.get("paths", {}))
    embedding_data = _coerce_recursive(data.get("embedding", {}))
    recall_data = _coerce_recursive(data.get("recall", {}))
    rerank_data = _coerce_recursive(data.get("rerank", {}))
    planner_data = _coerce_recursive(data.get("planner", {}))
    llm_data = _coerce_recursive(data.get("llm", {}))
    fallback_data = _coerce_recursive(data.get("fallback", {}))
    runtime_data = _coerce_recursive(data.get("runtime", {"seed": 2026, "log_level": "INFO"}))
    log_cfg = LogConfig(**logging_data)

    runtime_cfg = RuntimeConfig(
        runtime=runtime_data,
        paths=PathsConfig(**paths_data),
        embedding=EmbeddingConfig(**embedding_data),
        recall=RecallConfig(**recall_data),
        rerank=RerankConfig(**rerank_data),
        planner=PlannerConfig(**planner_data),
        llm=LLMConfig(**llm_data),
        fallback=FallbackConfig(**fallback_data),
        log=log_cfg,
        logging=log_cfg,
    )
    return runtime_cfg
