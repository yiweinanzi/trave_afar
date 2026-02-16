"""
Tests for env override behavior in service.config_loader.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.service.config_loader import apply_env_overrides


def _base_config():
    return {
        "llm": {
            "qwen_model": "models/Qwen3-8B",
            "use_gpu": True,
            "enabled": True,
        },
        "embedding": {
            "model_path": "models/Qwen3-Embedding-4B",
            "use_gpu": True,
        },
        "rerank": {
            "qwen_reranker_path": "models/Qwen3-Reranker-4B",
            "use_reranker_model": True,
        },
        "logging": {
            "max_file_size_mb": 10,
        },
    }


def test_apply_env_overrides_supports_alias_keys(monkeypatch):
    monkeypatch.setenv("GOAFAR_LLM_MODEL", "models/Custom-Qwen")
    monkeypatch.setenv("GOAFAR_RERANK_PATH", "models/Custom-Reranker")
    monkeypatch.setenv("GOAFAR_EMBEDDING_MODEL", "models/Custom-Embedding")
    monkeypatch.setenv("GOAFAR_LOG_MAX_SIZE", "20")

    cfg = apply_env_overrides(_base_config())

    assert cfg["llm"]["qwen_model"] == "models/Custom-Qwen"
    assert cfg["rerank"]["qwen_reranker_path"] == "models/Custom-Reranker"
    assert cfg["embedding"]["model_path"] == "models/Custom-Embedding"
    assert cfg["logging"]["max_file_size_mb"] == 20


def test_apply_env_overrides_ignores_unknown_keys(monkeypatch):
    monkeypatch.setenv("GOAFAR_LLM_USE_GPU", "false")
    monkeypatch.setenv("GOAFAR_NOT_A_REAL_OVERRIDE", "123")

    cfg = apply_env_overrides(_base_config())

    assert cfg["llm"]["use_gpu"] is False
    assert "not" not in cfg
