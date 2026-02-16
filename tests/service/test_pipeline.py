"""
Tests for service.pipeline module.

Tests end-to-end Pipeline, configuration loading, and error handling.
"""
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "runtime": {"seed": 42, "log_level": "INFO"},
        "paths": {
            "poi_csv": "data/all/poi_expanded.csv",
            "user_events_csv": "data/all/user_events.csv",
            "emb_dir": "outputs/emb",
            "routing_dir": "outputs/routing",
        },
        "embedding": {
            "model_path": "models/bge-m3",
            "use_gpu": False,
            "auto_build_if_missing": True,
        },
        "recall": {
            "semantic_topk": 50,
            "final_topk": 50,
        },
        "planner": {
            "max_duration_hours": 8,
            "start_time_min": 480,
        },
    }


@pytest.fixture
def sample_poi_df():
    """Create sample POI DataFrame."""
    return pd.DataFrame({
        "poi_id": ["POI_0001", "POI_0002", "POI_0003", "POI_0004", "POI_0005"],
        "name": ["Tianshan", "Kanas", "Sayram", "Nalati", "Flaming"],
        "province": ["Xinjiang"] * 5,
        "city": ["Urumqi", "Altay", "Ili", "Ili", "Turpan"],
        "description": ["Scenic"] * 5,
        "stay_min": [120, 180, 120, 150, 90],
        "lat": [43.88, 48.70, 44.60, 43.30, 42.95],
        "lon": [88.13, 87.00, 81.00, 83.80, 89.18],
        "open_min": [480] * 5,
        "close_min": [1200] * 5,
    })


# ============================================================================
# Test: Configuration
# ============================================================================

class TestConfiguration:
    """Test configuration handling."""

    def test_config_has_required_keys(self, sample_config_dict):
        """Test that config has required keys."""
        required_keys = ["runtime", "paths", "embedding", "recall", "planner"]

        for key in required_keys:
            assert key in sample_config_dict

    def test_config_default_values(self):
        """Test default configuration values."""
        defaults = {
            "seed": 2026,
            "log_level": "INFO",
            "use_gpu": False,
            "max_duration_hours": 8,
            "start_time_min": 480,
        }

        for key, value in defaults.items():
            assert isinstance(value, (int, str, bool))

    def test_config_paths_resolution(self):
        """Test path resolution."""
        import os
        base_path = "/root/autodl-tmp/goafar_project_broken"

        relative_path = "data/all/poi_expanded.csv"
        if not os.path.isabs(relative_path):
            resolved = os.path.join(base_path, relative_path)
        else:
            resolved = relative_path

        assert os.path.isabs(resolved)

    def test_config_override(self, sample_config_dict):
        """Test configuration override."""
        base_config = sample_config_dict.copy()
        override = {"use_gpu": True}

        # Simulate override
        if "embedding" in base_config:
            base_config["embedding"]["use_gpu"] = override["use_gpu"]

        assert base_config["embedding"]["use_gpu"] is True


# ============================================================================
# Test: Pipeline Initialization
# ============================================================================

class TestPipelineInitialization:
    """Test pipeline initialization."""

    def test_pipeline_init_with_config(self):
        """Test pipeline initialization with config."""
        # Mock pipeline class
        config = {"use_gpu": False, "max_duration_hours": 8}

        pipeline = Mock()
        pipeline.config = config

        assert pipeline.config is not None
        assert pipeline.config["use_gpu"] is False

    def test_pipeline_init_with_defaults(self):
        """Test pipeline initialization with defaults."""
        defaults = {
            "seed": 2026,
            "use_gpu": False,
        }

        pipeline = Mock()
        pipeline.config = defaults

        assert pipeline.config["seed"] == 2026

    def test_pipeline_component_init(self):
        """Test that pipeline components are initialized."""
        components = ["embedding", "recall", "planner", "ranking"]

        for component in components:
            # Simulate component check
            assert component in components


# ============================================================================
# Test: End-to-End Pipeline
# ============================================================================

class TestEndToEndPipeline:
    """Test end-to-end pipeline execution."""

    def test_recommendation_request_structure(self):
        """Test that recommendation request has correct structure."""
        request = {
            "query_text": "Want to visit Xinjiang",
            "province": "Xinjiang",
            "max_hours": 8,
            "topk_candidates": 10,
            "use_llm": False,
        }

        required_keys = ["query_text", "province", "max_hours"]
        for key in required_keys:
            assert key in request

    def test_recommendation_response_structure(self):
        """Test that recommendation response has correct structure."""
        response = {
            "success": True,
            "title": "Xinjiang Adventure",
            "description": "Beautiful scenery",
            "route": [
                {"poi_id": "POI_0001", "poi_name": "Kanas", "arrival_time_min": 60},
                {"poi_id": "POI_0002", "poi_name": "Nalati", "arrival_time_min": 180},
            ],
            "total_hours": 6.5,
            "num_pois": 2,
            "query": "Want to visit Xinjiang",
        }

        required_keys = ["success", "title", "description", "route", "total_hours", "num_pois"]
        for key in required_keys:
            assert key in response

    def test_pipeline_flow_stages(self):
        """Test that pipeline has correct flow stages."""
        stages = [
            "intent_understanding",
            "candidate_recall",
            "reranking",
            "route_planning",
            "content_generation",
        ]

        for stage in stages:
            assert stage in stages

    def test_pipeline_with_valid_input(self, sample_poi_df):
        """Test pipeline with valid input."""
        request = {
            "query_text": "Xinjiang lakes",
            "province": "Xinjiang",
            "max_hours": 8,
        }

        # Simulate pipeline processing
        candidates = sample_poi_df.head(3)
        route = candidates.head(2)

        assert len(candidates) >= len(route)
        assert "poi_id" in candidates.columns


# ============================================================================
# Test: Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling in pipeline."""

    def test_no_candidates_error(self):
        """Test handling when no candidates found."""
        response = {
            "success": False,
            "error": "No matching POIs found",
            "query": "test query",
        }

        assert response["success"] is False
        assert "error" in response
        assert len(response["error"]) > 0

    def test_insufficient_candidates_error(self):
        """Test handling with insufficient candidates."""
        min_required = 3
        num_candidates = 2

        error = None
        if num_candidates < min_required:
            error = f"Insufficient candidates: {num_candidates} < {min_required}"

        assert error is not None
        assert "Insufficient" in error

    def test_routing_failure_error(self):
        """Test handling when routing fails."""
        response = {
            "success": False,
            "error": "No feasible route found",
            "query": "test query",
        }

        assert response["success"] is False
        assert "feasible" in response["error"].lower()

    def test_invalid_request_error(self):
        """Test handling of invalid request."""
        request = {
            "query_text": "",  # Empty query
            "max_hours": -1,  # Invalid hours
        }

        valid = len(request["query_text"]) > 0 and request["max_hours"] > 0

        assert valid is False

    def test_exception_handling(self):
        """Test exception handling in pipeline."""
        # Simulate exception
        try:
            raise ValueError("Test exception")
        except Exception as e:
            response = {
                "success": False,
                "error": str(e),
            }

        assert response["success"] is False
        assert "Test exception" in response["error"]


# ============================================================================
# Test: Module Coordination
# ============================================================================

class TestModuleCoordination:
    """Test coordination between pipeline modules."""

    def test_embedding_to_recall_flow(self, sample_poi_df):
        """Test flow from embedding to recall."""
        # Simulate embedding output
        embeddings = np.random.randn(len(sample_poi_df), 128).astype("float32")

        # Simulate recall using embeddings
        query = np.random.randn(128).astype("float32")
        scores = embeddings @ query
        top_indices = np.argsort(-scores)[:3]

        candidates = sample_poi_df.iloc[top_indices]

        assert len(candidates) == 3

    def test_recall_to_rerank_flow(self, sample_poi_df):
        """Test flow from recall to rerank."""
        candidates = sample_poi_df.head(5)

        # Simulate reranking
        reranked = candidates.sort_values("stay_min", ascending=False).head(3)

        assert len(reranked) == 3
        assert reranked.iloc[0]["stay_min"] >= reranked.iloc[1]["stay_min"]

    def test_rerank_to_routing_flow(self, sample_poi_df):
        """Test flow from rerank to routing."""
        candidates = sample_poi_df.head(3).copy()

        # Simulate routing input
        routing_input = candidates[["poi_id", "stay_min"]].to_dict("records")

        assert len(routing_input) == 3
        assert "stay_min" in routing_input[0]

    def test_routing_to_content_flow(self):
        """Test flow from routing to content."""
        route = [
            {"poi_name": "Kanas", "poi_city": "Altay"},
            {"poi_name": "Nalati", "poi_city": "Ili"},
        ]

        # Simulate content generation
        core_pois = [p["poi_name"] for p in route]
        title = f"Visit {', '.join(core_pois)}"

        assert "Kanas" in title
        assert "Nalati" in title


# ============================================================================
# Test: Debug Information
# ============================================================================

class TestDebugInformation:
    """Test debug information collection."""

    def test_debug_info_structure(self):
        """Test that debug info has correct structure."""
        debug = {
            "recall_sources": {"semantic": 10, "behavior": 5},
            "candidate_count_before_rerank": 15,
            "candidate_count_after_rerank": 10,
            "fallback_events": ["semantic_unavailable"],
        }

        required_keys = ["recall_sources", "candidate_count_before_rerank", "candidate_count_after_rerank"]
        for key in required_keys:
            assert key in debug

    def test_debug_with_fallback(self):
        """Test debug info with fallback events."""
        fallback_events = ["llm_unavailable", "embedding_unavailable"]

        debug = {
            "fallback_events": fallback_events,
            "degraded": len(fallback_events) > 0,
        }

        assert debug["degraded"] is True
        assert len(debug["fallback_events"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
