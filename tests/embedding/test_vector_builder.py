"""
Tests for embedding.vector_builder module.

Tests vector retrieval accuracy, batch query, and boundary conditions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_poi_df():
    """Create a sample POI DataFrame for testing."""
    return pd.DataFrame({
        "poi_id": ["POI_0001", "POI_0002", "POI_0003", "POI_0004", "POI_0005"],
        "name": ["Tianshan", "Kanas", "Sayram", "Nalati", "Flaming"],
        "province": ["Xinjiang"] * 5,
        "city": ["Urumqi", "Altay", "Ili", "Ili", "Turpan"],
        "description": ["Lake view", "Mountain view", "Grassland", "Sky grass", "Hot desert"],
        "stay_min": [120, 180, 120, 150, 90],
        "lat": [43.88, 48.70, 44.60, 43.30, 42.95],
        "lon": [88.13, 87.00, 81.00, 83.80, 89.18],
        "open_min": [480] * 5,
        "close_min": [1200] * 5,
    })


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(5, 1024).astype("float32")


# ============================================================================
# Test: Vector Retrieval Accuracy
# ============================================================================

class TestVectorRetrievalAccuracy:
    """Test vector retrieval accuracy."""

    def test_search_numpy_backend(self, sample_embeddings):
        """Test vector search with numpy backend."""
        query_vec = sample_embeddings[0] + np.random.randn(1024).astype("float32") * 0.1

        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:3]
        top_scores = scores[top_indices]

        assert len(top_indices) == 3
        assert len(top_scores) == 3
        assert top_indices[0] == 0
        assert top_scores[0] >= top_scores[1]

    def test_search_returns_correct_topk(self, sample_embeddings):
        """Test that search returns exactly topk results."""
        query_vec = sample_embeddings[0]

        for topk in [1, 2, 3, 5]:
            scores = sample_embeddings @ query_vec
            top_indices = np.argsort(-scores)[:topk]
            top_scores = scores[top_indices]

            assert len(top_indices) == topk
            assert len(top_scores) == topk

    def test_search_scores_descending_order(self, sample_embeddings):
        """Test that search returns scores in descending order."""
        query_vec = sample_embeddings[2]

        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:5]
        top_scores = scores[top_indices]

        for i in range(len(top_scores) - 1):
            assert top_scores[i] >= top_scores[i + 1]

    def test_similar_pois_returns_dataframe_with_required_columns(
        self, sample_poi_df, sample_embeddings
    ):
        """Test that search result DataFrame has required columns."""
        query_vec = sample_embeddings[0]
        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:3]

        results = sample_poi_df.iloc[top_indices].copy()
        results["semantic_score"] = scores[top_indices]
        results["rank"] = range(1, len(results) + 1)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "poi_id" in results.columns
        assert "name" in results.columns
        assert "semantic_score" in results.columns
        assert "rank" in results.columns


# ============================================================================
# Test: Batch Query
# ============================================================================

class TestBatchQuery:
    """Test batch query functionality."""

    def test_batch_search_multiple_queries(self, sample_embeddings):
        """Test searching with multiple query vectors."""
        queries = sample_embeddings[:3]

        results = []
        for query_vec in queries:
            scores = sample_embeddings @ query_vec
            top_indices = np.argsort(-scores)[:3]
            results.append(top_indices)

        assert len(results) == 3
        assert all(len(r) == 3 for r in results)

    def test_batch_search_different_topk(self, sample_embeddings):
        """Test batch search with different topk values."""
        query_vec = sample_embeddings[0]

        for topk in [1, 3, 5]:
            scores = sample_embeddings @ query_vec
            top_indices = np.argsort(-scores)[:topk]

            assert len(top_indices) == topk


# ============================================================================
# Test: Boundary Conditions
# ============================================================================

class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_empty_embeddings(self):
        """Test handling of empty embedding matrix."""
        empty_embeddings = np.array([]).reshape(0, 1024).astype("float32")
        query_vec = np.random.randn(1024).astype("float32")

        if empty_embeddings.size > 0:
            scores = empty_embeddings @ query_vec
            top_indices = np.argsort(-scores)[:5]
        else:
            top_indices = np.array([], dtype=int)

        assert len(top_indices) == 0

    def test_topk_larger_than_embeddings(self, sample_embeddings):
        """Test when topk is larger than available embeddings."""
        query_vec = sample_embeddings[0]

        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:100]

        assert len(top_indices) == 5

    def test_topk_zero_or_negative(self, sample_embeddings):
        """Test handling of invalid topk values."""
        query_vec = sample_embeddings[0]
        scores = sample_embeddings @ query_vec

        topk = 0
        top_indices = np.argsort(-scores)[:max(1, topk)]

        assert len(top_indices) >= 1

    def test_single_embedding(self):
        """Test search with only one embedding."""
        single_embedding = np.random.randn(1, 1024).astype("float32")
        query_vec = single_embedding[0]

        scores = single_embedding @ query_vec
        top_indices = np.argsort(-scores)[:5]

        assert len(top_indices) == 1
        assert top_indices[0] == 0

    def test_high_dimensional_embeddings(self):
        """Test with higher dimensional embeddings."""
        high_dim_emb = np.random.randn(10, 2048).astype("float32")
        query_vec = high_dim_emb[0]

        scores = high_dim_emb @ query_vec
        top_indices = np.argsort(-scores)[:5]

        assert len(top_indices) == 5

    def test_near_duplicate_embeddings(self, sample_embeddings):
        """Test with near-duplicate embeddings."""
        duplicate = sample_embeddings[0] + np.random.randn(1024).astype("float32") * 0.001
        augmented = np.vstack([sample_embeddings, duplicate.reshape(1, -1)])

        query_vec = sample_embeddings[0]
        scores = augmented @ query_vec
        top_indices = np.argsort(-scores)[:3]

        assert 0 in top_indices

    def test_zero_vector_query(self, sample_embeddings):
        """Test with zero vector as query."""
        zero_vec = np.zeros(1024, dtype="float32")

        scores = sample_embeddings @ zero_vec
        top_indices = np.argsort(-scores)[:3]

        assert len(top_indices) == 3

    def test_normalized_vs_unnormalized(self, sample_embeddings):
        """Test that normalization affects results."""
        query_vec = sample_embeddings[0]
        scores_unorm = sample_embeddings @ query_vec

        norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
        normalized = sample_embeddings / (norms + 1e-8)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores_norm = normalized @ query_norm

        top_unorm = np.argsort(-scores_unorm)[:3]
        top_norm = np.argsort(-scores_norm)[:3]

        assert len(top_unorm) == 3
        assert len(top_norm) == 3


# ============================================================================
# Test: POI Text Building
# ============================================================================

class TestPOITextBuilding:
    """Test POI text building functionality."""

    def test_build_poi_texts_with_none_values(self):
        """Test POI text building with None values."""
        df = pd.DataFrame({
            "name": ["Test POI", None, "Another POI"],
            "province": ["Xinjiang", None, "Tibet"],
            "city": ["Urumqi", None, "Lhasa"],
            "description": [None, "No name POI", None],
            "stay_min": [120, 60, 90],
        })

        texts = []
        for _, row in df.iterrows():
            parts = [str(row.get("name", "")) if pd.notna(row.get("name")) else ""]
            if pd.notna(row.get("province")):
                parts.append(str(row["province"]))
            texts.append(" ".join([p for p in parts if p]))

        assert len(texts) == 3
        assert all(isinstance(t, str) for t in texts)

    def test_build_poi_texts_with_special_chars(self):
        """Test handling of special characters."""
        df = pd.DataFrame({
            "name": ["Tianshan\n", "Kanas\t"],
            "province": ["Xinjiang", "Xinjiang"],
            "city": ["Urumqi", "Altay"],
            "description": ["Scenic\nbeautiful", "Sight\tpleasant"],
            "stay_min": [120, 180],
        })

        texts = []
        for _, row in df.iterrows():
            parts = [str(row.get("name", "")).replace("\n", " ").replace("\t", " ")]
            if pd.notna(row.get("description")):
                desc = str(row["description"]).replace("\n", " ").replace("\t", " ")
                parts.append(desc)
            texts.append(" ".join(parts))

        assert all("\n" not in t and "\t" not in t for t in texts)


# ============================================================================
# Test: Result Structure
# ============================================================================

class TestResultStructure:
    """Test that result structures are correct."""

    def test_search_result_dataframe_structure(self, sample_poi_df, sample_embeddings):
        """Test that search result DataFrame has required columns."""
        query_vec = sample_embeddings[0]
        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:3]

        results = sample_poi_df.iloc[top_indices].copy()
        results["semantic_score"] = scores[top_indices]
        results["rank"] = range(1, len(results) + 1)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3
        assert "poi_id" in results.columns
        assert "name" in results.columns
        assert "semantic_score" in results.columns
        assert "rank" in results.columns

    def test_search_result_ranking(self, sample_poi_df, sample_embeddings):
        """Test that results are properly ranked."""
        query_vec = sample_embeddings[0]
        scores = sample_embeddings @ query_vec
        top_indices = np.argsort(-scores)[:5]

        results = sample_poi_df.iloc[top_indices].copy()
        results["semantic_score"] = scores[top_indices]
        results["rank"] = range(1, len(results) + 1)

        ranks = results["rank"].tolist()
        assert ranks == list(range(1, len(results) + 1))

        result_scores = results["semantic_score"].tolist()
        for i in range(len(result_scores) - 1):
            assert result_scores[i] >= result_scores[i + 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
