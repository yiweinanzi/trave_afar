"""
GNN Integration with GoAfar Recommendation Pipeline

This module provides integration between the GNN model and the existing
GoAfar recommendation system. It implements a provider pattern similar
to RecBoleProvider for seamless integration.

Usage:
    # In recommendation pipeline
    gnn_provider = GNNProvider(model_path="outputs/gnn/model.pkl")
    candidates_df, metadata = gnn_provider.predict(user_id, top_k, poi_df)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.model.gnn_model import (
    POIGNNRecommender,
    GNNConfig,
    POIGraphBuilder,
    create_gnn_model,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GNNProvider:
    """
    GNN-based POI recommendation provider.

    Provides a unified interface compatible with the GoAfar recommendation
    pipeline. Supports:

    - POI embedding based similarity
    - Graph walk recommendations
    - User preference scoring
    - Seamless integration with candidate merger

    Usage:
        provider = GNNProvider(
            model_path="outputs/gnn/model.pkl",
            enabled=True,
        )

        candidates_df, metadata = provider.predict(
            user_id="user_123",
            top_k=30,
            poi_df=poi_data,
        )
    """

    def __init__(
        self,
        model_path: Optional[str] = "outputs/gnn/model.pkl",
        enabled: bool = True,
        fallback_to_popular: bool = True,
        similarity_weight: float = 0.7,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize GNN Provider.

        Args:
            model_path: Path to trained GNN model
            enabled: Whether to use GNN recommendations
            fallback_to_popular: Fallback to popularity if model unavailable
            similarity_weight: Weight for similarity scoring
            diversity_weight: Weight for diversity scoring
        """
        self.model_path = Path(model_path) if model_path else None
        self.enabled = enabled
        self.fallback_to_popular = fallback_to_popular
        self.similarity_weight = similarity_weight
        self.diversity_weight = diversity_weight

        self.recommender: Optional[POIGNNRecommender] = None
        self.available = False
        self.user_history: Dict[str, List[str]] = {}

        # Load model
        if enabled:
            self._load_model()

    def _load_model(self) -> None:
        """Load GNN model from disk."""
        if self.model_path is None or not self.model_path.exists():
            logger.warning(f"GNN model not found: {self.model_path}")
            if self.fallback_to_popular:
                logger.info("Will fallback to popularity-based recommendations")
            return

        try:
            self.recommender = create_gnn_model()
            self.recommender.load_model(str(self.model_path))
            self.available = True
            logger.info(f"GNN model loaded from {self.model_path}")
            logger.info(f"  Embeddings: {self.recommender.embeddings.shape if self.recommender.embeddings is not None else 0}")
        except Exception as e:
            logger.error(f"Failed to load GNN model: {e}")
            if self.fallback_to_popular:
                logger.info("Will fallback to popularity-based recommendations")

    def update_user_history(self, user_id: str, poi_id: str) -> None:
        """
        Update user interaction history.

        Args:
            user_id: User identifier
            poi_id: POI identifier
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        self.user_history[user_id].append(poi_id)

    def get_user_history(self, user_id: str) -> List[str]:
        """
        Get user interaction history.

        Args:
            user_id: User identifier

        Returns:
            List of POI IDs the user has interacted with
        """
        return self.user_history.get(user_id, [])

    def predict(
        self,
        user_id: str,
        top_k: int = 30,
        poi_df: Optional[pd.DataFrame] = None,
        filter_history: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate POI recommendations for a user.

        Args:
            user_id: User identifier
            top_k: Number of recommendations to return
            poi_df: Available POI data
            filter_history: Whether to filter out visited POIs

        Returns:
            (recommendations_df, metadata)
        """
        metadata = {
            "user_id": user_id,
            "method": "none",
            "num_visited": 0,
            "filtered": False,
        }

        # Check availability
        if not self.available:
            if self.fallback_to_popular:
                return self._predict_by_popularity(top_k, poi_df, metadata)
            return pd.DataFrame(columns=["poi_id", "gnn_score"]), metadata

        # Get user history
        visited = self.get_user_history(user_id)
        metadata["num_visited"] = len(visited)

        # Build candidate POIs
        candidates = self._build_candidates(poi_df, visited if filter_history else None)
        if not candidates:
            logger.warning(f"No candidates available for user {user_id}")
            return pd.DataFrame(columns=["poi_id", "gnn_score"]), metadata

        # Get recommendations
        predictions = self.recommender.predict(
            user_id=user_id,
            candidate_pois=candidates,
            context={"visited_pois": visited},
        )

        # Build result DataFrame
        rec_df = pd.DataFrame([
            {"poi_id": poi_id, **scores}
            for poi_id, scores in predictions
        ])

        # Normalize scores
        if "gnn_score" in rec_df.columns and len(rec_df) > 0:
            max_score = rec_df["gnn_score"].max()
            if max_score > 0:
                rec_df["gnn_score"] = rec_df["gnn_score"] / max_score
            else:
                rec_df["gnn_score"] = 1.0

        # Limit to top_k
        rec_df = rec_df.head(top_k).reset_index(drop=True)

        metadata["method"] = "gnn"
        return rec_df, metadata

    def _build_candidates(
        self,
        poi_df: Optional[pd.DataFrame],
        exclude_ids: Optional[set] = None,
    ) -> List[Dict]:
        """Build candidate POI list."""
        candidates = []
        exclude = exclude_ids or set()

        if poi_df is not None and len(poi_df) > 0:
            # Use provided POI data
            for _, row in poi_df.iterrows():
                poi_id = str(row.get("poi_id", row.get("osm_id", "")))
                if poi_id and poi_id not in exclude:
                    candidates.append({
                        "poi_id": poi_id,
                        "name": row.get("name", ""),
                        "fclass": row.get("fclass", "unknown"),
                        "province": row.get("province", ""),
                        "latitude": row.get("lat", 0),
                        "longitude": row.get("lon", 0),
                        "popularity": row.get("popularity", 0),
                    })
        else:
            # Use all POIs from graph
            if self.recommender and self.recommender.builder.nodes:
                for poi_id, node in self.recommender.builder.nodes.items():
                    if poi_id not in exclude:
                        candidates.append({
                            "poi_id": poi_id,
                            "name": node.get("name", ""),
                            "fclass": node.get("fclass", ""),
                            "province": node.get("province", ""),
                            "latitude": node.get("latitude", 0),
                            "longitude": node.get("longitude", 0),
                            "popularity": node.get("population", 0),
                        })

        return candidates

    def _predict_by_popularity(
        self,
        top_k: int,
        poi_df: Optional[pd.DataFrame],
        metadata: Dict,
    ) -> Tuple[pd.DataFrame, Dict]:
        """Fallback popularity-based recommendations."""
        metadata["method"] = "popularity"

        candidates = self._build_candidates(poi_df)

        # Sort by popularity
        candidates.sort(key=lambda x: x.get("popularity", 0), reverse=True)

        # Build DataFrame
        rec_df = pd.DataFrame([
            {"poi_id": c["poi_id"], "gnn_score": min(c.get("popularity", 0) / 100, 1.0)}
            for c in candidates[:top_k]
        ])

        return rec_df, metadata

    def get_similar_pois(
        self,
        poi_id: str,
        top_k: int = 10,
        exclude_ids: Optional[set] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find similar POIs.

        Args:
            poi_id: Query POI ID
            top_k: Number of results
            exclude_ids: POI IDs to exclude

        Returns:
            List of (poi_id, similarity) tuples
        """
        if not self.available:
            return []

        return self.recommender.get_similar_pois(
            poi_id,
            top_k=top_k,
            exclude_ids=exclude_ids,
        )

    def graph_walk(
        self,
        start_poi: str,
        num_steps: int = 3,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations via graph walk.

        Args:
            start_poi: Starting POI ID
            num_steps: Number of steps
            top_k: Number of results

        Returns:
            List of (poi_id, score) tuples
        """
        if not self.available:
            return []

        return self.recommender.graph_walk_recommend(
            start_poi,
            num_steps=num_steps,
            top_k=top_k,
        )

    def get_poi_embedding(self, poi_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a POI.

        Args:
            poi_id: POI identifier

        Returns:
            Embedding vector or None
        """
        if not self.available:
            return None

        return self.recommender.get_poi_embedding(poi_id)

    def export_embeddings(
        self,
        output_path: str,
        format: str = "npy",
    ) -> None:
        """
        Export POI embeddings.

        Args:
            output_path: Output file path
            format: Format (npy, csv, json)
        """
        if not self.available:
            logger.warning("No embeddings to export")
            return

        import json

        if format == "npy":
            np.save(output_path, self.recommender.embeddings)
        elif format == "csv":
            import pandas as pd
            df_data = []
            for poi_id, idx in self.recommender.poi_id_map.items():
                emb = self.recommender.embeddings[idx]
                row = {"poi_id": poi_id}
                row.update({f"dim_{i}": v for i, v in enumerate(emb)})
                df_data.append(row)
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
        elif format == "json":
            data = {}
            for poi_id, idx in self.recommender.poi_id_map.items():
                emb = self.recommender.embeddings[idx]
                data[poi_id] = emb.tolist()
            with open(output_path, "w") as f:
                json.dump(data, f)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Embeddings exported to {output_path}")


class GNNScorer:
    """
    GNN-based scorer for re-ranking POI candidates.

    Can be used in the re-ranking stage to score candidates
    based on graph structure and embeddings.
    """

    def __init__(
        self,
        provider: GNNProvider,
        score_weight: float = 0.5,
    ):
        """
        Initialize GNN scorer.

        Args:
            provider: GNN recommendation provider
            score_weight: Weight for GNN score in final ranking
        """
        self.provider = provider
        self.score_weight = score_weight

    def score(
        self,
        user_id: str,
        candidates: List[Dict],
        original_scores: Optional[List[float]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Score and re-rank candidates.

        Args:
            user_id: User identifier
            candidates: List of candidate POI dicts
            original_scores: Original scores from previous stage

        Returns:
            List of (poi_id, final_score) tuples
        """
        if not self.provider.available:
            # Return original scores
            if original_scores:
                return [(c.get("poi_id", ""), s) for c, s in zip(candidates, original_scores)]
            return [(c.get("poi_id", ""), 0.0) for c in candidates]

        # Get GNN scores
        _, gnn_df = self.provider.predict(
            user_id=user_id,
            top_k=len(candidates),
            poi_df=pd.DataFrame(candidates),
        )

        # Build score map
        gnn_scores = dict(zip(gnn_df["poi_id"], gnn_df.get("gnn_score", 0)))

        # Combine scores
        results = []
        for i, candidate in enumerate(candidates):
            poi_id = candidate.get("poi_id", "")
            gnn_score = gnn_scores.get(poi_id, 0.0)

            if original_scores:
                orig_score = original_scores[i] / max(original_scores) if max(original_scores) > 0 else 0
            else:
                orig_score = 0.5

            # Weighted combination
            final_score = (
                self.score_weight * gnn_score +
                (1 - self.score_weight) * orig_score
            )

            results.append((poi_id, final_score))

        # Sort by final score
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def create_gnn_provider(
    model_path: str = "outputs/gnn/model.pkl",
    enabled: bool = True,
    **kwargs
) -> GNNProvider:
    """
    Create a GNN recommendation provider.

    Args:
        model_path: Path to trained model
        enabled: Whether to enable GNN
        **kwargs: Additional arguments for GNNProvider

    Returns:
        Configured GNNProvider
    """
    return GNNProvider(
        model_path=model_path,
        enabled=enabled,
        **kwargs
    )
