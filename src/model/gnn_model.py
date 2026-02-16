"""
Graph Neural Network Model for POI Recommendation in GoAfar.

This module implements GraphSAGE-based GNN models for:
- POI representation learning
- Link prediction (POI-POI relationships)
- Node classification (POI category prediction)
- Graph contrastive learning
- Recommendation via graph walk

Author: GoAfar Team
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GNNConfig:
    """Configuration for GNN model training and inference."""

    # Model architecture
    model_type: str = "graphsage"  # graphsage, gat, gcn
    num_layers: int = 2
    hidden_dim: int = 128
    output_dim: int = 64
    dropout: float = 0.1

    # GraphSAGE specific
    aggregator: str = "mean"  # mean, add, max

    # GAT specific
    attention_heads: int = 4

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 50
    early_stop_patience: int = 10

    # Loss weights
    link_pred_weight: float = 1.0
    node_cls_weight: float = 0.5
    contrastive_weight: float = 0.3
    contrastive_temperature: float = 0.5

    # Graph construction
    cooccurrence_window: int = 5  # For route-based co-occurrence
    geo_radius_km: float = 1.0  # For geographic proximity edges
    category_similarity_threshold: float = 0.8  # For category similarity edges

    # Device
    device: str = "cpu"

    # Paths
    graph_dir: str = "outputs/gnn"
    cache_dir: str = "outputs/gnn/cache"


@dataclass
class GraphMetrics:
    """Metrics tracked during GNN training."""
    epoch: int
    train_loss: float
    train_link_loss: float
    train_node_loss: float
    train_contrastive_loss: float
    val_loss: float = 0.0
    val_link_auc: float = 0.0
    val_node_acc: float = 0.0


# ============================================================================
# Graph Builder
# ============================================================================

class POIGraphBuilder:
    """
    Build POI graph from various data sources.

    Graph Structure:
    - Nodes: POIs
    - Edges:
        * Co-occurrence: POIs appearing together in routes
        * Geographic: Nearby POIs within radius
        * Category: Similar category POIs
    """

    def __init__(
        self,
        config: Optional[GNNConfig] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize graph builder.

        Args:
            config: GNN configuration
            cache_dir: Directory to cache built graphs
        """
        self.config = config or GNNConfig()
        self.cache_dir = Path(cache_dir or self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Graph data structures
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[str, List[Tuple]] = {
            "cooccurrence": [],
            "geographic": [],
            "category": [],
        }
        self.node_id_map: Dict[str, int] = {}
        self.id_node_map: Dict[int, str] = {}

        # Feature maps for encoding
        self.category_map: Dict[str, int] = {}
        self.province_map: Dict[str, int] = {}

        logger.info("POIGraphBuilder initialized")

    def add_pois_from_csv(
        self,
        poi_csv: Union[str, Path],
        provinces: Optional[List[str]] = None,
    ) -> int:
        """Load POIs from CSV files."""
        poi_path = Path(poi_csv)

        if poi_path.is_dir():
            all_pois = []
            for f in poi_path.glob("*_pois.csv"):
                if provinces:
                    province = f.stem.split("_")[0]
                    if province not in provinces:
                        continue
                all_pois.append(f)

            dfs = []
            for f in all_pois:
                df = pd.read_csv(f)
                dfs.append(df)

            if dfs:
                poi_df = pd.concat(dfs, ignore_index=True)
            else:
                logger.warning(f"No POI files found in {poi_path}")
                return 0
        else:
            poi_df = pd.read_csv(poi_csv)

        required_cols = ["osm_id"]
        if not all(col in poi_df.columns for col in required_cols):
            logger.error(f"POI CSV missing required columns: {required_cols}")
            return 0

        count = 0
        for _, row in poi_df.iterrows():
            poi_id = str(row["osm_id"])
            if poi_id in self.nodes:
                continue

            node = {
                "poi_id": poi_id,
                "name": row.get("name", ""),
                "fclass": row.get("fclass", "unknown"),
                "province": row.get("province", ""),
                "latitude": row.get("lat", 0.0),
                "longitude": row.get("lon", 0.0),
            }

            if "population" in row:
                node["population"] = row["population"]

            self.nodes[poi_id] = node
            count += 1

        logger.info(f"Loaded {count} POIs from {poi_csv}")
        return count

    def add_geographic_edges(
        self,
        radius_km: float = 1.0,
        max_neighbors: int = 10,
    ) -> int:
        """Add edges based on geographic proximity."""
        edges_added = 0
        pois = list(self.nodes.values())

        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError:
            logger.warning("scikit-learn not available, skipping geographic edges")
            return 0

        coords = []
        poi_ids = []
        for poi in pois:
            lat = poi.get("latitude", 0.0)
            lon = poi.get("longitude", 0.0)
            if lat != 0 and lon != 0:
                coords.append([lat, lon])
                poi_ids.append(poi["poi_id"])

        if not coords:
            logger.warning("No valid coordinates for geographic edges")
            return 0

        coords = np.array(coords)
        radius_deg = radius_km / 111.0

        nbrs = NearestNeighbors(
            algorithm="ball_tree",
            metric="haversine",
            radius=radius_deg,
        )
        nbrs.fit(np.radians(coords))

        distances, indices = nbrs.radius_neighbors(np.radians(coords))

        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            poi_a = poi_ids[i]

            sorted_neighbors = sorted(
                zip(dists, idxs),
                key=lambda x: x[0]
            )[:max_neighbors]

            for dist_rad, j in sorted_neighbors:
                if j == i:
                    continue
                poi_b = poi_ids[j]
                dist_km = dist_rad * 6371.0

                if poi_b in self.nodes:
                    self.edges["geographic"].append((
                        poi_a, poi_b,
                        {"weight": 1.0 / (1.0 + dist_km)}
                    ))
                    edges_added += 1

        logger.info(f"Added {edges_added} geographic edges")
        return edges_added

    def add_category_edges(
        self,
        similarity_threshold: float = 0.8,
    ) -> int:
        """Add edges between POIs with similar categories."""
        category_pois: Dict[str, Set[str]] = defaultdict(set)
        for poi_id, poi in self.nodes.items():
            fclass = poi.get("fclass", "unknown")
            category_pois[fclass].add(poi_id)

        categories = list(category_pois.keys())
        edges_added = 0

        for i, cat_a in enumerate(categories):
            for cat_b in categories[i + 1:]:
                set_a = category_pois[cat_a]
                set_b = category_pois[cat_b]

                intersection = len(set_a & set_b)
                union = len(set_a | set_b)

                if union > 0:
                    similarity = intersection / union
                    if similarity >= similarity_threshold:
                        for poi_a in set_a:
                            for poi_b in set_b:
                                self.edges["category"].append((
                                    poi_a, poi_b,
                                    {"weight": similarity}
                                ))
                                edges_added += 1

        logger.info(f"Added {edges_added} category similarity edges")
        return edges_added

    def build_feature_maps(self) -> None:
        """Build feature maps for categorical variables."""
        categories = set()
        provinces = set()

        for poi in self.nodes.values():
            categories.add(poi.get("fclass", "unknown"))
            provinces.add(poi.get("province", "unknown"))

        self.category_map = {cat: i for i, cat in enumerate(sorted(categories))}
        self.province_map = {prov: i for i, prov in enumerate(sorted(provinces))}

        logger.info(f"Feature maps: {len(self.category_map)} categories, "
                   f"{len(self.province_map)} provinces")

    def compute_node_features(self) -> np.ndarray:
        """Compute feature matrix for all nodes."""
        if not self.nodes:
            raise ValueError("No nodes in graph")

        self.build_feature_maps()

        lats = [poi.get("latitude", 0.0) for poi in self.nodes.values()]
        lons = [poi.get("longitude", 0.0) for poi in self.nodes.values()]
        pops = [poi.get("population", 0) for poi in self.nodes.values()]

        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        pop_max = max(pops) if pops else 1

        features = []
        for poi in self.nodes.values():
            feat = []

            cat_idx = self.category_map.get(poi.get("fclass", "unknown"), 0)
            prov_idx = self.province_map.get(poi.get("province", "unknown"), 0)
            feat.extend([cat_idx, prov_idx])

            lat = poi.get("latitude", 0.0)
            lon = poi.get("longitude", 0.0)

            if lat_max > lat_min:
                lat_norm = (lat - lat_min) / (lat_max - lat_min)
            else:
                lat_norm = 0.5

            if lon_max > lon_min:
                lon_norm = (lon - lon_min) / (lon_max - lon_min)
            else:
                lon_norm = 0.5

            feat.extend([lat_norm, lon_norm])

            pop = poi.get("population", 0)
            if pop_max > 0:
                pop_norm = pop / pop_max
            else:
                pop_norm = 0.0

            feat.append(pop_norm)
            features.append(feat)

        return np.array(features, dtype=np.float32)

    def build_graph(self, edge_types: List[str] = None) -> Optional[Any]:
        """Build PyTorch Geometric graph data structure."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            logger.warning("PyTorch Geometric not available, returning None")
            return None

        if edge_types is None:
            edge_types = list(self.edges.keys())

        self.node_id_map = {pid: i for i, pid in enumerate(self.nodes.keys())}
        self.id_node_map = {i: pid for pid, i in self.node_id_map.items()}

        x = self.compute_node_features()
        x_tensor = torch.from_numpy(x).float()

        all_edges = []
        edge_weights = []

        for edge_type in edge_types:
            for src, dst, attr in self.edges.get(edge_type, []):
                if src in self.node_id_map and dst in self.node_id_map:
                    src_idx = self.node_id_map[src]
                    dst_idx = self.node_id_map[dst]
                    all_edges.append([src_idx, dst_idx])
                    edge_weights.append(attr.get("weight", 1.0))

        if not all_edges:
            logger.warning("No edges in graph")
            return None

        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        graph = Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.nodes),
        )

        logger.info(f"Built graph with {graph.num_nodes} nodes and "
                   f"{graph.num_edges} edges")
        logger.info(f"  Feature dim: {x.shape[1]}")

        return graph

    def save_graph(self, path: Optional[Union[str, Path]] = None) -> str:
        """Save graph data to disk."""
        if path is None:
            path = self.cache_dir / "graph.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "nodes": self.nodes,
            "edges": self.edges,
            "node_id_map": self.node_id_map,
            "id_node_map": self.id_node_map,
            "category_map": self.category_map,
            "province_map": self.province_map,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Graph saved to {path}")
        return str(path)

    def load_graph(self, path: Optional[Union[str, Path]] = None) -> None:
        """Load graph data from disk."""
        if path is None:
            path = self.cache_dir / "graph.pkl"
        else:
            path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.nodes = data["nodes"]
        self.edges = data["edges"]
        self.node_id_map = data.get("node_id_map", {})
        self.id_node_map = data.get("id_node_map", {})
        self.category_map = data.get("category_map", {})
        self.province_map = data.get("province_map", {})

        logger.info(f"Graph loaded from {path}")
        logger.info(f"  Nodes: {len(self.nodes)}")

        for edge_type, edges in self.edges.items():
            logger.info(f"  {edge_type} edges: {len(edges)}")


# ============================================================================
# POI GNN Recommender
# ============================================================================

class POIGNNRecommender:
    """
    GNN-based POI recommender.

    Features:
    - POI embedding extraction from trained GNN
    - Similar POI query via embedding similarity
    - Graph walk based recommendation
    - Integration with GoAfar recommendation pipeline
    """

    def __init__(
        self,
        builder: Optional[POIGraphBuilder] = None,
        config: Optional[GNNConfig] = None,
    ):
        """Initialize GNN recommender."""
        self.config = config or GNNConfig()
        self.builder = builder or POIGraphBuilder(self.config)

        # Node embeddings
        self.embeddings: Optional[np.ndarray] = None
        self.poi_id_map: Dict[str, int] = {}
        self.id_poi_map: Dict[int, str] = {}

        logger.info("POIGNNRecommender initialized")

    def get_poi_embedding(self, poi_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific POI."""
        if self.embeddings is None:
            logger.warning("No embeddings available")
            return None

        idx = self.poi_id_map.get(str(poi_id))
        if idx is None:
            return None

        return self.embeddings[idx]

    def get_similar_pois(
        self,
        poi_id: str,
        top_k: int = 10,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Find similar POIs by embedding similarity."""
        if self.embeddings is None:
            logger.warning("No embeddings available")
            return []

        query_idx = self.poi_id_map.get(str(poi_id))
        if query_idx is None:
            logger.warning(f"POI {poi_id} not found")
            return []

        query_emb = self.embeddings[query_idx]

        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_emb)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        similarities = similarities / (norms + 1e-8)

        # Get top-k
        exclude_set = exclude_ids or {poi_id}
        candidates = [
            (idx, sim)
            for idx, sim in enumerate(similarities)
            if idx != query_idx and self.id_poi_map[idx] not in exclude_set
        ]

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        return [(self.id_poi_map[idx], sim) for idx, sim in top_candidates]

    def graph_walk_recommend(
        self,
        start_poi: str,
        num_steps: int = 3,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Generate recommendations via graph walk."""
        if self.embeddings is None:
            return []

        current = str(start_poi)
        visited = {current}
        scores = defaultdict(float)

        for step in range(num_steps):
            similar = self.get_similar_pois(current, top_k * 2, visited)

            for poi_id, sim in similar:
                decay = 0.8 ** step
                scores[poi_id] += sim * decay

            for poi_id, _ in similar:
                if poi_id not in visited:
                    current = poi_id
                    visited.add(current)
                    break
            else:
                break

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def predict(
        self,
        user_id: str,
        candidate_pois: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Predict scores for candidate POIs."""
        context = context or {}
        visited = set(context.get("visited_pois", []))

        results = []

        for poi in candidate_pois:
            poi_id = poi.get("poi_id", "")

            max_sim = 0.0
            for visited_id in visited:
                sim_pois = self.get_similar_pois(visited_id, top_k=50)
                for sid, sim in sim_pois:
                    if sid == poi_id:
                        max_sim = max(max_sim, sim)

            pop_score = poi.get("popularity", 0) / 100.0
            combined = 0.7 * max_sim + 0.3 * min(pop_score, 1.0)

            results.append((poi_id, {"gnn_score": combined}))

        results.sort(key=lambda x: x[1]["gnn_score"], reverse=True)
        return results

    def save_model(self, path: Optional[str] = None) -> str:
        """Save model and embeddings."""
        if path is None:
            path = Path(self.config.graph_dir) / "model.pkl"
        else:
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config,
            "embeddings": self.embeddings,
            "poi_id_map": self.poi_id_map,
            "id_poi_map": self.id_poi_map,
            "builder_nodes": self.builder.nodes,
            "builder_edges": self.builder.edges,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {path}")
        return str(path)

    def load_model(self, path: str) -> None:
        """Load model and embeddings."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.config = data["config"]
        self.embeddings = data["embeddings"]
        self.poi_id_map = data["poi_id_map"]
        self.id_poi_map = data["id_poi_map"]

        # Restore builder
        self.builder.nodes = data["builder_nodes"]
        self.builder.edges = data["builder_edges"]

        logger.info(f"Model loaded from {path}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_gnn_model(
    model_type: str = "graphsage",
    hidden_dim: int = 128,
    output_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    device: str = "cpu",
) -> POIGNNRecommender:
    """Create a GNN recommender with default configuration."""
    config = GNNConfig(
        model_type=model_type,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )
    return POIGNNRecommender(config=config)


def create_graph_builder(
    cache_dir: str = "outputs/gnn/cache",
) -> POIGraphBuilder:
    """Create a POI graph builder."""
    return POIGraphBuilder(cache_dir=cache_dir)
