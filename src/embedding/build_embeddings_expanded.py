"""
GPU-accelerated POI embedding builder for expanded dataset.

Supports:
- 127,978 POIs from data/all/poi_expanded.csv
- Incremental building with checkpoint resumption
- Progress tracking and ETA estimation
- FAISS index for fast retrieval
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embedding.bge_m3_encoder import BGEM3Encoder
from src.utils.id_mapping import normalize_poi_id

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding build process."""

    # Data paths
    poi_csv: str = "data/all/poi_expanded.csv"
    output_dir: str = "outputs/emb"

    # Model
    model_path: str = "models/Qwen3-Embedding-4B"
    fallback_model_path: str = "models/Xorbits/bge-m3"

    # Processing
    batch_size: int = 256  # GPU can handle larger batches
    max_length: int = 512
    num_workers: int = 4

    # Checkpointing
    checkpoint_interval: int = 10000  # Save every N POIs
    checkpoint_dir: str = "outputs/emb/checkpoints"

    # Output
    save_faiss: bool = True
    faiss_index_type: str = "HNSW"  # HNSW or IVF or Flat


class CheckpointManager:
    """Manage incremental embedding build with checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progress.json"

    def load_progress(self) -> Tuple[int, np.ndarray, pd.DataFrame]:
        """Load progress from checkpoint."""
        if not self.checkpoint_file.exists():
            return 0, None, None

        with open(self.checkpoint_file, "r") as f:
            data = json.load(f)

        start_idx = data.get("processed_count", 0)
        embeddings_path = self.checkpoint_dir / "embeddings.npy"
        metadata_path = self.checkpoint_dir / "metadata.csv"

        embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
        metadata = pd.read_csv(metadata_path) if metadata_path.exists() else None

        logger.info(f"Resuming from checkpoint: {start_idx} POIs processed")
        return start_idx, embeddings, metadata

    def save_progress(
        self,
        processed_count: int,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        """Save current progress."""
        # Save progress info
        with open(self.checkpoint_file, "w") as f:
            json.dump({"processed_count": processed_count}, f)

        # Save embeddings
        np.save(self.checkpoint_dir / "embeddings.npy", embeddings)

        # Save metadata
        metadata.to_csv(self.checkpoint_dir / "metadata.csv", index=False)

        logger.debug(f"Checkpoint saved: {processed_count} POIs")


class ExpandedEmbeddingBuilder:
    """
    Build embeddings for expanded POI dataset (127K+ items).

    Features:
    - Incremental building with checkpoint resume
    - ETA estimation
    - FAISS index creation
    - Multi-GPU support (future)
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.encoder = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_encoder(self) -> BGEM3Encoder:
        """Initialize encoder with fallback."""
        model_path = Path(self.config.model_path)

        if not model_path.exists() and self.config.fallback_model_path:
            logger.warning(f"{self.config.model_path} not found, trying fallback")
            model_path = Path(self.config.fallback_model_path)

        logger.info(f"Loading encoder from {model_path} on {self.device}")
        return BGEM3Encoder(model_path=str(model_path), use_gpu=(self.device == "cuda"))

    def _build_texts(self, df: pd.DataFrame) -> List[str]:
        """Build text representations from POI data."""
        texts = []

        for _, row in df.iterrows():
            parts = [str(row["name"])]

            # Add location
            if pd.notna(row.get("province")):
                parts.append(str(row["province"]))
            if pd.notna(row.get("city")) and row.get("city") != row.get("province"):
                parts.append(str(row["city"]))

            # Add description (truncated)
            if pd.notna(row.get("description")) and row["description"]:
                desc = str(row["description"]).replace("\n", " ")[:200]
                parts.append(desc)

            # Add stay time
            if pd.notna(row.get("stay_min")):
                stay_hours = row["stay_min"] / 60
                parts.append(f"建议停留{stay_hours:.1f}小时")

            texts.append(" ".join(parts))

        return texts

    def _build_faiss_index(
        self,
        embeddings: np.ndarray,
    ) -> Any:
        """Build FAISS index for fast retrieval."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, skipping index build")
            return None

        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index (dim={dimension}, type={self.config.faiss_index_type})")

        if self.config.faiss_index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
            index.hnsw.efConstruction = 200
        elif self.config.faiss_index_type == "IVF":
            nlist = min(1000, len(embeddings) // 10)
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:  # Flat
            index = faiss.IndexFlatL2(dimension)

        index.train(embeddings)
        index.add(embeddings)

        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index

    def build(self) -> Tuple[np.ndarray, pd.DataFrame, Any]:
        """Build embeddings for all POIs."""
        # Load data
        logger.info(f"Loading POI data from {self.config.poi_csv}")
        df = pd.read_csv(self.config.poi_csv)
        total_count = len(df)
        logger.info(f"Total POIs to process: {total_count:,}")

        # Normalize IDs
        if "poi_id" in df.columns:
            df["poi_id"] = df["poi_id"].apply(normalize_poi_id)

        # Load checkpoint
        start_idx, existing_embeddings, existing_metadata = self.checkpoint_manager.load_progress()

        if existing_embeddings is not None:
            embeddings = existing_embeddings
            metadata = existing_metadata
        else:
            dimension = 1024  # Adjust based on model
            embeddings = np.zeros((total_count, dimension), dtype=np.float32)
            metadata = df.iloc[0:0].copy()

        # Initialize encoder
        self.encoder = self._load_encoder()

        # Process in batches
        batch_size = self.config.batch_size
        end_idx = total_count

        logger.info(f"Starting from POI {start_idx:,} of {total_count:,}")

        with tqdm(total=total_count, initial=start_idx, desc="Building embeddings") as pbar:
            for batch_start in range(start_idx, end_idx, batch_size):
                batch_end = min(batch_start + batch_size, end_idx)
                batch_df = df.iloc[batch_start:batch_end]

                # Build texts
                texts = self._build_texts(batch_df)

                # Encode
                batch_embeddings = self.encoder.encode_texts(
                    texts,
                    batch_size=len(texts),
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_v2=False,
                )["dense_vecs"]

                # Store
                embeddings[batch_start:batch_end] = batch_embeddings
                metadata = pd.concat([metadata, batch_df], ignore_index=True)

                # Update progress
                pbar.update(len(texts))

                # Checkpoint
                processed_count = batch_end
                if processed_count % self.config.checkpoint_interval == 0:
                    self.checkpoint_manager.save_progress(
                        processed_count,
                        embeddings[:processed_count],
                        metadata,
                    )

        # Build FAISS index
        faiss_index = None
        if self.config.save_faiss:
            faiss_index = self._build_faiss_index(embeddings)

        # Save final outputs
        self._save_outputs(embeddings, metadata, faiss_index)

        return embeddings, metadata, faiss_index

    def _save_outputs(
        self,
        embeddings: np.ndarray,
        metadata: pd.DataFrame,
        faiss_index: Any,
    ) -> None:
        """Save final outputs."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        emb_path = output_dir / "poi_expanded_emb.npy"
        np.save(emb_path, embeddings)
        logger.info(f"Saved embeddings to {emb_path}")

        # Save metadata
        meta_path = output_dir / "poi_expanded_meta.csv"
        metadata.to_csv(meta_path, index=False)
        logger.info(f"Saved metadata to {meta_path}")

        # Save FAISS index
        if faiss_index is not None:
            index_path = output_dir / "poi_expanded_faiss.index"
            try:
                import faiss
                faiss.write_index(faiss_index, str(index_path))
                logger.info(f"Saved FAISS index to {index_path}")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

        # Save config
        config_path = output_dir / "build_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "total_pois": len(metadata),
                "embedding_dim": embeddings.shape[1],
                "model_path": self.config.model_path,
                "faiss_index_type": self.config.faiss_index_type,
            }, f, indent=2)

        logger.info("Build outputs saved successfully")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build embeddings for expanded POI dataset (127K+ items)"
    )
    parser.add_argument(
        "--poi-csv",
        default="data/all/poi_expanded.csv",
        help="Path to expanded POI CSV file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/emb",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--model",
        default="models/Qwen3-Embedding-4B",
        help="Embedding model path"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore existing checkpoints"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Save checkpoint every N POIs"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config
    config = EmbeddingConfig(
        poi_csv=args.poi_csv,
        output_dir=args.output_dir,
        model_path=args.model,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Clear checkpoints if requested
    if args.no_cache:
        import shutil
        checkpoint_dir = Path(config.checkpoint_dir)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info("Cleared existing checkpoints")

    # Build
    builder = ExpandedEmbeddingBuilder(config)
    embeddings, metadata, faiss_index = builder.build()

    logger.info("=" * 60)
    logger.info("Build Summary:")
    logger.info(f"  POIs processed: {len(metadata):,}")
    logger.info(f"  Embedding shape: {embeddings.shape}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
