"""
语义嵌入模块 - BGE-M3, Qwen3-Embedding
"""
# Optional import for BGE-M3 (may not be available due to version conflicts)
try:
    from .bge_m3_encoder import BGEM3Encoder
    _bge_available = True
except (ImportError, Exception) as e:
    BGEM3Encoder = None
    _bge_available = False

from .qwen3_encoder import Qwen3Embedding
from .vector_builder import build_poi_embeddings, search_similar_pois

__all__ = ['BGEM3Encoder', 'Qwen3Embedding', 'build_poi_embeddings', 'search_similar_pois']

