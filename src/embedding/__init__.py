"""
语义嵌入模块 - BGE-M3
"""
from .bge_m3_encoder import BGEM3Encoder
from .vector_builder import build_poi_embeddings, search_similar_pois

__all__ = ['BGEM3Encoder', 'build_poi_embeddings', 'search_similar_pois']

