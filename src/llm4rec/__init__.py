"""
LLM4Rec - 大模型增强推荐模块

核心功能:
1. Intent Understanding - 意图理解
2. Item Encoding - POI特征增强  
3. LLM Reranking - 候选重排序
4. Explanation Generation - 推荐解释生成
5. Qwen Recommender - 基于Qwen3的推荐器
"""

from .intent_understanding import IntentUnderstandingModule
from .item_encoding import POIEncodingModule
from .llm_reranker import LLMReranker
from .explanation import ExplanationGenerator
from .qwen_recommender import QwenRecommender

__all__ = [
    'IntentUnderstandingModule',
    'POIEncodingModule', 
    'LLMReranker',
    'ExplanationGenerator',
    'QwenRecommender'
]

