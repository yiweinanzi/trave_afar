"""
内容生成模块 - 文案生成
支持模板生成和LLM生成两种模式
"""
from .title_generator import generate_title, generate_description
from .llm_generator import LLMGenerator

__all__ = ['generate_title', 'generate_description', 'LLMGenerator']

