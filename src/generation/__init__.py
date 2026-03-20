"""文本生成模块"""
from .literary_generator import LiteraryGenerator, GenerationResult, MultiGenerationResult
from .prompts import PromptTemplates, SYSTEM_PROMPTS

__all__ = [
    "LiteraryGenerator",
    "GenerationResult",
    "MultiGenerationResult",
    "PromptTemplates",
    "SYSTEM_PROMPTS",
]
