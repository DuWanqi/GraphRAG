"""文本生成模块"""
from .literary_generator import LiteraryGenerator, GenerationResult, MultiGenerationResult
from .prompts import PromptTemplates, get_system_prompt

# 兼容旧代码：之前外部可能引用 SYSTEM_PROMPTS
# 现在统一通过 prompts.json 读取，保留同名变量作为只读映射。
SYSTEM_PROMPTS = {
    "default": get_system_prompt("default"),
    "historian": get_system_prompt("historian"),
    "novelist": get_system_prompt("novelist"),
}

__all__ = [
    "LiteraryGenerator",
    "GenerationResult",
    "MultiGenerationResult",
    "PromptTemplates",
    "SYSTEM_PROMPTS",
    "get_system_prompt",
]
