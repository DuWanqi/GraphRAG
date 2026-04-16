"""文本生成模块（重模块延迟加载，便于无 LLM 环境单测分段/预算）。"""

from __future__ import annotations

from typing import Any

from .memoir_segmenter import (
    MemoirSegment,
    SegmentMeta,
    SegmentationReport,
    SegmentationIssue,
    segment_memoir,
    validate_segmentation,
    extract_years,
)
from .chapter_budget import SegmentBudget, allocate_segment_budgets, legacy_maps_for_single_segment
from .chapter_context import ChapterContext, ChapterPosition, ChapterRecord
from .prompts import PromptTemplates, get_system_prompt
from .runtime_options import (
    single_segment_generation_config,
    estimate_long_form_generation_timeout,
    estimate_long_form_evaluation_timeout,
    build_long_form_eval_options,
)

__all__ = [
    "LiteraryGenerator",
    "GenerationResult",
    "MultiGenerationResult",
    "MemoirSegment",
    "SegmentMeta",
    "SegmentationReport",
    "SegmentationIssue",
    "segment_memoir",
    "validate_segmentation",
    "extract_years",
    "SegmentBudget",
    "allocate_segment_budgets",
    "legacy_maps_for_single_segment",
    "ChapterContext",
    "ChapterPosition",
    "ChapterRecord",
    "ChapterGenerationResult",
    "LongFormGenerationResult",
    "run_long_form_generation",
    "format_chapter_progress",
    "PromptTemplates",
    "SYSTEM_PROMPTS",
    "get_system_prompt",
    "single_segment_generation_config",
    "estimate_long_form_generation_timeout",
    "estimate_long_form_evaluation_timeout",
    "build_long_form_eval_options",
]


def __getattr__(name: str) -> Any:
    if name in ("LiteraryGenerator", "GenerationResult", "MultiGenerationResult"):
        from .literary_generator import LiteraryGenerator, GenerationResult, MultiGenerationResult

        return {
            "LiteraryGenerator": LiteraryGenerator,
            "GenerationResult": GenerationResult,
            "MultiGenerationResult": MultiGenerationResult,
        }[name]
    if name in (
        "ChapterGenerationResult",
        "LongFormGenerationResult",
        "run_long_form_generation",
        "format_chapter_progress",
    ):
        from . import long_form_orchestrator as lfo

        return getattr(lfo, name)
    if name == "SYSTEM_PROMPTS":
        return {
            "default": get_system_prompt("default"),
            "historian": get_system_prompt("historian"),
            "novelist": get_system_prompt("novelist"),
        }
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
