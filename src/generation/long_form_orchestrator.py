"""
长文分章编排：按段检索 + 生成，合并输出；不修改 LiteraryGenerator 核心逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..retrieval import MemoirRetriever, RetrievalResult
from .chapter_budget import SegmentBudget, allocate_segment_budgets
from .literary_generator import LiteraryGenerator, GenerationResult
from .memoir_segmenter import MemoirSegment, segment_memoir


@dataclass
class ChapterGenerationResult:
    segment_index: int
    segment_text: str
    retrieval_result: RetrievalResult
    generation: GenerationResult
    length_hint: str = ""


@dataclass
class LongFormGenerationResult:
    """分章生成完整结果。"""

    chapters: List[ChapterGenerationResult]
    merged_content: str
    full_memoir_text: str
    segments: List[MemoirSegment] = field(default_factory=list)


async def run_long_form_generation(
    memoir_text: str,
    retriever: MemoirRetriever,
    generator: LiteraryGenerator,
    *,
    length_bucket: str = "400-800",
    style: str = "standard",
    temperature: float = 0.7,
    retrieval_mode: str = "keyword",
    use_llm_parsing: bool = False,
    top_k: int = 10,
    chapter_separator: str = "\n\n---\n\n",
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> LongFormGenerationResult:
    """
    分段检索与生成，各章仅携带本段回忆录正文（不注入全篇）。
    """
    text = (memoir_text or "").strip()
    segments = segment_memoir(text, target_min_chars=target_min_chars, target_max_chars=target_max_chars)
    if not segments:
        return LongFormGenerationResult(
            chapters=[],
            merged_content="",
            full_memoir_text=text,
            segments=[],
        )

    budgets: List[SegmentBudget] = allocate_segment_budgets(segments, length_bucket)
    chapters: List[ChapterGenerationResult] = []
    parts: List[str] = []

    for seg, budget in zip(segments, budgets):
        rr = await retriever.retrieve(
            seg.text,
            top_k=top_k,
            use_llm_parsing=use_llm_parsing,
            mode=retrieval_mode,
        )
        gr = await generator.generate(
            memoir_text=seg.text,
            retrieval_result=rr,
            style=style,
            length_hint=budget.length_hint,
            temperature=temperature,
            max_tokens=budget.max_tokens,
        )
        chapters.append(
            ChapterGenerationResult(
                segment_index=seg.index,
                segment_text=seg.text,
                retrieval_result=rr,
                generation=gr,
                length_hint=budget.length_hint,
            )
        )
        parts.append(gr.content.strip())

    merged = chapter_separator.join(p for p in parts if p)
    return LongFormGenerationResult(
        chapters=chapters,
        merged_content=merged,
        full_memoir_text=text,
        segments=segments,
    )


def format_chapter_progress(prefix: str, chapter_idx: int, total: int) -> str:
    return f"{prefix}（第 {chapter_idx + 1}/{total} 章）…"
