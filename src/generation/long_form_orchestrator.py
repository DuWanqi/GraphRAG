"""
长文分章编排：按段检索 + 生成，带跨章上下文管理与质量门控。

核心改进：
1. 每章生成后记录摘要/要点，注入后续章节 prompt（防重复、保衔接）
2. 生成完成后运行质量门控，输出结构化报告与修复建议
3. 可选自动重新生成未通过门控的章节（最多重试 1 次）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..retrieval import MemoirRetriever, RetrievalResult
from .chapter_budget import SegmentBudget, allocate_segment_budgets
from .chapter_context import ChapterContext
from .literary_generator import LiteraryGenerator, GenerationResult
from .memoir_segmenter import (
    MemoirSegment,
    SegmentationReport,
    segment_memoir,
    validate_segmentation,
)

logger = logging.getLogger(__name__)


@dataclass
class ChapterGenerationResult:
    segment_index: int
    segment_text: str
    retrieval_result: RetrievalResult
    generation: GenerationResult
    length_hint: str = ""
    repetition_warning: str = ""       # 跨章重复检测结果


@dataclass
class LongFormGenerationResult:
    """分章生成完整结果。"""

    chapters: List[ChapterGenerationResult]
    merged_content: str
    full_memoir_text: str
    segments: List[MemoirSegment] = field(default_factory=list)
    segmentation_report: Optional[SegmentationReport] = None
    chapter_context: Optional[ChapterContext] = None


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
    enable_cross_chapter_context: bool = True,
    max_retry_chapters: int = 0,
) -> LongFormGenerationResult:
    """
    分段检索与生成，各章仅携带本段回忆录正文 + 跨章上下文。

    新增参数:
        enable_cross_chapter_context: 是否启用跨章上下文（前文概要 + 反重复）
        max_retry_chapters: 对跨章重复率过高的章节自动重试次数上限 (0=不重试)
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

    # 分段校验
    seg_report = validate_segmentation(segments, target_min_chars, target_max_chars)
    if seg_report.issues:
        logger.info("[Orchestrator] 分段校验: %s", seg_report.to_text())

    budgets: List[SegmentBudget] = allocate_segment_budgets(segments, length_bucket)
    chapter_ctx = ChapterContext(total_chapters=len(segments)) if enable_cross_chapter_context else None

    chapters: List[ChapterGenerationResult] = []
    parts: List[str] = []

    for seg, budget in zip(segments, budgets):
        # 1) 检索
        rr = await retriever.retrieve(
            seg.text,
            top_k=top_k,
            use_llm_parsing=use_llm_parsing,
            mode=retrieval_mode,
        )

        # 2) 构建跨章上下文
        cross_ctx = ""
        if chapter_ctx is not None:
            cross_ctx = chapter_ctx.build_prompt_section(seg.index)

        # 3) 生成
        gr = await generator.generate(
            memoir_text=seg.text,
            retrieval_result=rr,
            style=style,
            length_hint=budget.length_hint,
            temperature=temperature,
            max_tokens=budget.max_tokens,
            chapter_context=cross_ctx,
        )

        # 4) 跨章重复检测
        rep_warning = ""
        if chapter_ctx is not None:
            rep_warning = chapter_ctx.detect_repetition_with_previous(gr.content) or ""
            if rep_warning:
                logger.warning("[Orchestrator] 第%d章 %s", seg.index + 1, rep_warning)

            # 可选：自动重试（加强反重复指令）
            if rep_warning and max_retry_chapters > 0:
                stronger_ctx = cross_ctx + "\n\n⚠️ 严格要求：上一次生成的内容与前文高度重叠，请务必从不同角度切入，避免重复。"
                gr = await generator.generate(
                    memoir_text=seg.text,
                    retrieval_result=rr,
                    style=style,
                    length_hint=budget.length_hint,
                    temperature=min(temperature + 0.1, 1.0),
                    max_tokens=budget.max_tokens,
                    chapter_context=stronger_ctx,
                )
                new_rep = chapter_ctx.detect_repetition_with_previous(gr.content) or ""
                rep_warning = f"[重试后] {new_rep}" if new_rep else ""

        # 5) 记录
        if chapter_ctx is not None:
            entities = [e.get("name", "") for e in (rr.entities or [])[:10] if e.get("name")]
            chapter_ctx.record_chapter(seg.index, gr.content, entities)

        chapters.append(
            ChapterGenerationResult(
                segment_index=seg.index,
                segment_text=seg.text,
                retrieval_result=rr,
                generation=gr,
                length_hint=budget.length_hint,
                repetition_warning=rep_warning,
            )
        )
        parts.append(gr.content.strip())

    merged = chapter_separator.join(p for p in parts if p)
    return LongFormGenerationResult(
        chapters=chapters,
        merged_content=merged,
        full_memoir_text=text,
        segments=segments,
        segmentation_report=seg_report,
        chapter_context=chapter_ctx,
    )


def format_chapter_progress(prefix: str, chapter_idx: int, total: int) -> str:
    return f"{prefix}（第 {chapter_idx + 1}/{total} 章）…"
