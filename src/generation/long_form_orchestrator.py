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
from .chapter_budget import SegmentBudget, allocate_segment_budgets, allocate_segment_budgets_uniform
from .chapter_context import ChapterContext
from .literary_generator import LiteraryGenerator, GenerationResult
from .memoir_segmenter import (
    MemoirSegment,
    SegmentationReport,
    segment_memoir,
    validate_segmentation,
)

logger = logging.getLogger(__name__)


def _extract_entity_names(rr: RetrievalResult, limit: int = 10) -> List[str]:
    """从检索结果中提取实体名称（避免重复代码）"""
    names = [e.get("name", "") for e in (rr.entities or [])[:limit] if e.get("name")]
    return names


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
    chapter_gen_min_chars: Optional[int] = None,
    chapter_gen_max_chars: Optional[int] = None,
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

    if (
        chapter_gen_min_chars is not None
        and chapter_gen_max_chars is not None
    ):
        cg_lo = max(50, int(chapter_gen_min_chars))
        cg_hi = max(cg_lo + 1, int(chapter_gen_max_chars))
        budgets = allocate_segment_budgets_uniform(segments, cg_lo, cg_hi)
    else:
        budgets = allocate_segment_budgets(segments, length_bucket)

    chapter_ctx = ChapterContext(total_chapters=len(segments)) if enable_cross_chapter_context else None

    import asyncio

    chapters: List[ChapterGenerationResult] = []
    parts: List[str] = []

    # 预取第一章的检索结果
    next_retrieval: Optional[asyncio.Task] = asyncio.create_task(
        retriever.retrieve(
            segments[0].text, top_k=top_k,
            use_llm_parsing=use_llm_parsing, mode=retrieval_mode,
        )
    )

    for i, (seg, budget) in enumerate(zip(segments, budgets)):
        # 1) 获取当前章的检索结果（已经在上一轮预取）
        rr = await next_retrieval

        # 预取下一章的检索（与当前章的生成并行）
        if i + 1 < len(segments):
            next_retrieval = asyncio.create_task(
                retriever.retrieve(
                    segments[i + 1].text, top_k=top_k,
                    use_llm_parsing=use_llm_parsing, mode=retrieval_mode,
                )
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
            entities = _extract_entity_names(rr)
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


async def regenerate_chapters(
    result: LongFormGenerationResult,
    retriever: MemoirRetriever,
    generator: LiteraryGenerator,
    chapters_to_regenerate: List[int],
    prompt_adjustments: Optional[dict[int, str]] = None,
    *,
    style: str = "standard",
    temperature: float = 0.7,
    retrieval_mode: str = "keyword",
    use_llm_parsing: bool = False,
    top_k: int = 10,
    chapter_separator: str = "\n\n---\n\n",
) -> LongFormGenerationResult:
    """
    对指定章节重新生成，返回更新后的 LongFormGenerationResult。

    利用 RemediationPlan 中的 prompt_adjustments 加强对应章节的 prompt，
    其余章节保持不变。
    """
    adjustments = prompt_adjustments or {}

    for ch_idx in chapters_to_regenerate:
        if ch_idx < 0 or ch_idx >= len(result.chapters):
            logger.warning("[Regenerate] 章节索引 %d 超出范围，跳过", ch_idx)
            continue

        old_ch = result.chapters[ch_idx]
        seg_text = old_ch.segment_text

        rr = await retriever.retrieve(
            seg_text, top_k=top_k,
            use_llm_parsing=use_llm_parsing, mode=retrieval_mode,
        )

        cross_ctx = ""
        if result.chapter_context is not None:
            cross_ctx = result.chapter_context.build_prompt_section(ch_idx)

        adj = adjustments.get(ch_idx, "")
        if adj:
            cross_ctx += f"\n\n【修复指令】{adj}"

        gr = await generator.generate(
            memoir_text=seg_text,
            retrieval_result=rr,
            style=style,
            length_hint=old_ch.length_hint,
            temperature=min(temperature + 0.05, 1.0),
            max_tokens=old_ch.generation.max_tokens if hasattr(old_ch.generation, "max_tokens") else 1024,
            chapter_context=cross_ctx,
        )

        if result.chapter_context is not None:
            entities = _extract_entity_names(rr)
            result.chapter_context.record_chapter(ch_idx, gr.content, entities)

        result.chapters[ch_idx] = ChapterGenerationResult(
            segment_index=old_ch.segment_index,
            segment_text=seg_text,
            retrieval_result=rr,
            generation=gr,
            length_hint=old_ch.length_hint,
            repetition_warning=f"[质量门控重试]",
        )
        logger.info("[Regenerate] 第%d章已重新生成 (%d字)", ch_idx + 1, len(gr.content))

    parts = [ch.generation.content.strip() for ch in result.chapters]
    result.merged_content = chapter_separator.join(p for p in parts if p)

    return result


def format_chapter_progress(prefix: str, chapter_idx: int, total: int) -> str:
    return f"{prefix}（第 {chapter_idx + 1}/{total} 章）…"
