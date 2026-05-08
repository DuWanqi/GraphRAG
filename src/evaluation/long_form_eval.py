"""
长文分章结果的评估聚合：
- 段级指标（准确性、相关性、文学性）
- 段级事实检查（可选，带超时保护）
- 篇级跨章指标（重复度、风格一致性、总结句比率）
- 质量门控 + 可执行修复建议
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import (
    MetricResult,
    calculate_all_metrics,
    aggregate_scores,
    LiteraryMetrics,
    CrossChapterMetrics,
)
from ..evaluation.quality_gate import (
    QualityGateResult,
    QualityThresholds,
    check_quality_gate,
)


def _parse_length_hint_range(length_hint: str) -> tuple[int, int]:
    """从「200-400字」类字符串解析最佳长度区间。"""
    m = re.search(r"(\d+)\s*[-–~至]\s*(\d+)", length_hint)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m2 = re.search(r"(\d+)", length_hint)
    if m2:
        x = int(m2.group(1))
        return (max(50, x - 50), x + 200)
    return (200, 500)


def _extract_entities_from_text(text: str) -> List[str]:
    """从文本中提取实体（简单规则：中文人名、地名等）"""
    entities = []

    # 提取人名模式
    # 1. 老X（老张、老王等）- 修复正则，避免匹配"老张是个"
    old_pattern = r'老[张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚]'
    matches = re.findall(old_pattern, text)
    entities.extend(matches)

    # 2. X队长、X先生、X老板等（带职位）
    title_pattern = r'[张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚]\w{0,2}(?:队长|先生|老板|师傅|医生|校长)'
    matches = re.findall(title_pattern, text)
    entities.extend(matches)

    # 3. 常见双字名（赵德明、张小军等）
    common_names = ['德明', '小军', '父亲', '母亲', '老婆', '女朋友']
    for name in common_names:
        if name in text:
            # 如果前面有姓氏，提取全名
            full_name_pattern = f'[张王李赵刘陈杨黄周吴徐孙马朱胡郭何高林罗郑梁宋谢唐韩曹许邓萧冯曾程蔡彭潘袁于董余苏叶吕魏蒋田杜丁沈姜范江傅钟卢汪戴崔任陆廖姚方金邱夏谭韦贾邹石熊孟秦阎薛侯雷白龙段郝孔邵史毛常万顾赖武康贺严尹钱施牛洪龚]{name}'
            full_matches = re.findall(full_name_pattern, text)
            if full_matches:
                entities.extend(full_matches)
            else:
                entities.append(name)

    # 4. 历史人物（邓小平等）
    historical_figures = ['邓小平', '毛泽东', '周恩来']
    for fig in historical_figures:
        if fig in text:
            entities.append(fig)

    # 提取地名（常见地名）
    location_keywords = [
        '北京', '上海', '深圳', '广州', '陕北', '延安', '张家塬', '华强北',
        '武汉', '香港', '兰州', '东南亚', '深圳湾', '窑洞', '公社', '生产队',
        '省城', '县城', '渭河', '黄土高坡'
    ]
    for loc in location_keywords:
        if loc in text:
            entities.append(loc)

    # 提取机构/组织
    org_keywords = ['联想', '海尔', '知青', '研究所']
    for org in org_keywords:
        if org in text:
            entities.append(org)

    # 去重并保持顺序
    seen = set()
    unique_entities = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            unique_entities.append(e)

    return unique_entities


def _extract_entities_info(chapter: Any, record: SegmentEvalRecord) -> Dict[str, Any]:
    """提取实体信息：RAG检索的实体 vs 生成文本中的实体"""

    # 1. RAG 检索到的实体
    rag_entities = []
    for e in (chapter.retrieval_result.entities or [])[:15]:
        name = e.get("name") or e.get("title") or ""
        if name:
            rag_entities.append(name)

    # 2. 生成文本中提取的实体
    generated_text = chapter.generation.content
    text_entities = _extract_entities_from_text(generated_text)

    # 3. 计算覆盖情况
    rag_entities_in_text = [e for e in rag_entities if e in generated_text]
    text_entities_not_in_rag = [e for e in text_entities if e not in rag_entities]

    return {
        "rag_retrieved": rag_entities,
        "rag_used_in_text": rag_entities_in_text,
        "rag_coverage": len(rag_entities_in_text) / len(rag_entities) if rag_entities else 0,
        "text_extracted": text_entities,
        "text_only_entities": text_entities_not_in_rag,
    }


def document_year_diversity(merged_text: str) -> MetricResult:
    """篇级轻量：抽取年份，多年代时略降分。"""
    years = re.findall(r"(?:19|20)\d{2}", merged_text)
    if not years:
        return MetricResult(
            name="year_diversity",
            value=0.5,
            max_value=1.0,
            explanation="合并文本中未检出四位年份",
        )
    uniq = sorted(set(years))
    if len(uniq) <= 3:
        return MetricResult(
            name="year_diversity",
            value=1.0,
            max_value=1.0,
            explanation=f"检出年份种类 {len(uniq)}：{', '.join(uniq[:5])}",
        )
    return MetricResult(
        name="year_diversity",
        value=0.75,
        max_value=1.0,
        explanation=f"多年代穿插 {len(uniq)} 种，属长文常见情况",
    )


@dataclass
class SegmentEvalRecord:
    segment_index: int
    memoir_snippet: str
    generated_snippet: str
    evaluation: Optional[Any]
    metrics: Dict[str, MetricResult]
    fact_check: Optional[Any]
    fact_check_skipped_reason: Optional[str] = None
    novel_content_info: Optional[Dict[str, Any]] = None  # 新内容评估信息


@dataclass
class LongFormEvalResult:
    segments: List[SegmentEvalRecord]
    document_metrics: Dict[str, MetricResult]
    cross_chapter_metrics: Dict[str, MetricResult]
    aggregated_score: float
    summary_text: str
    quality_gate: Optional[QualityGateResult] = None
    raw_json_ready: Dict[str, Any] = field(default_factory=dict)


def _metrics_for_segment(
    memoir_snippet: str,
    generated_snippet: str,
    length_hint: str,
    retrieval_entities: Optional[List[str]] = None,
    reference_year: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    novel_content_brief: Optional[Any] = None,
    task_type: str = "expansion",
    min_required_entities: int = 2,
) -> Dict[str, MetricResult]:
    lo, hi = _parse_length_hint_range(length_hint)

    if task_type == "expansion":
        # For expansion: allow much more flexibility
        # Original memoir length as baseline
        memoir_len = len(memoir_snippet)

        # Expansion factor: 2-6x is reasonable
        margin = max(200, hi - lo)

        literary_length_min = max(50, memoir_len)  # At least as long as original
        literary_length_max = max(hi * 3, memoir_len * 8)  # Allow 3x hint or 8x original
        literary_optimal_min = max(memoir_len * 2, lo)  # At least 2x original
        literary_optimal_max = max(memoir_len * 5, hi)  # Up to 5x original
    else:
        # For summarization: keep strict bounds
        margin = max(80, (hi - lo) // 2)
        literary_length_min = max(50, lo - margin)
        literary_length_max = min(20000, hi + margin * 2)
        literary_optimal_min = max(50, lo)
        literary_optimal_max = max(lo + 1, hi)

    return calculate_all_metrics(
        memoir_snippet,
        generated_snippet,
        reference_entities=retrieval_entities or [],
        reference_year=reference_year,
        keywords=keywords,
        literary_length_min=literary_length_min,
        literary_length_max=literary_length_max,
        literary_optimal_min=literary_optimal_min,
        literary_optimal_max=literary_optimal_max,
        literary_paragraph_relaxed=True,
        novel_content_brief=novel_content_brief,
        task_type=task_type,
        min_required_entities=min_required_entities,
    )


async def evaluate_long_form(
    long_form: Any,
    *,
    llm_adapter: Optional[Any] = None,
    use_llm_eval: bool = False,
    enable_fact_check: bool = True,
    max_atomic_facts_per_segment: int = 12,
    fact_check_timeout_per_segment: float = 60.0,
    use_rule_decompose: bool = True,
    batch_size: int = 5,
    quality_thresholds: Optional[QualityThresholds] = None,
    enable_quality_gate: bool = True,
) -> LongFormEvalResult:
    """
    对分章生成结果跑通评估 pipeline：
    1. 段级指标 + 可选 LLM-as-Judge + 可选事实检查
    2. 篇级跨章指标（重复度、风格一致性、总结句比率）
    3. 质量门控（通过/不通过 + 修复建议）
    """
    from .evaluator import Evaluator
    from .factscore_adapter import FActScoreChecker

    evaluator = Evaluator(llm_adapter=llm_adapter)
    fact_checker: Optional[Any] = None
    if enable_fact_check:
        fact_checker = FActScoreChecker(
            llm_adapter=llm_adapter,
            use_rule_decompose=use_rule_decompose,
        )

    async def _eval_one_chapter(ch: Any) -> SegmentEvalRecord:
        """评估单章（指标 + evaluator + fact_check），可并发。"""
        gen_text = ch.generation.content

        # 获取 novel_content_brief（如果可用）
        novel_brief = getattr(ch.retrieval_result, '_novel_content_brief', None)

        # 使用 novel entities 作为参考实体
        ref_entities: List[str] = []
        if novel_brief:
            ref_entities = novel_brief.novel_entity_names[:10]
        else:
            # 回退到所有实体
            for e in (ch.retrieval_result.entities or [])[:15]:
                name = e.get("name") or e.get("title") or ""
                if name:
                    ref_entities.append(name)

        ctx = ch.retrieval_result.context
        hint = ch.length_hint or f"{max(80, len(gen_text) // 2)}-{max(100, len(gen_text) + 100)}字"

        metrics = _metrics_for_segment(
            ch.segment_text, gen_text, hint,
            retrieval_entities=ref_entities,
            reference_year=ctx.year,
            keywords=ctx.keywords,
            novel_content_brief=novel_brief,
            task_type="expansion",
            min_required_entities=2,
        )

        # 提取新内容评估信息
        novel_content_info = None
        if novel_brief is not None:
            # 调用 analyze_novel_content 获取完整分析结果
            from .novel_content_metrics import analyze_novel_content

            analysis = analyze_novel_content(
                memoir_text=ch.segment_text,
                generated_text=gen_text,
                novel_content_brief=novel_brief,
            )

            novel_content_info = {
                "has_novel_content": novel_brief.has_novel_content,
                "novel_entities_available": analysis.novel_entities_available,
                "novel_entities_used": analysis.novel_entities_used,
                "new_facts_in_output": analysis.new_facts_in_output,
                "grounded_facts": analysis.grounded_facts,
                "ungrounded_facts": analysis.ungrounded_facts,
                "information_gain": analysis.information_gain,
                "expansion_grounding": analysis.expansion_grounding,
                "summary": novel_brief.summary,
            }
            # 从 metrics 中提取指标值（覆盖上面的计算值，保持一致）
            if "information_gain" in metrics:
                novel_content_info["information_gain"] = metrics["information_gain"].value
            if "expansion_grounding" in metrics:
                novel_content_info["expansion_grounding"] = metrics["expansion_grounding"].value

        eval_result: Optional[Any] = None
        try:
            eval_result = await evaluator.evaluate(
                memoir_text=ch.segment_text,
                generated_text=gen_text,
                retrieval_result=ch.retrieval_result,
                use_llm=use_llm_eval and llm_adapter is not None,
                enable_fact_check=False,
            )
        except Exception:
            eval_result = None

        fc: Optional[Any] = None
        skip_reason: Optional[str] = None
        if enable_fact_check and llm_adapter and fact_checker is not None:
            try:
                fc = await asyncio.wait_for(
                    fact_checker.check(
                        memoir_text=ch.segment_text,
                        generated_text=gen_text,
                        retrieval_result=ch.retrieval_result,
                        use_llm=True,
                        use_rule_decompose=use_rule_decompose,
                        max_atomic_facts=max_atomic_facts_per_segment,
                        batch_size=batch_size,
                    ),
                    timeout=fact_check_timeout_per_segment,
                )
            except asyncio.TimeoutError:
                skip_reason = "fact_check_timeout"
            except Exception as e:
                skip_reason = f"fact_check_error:{e}"
        elif enable_fact_check and not llm_adapter and fact_checker is not None:
            fc = await fact_checker.check(
                memoir_text=ch.segment_text,
                generated_text=gen_text,
                retrieval_result=ch.retrieval_result,
                use_llm=False,
                max_atomic_facts=max_atomic_facts_per_segment,
                batch_size=batch_size,
            )

        return SegmentEvalRecord(
            segment_index=ch.segment_index,
            memoir_snippet=ch.segment_text[:200] + ("…" if len(ch.segment_text) > 200 else ""),
            generated_snippet=gen_text[:200] + ("…" if len(gen_text) > 200 else ""),
            evaluation=eval_result,
            metrics=metrics,
            fact_check=fc,
            fact_check_skipped_reason=skip_reason,
            novel_content_info=novel_content_info,
        )

    # ---- 并发评估所有章节 ----
    records: List[SegmentEvalRecord] = list(
        await asyncio.gather(*[_eval_one_chapter(ch) for ch in long_form.chapters])
    )

    weights: List[float] = []
    seg_scores: List[float] = []
    fact_scores_list: List[Optional[float]] = []
    chapters_content: List[str] = []
    target_chars_list: List[int] = []

    for ch, rec in zip(long_form.chapters, records):
        gen_text = ch.generation.content
        chapters_content.append(gen_text)
        weights.append(float(max(1, len(gen_text))))

        hint = ch.length_hint or f"{max(80, len(gen_text) // 2)}-{max(100, len(gen_text) + 100)}字"
        lo, hi = _parse_length_hint_range(hint)
        target_chars_list.append((lo + hi) // 2)

        seg_agg = aggregate_scores(rec.metrics)
        seg_scores.append(rec.evaluation.overall_score if rec.evaluation else seg_agg)
        fact_scores_list.append(rec.fact_check.factscore if rec.fact_check else None)

    # ---- 段级加权综合分 ----
    wsum = sum(weights) or 1.0
    aggregated = sum(s * (weights[i] / wsum) for i, s in enumerate(seg_scores))

    # ---- 篇级指标 ----
    merged = long_form.merged_content
    doc_metrics: Dict[str, MetricResult] = {}
    doc_metrics["year_diversity"] = document_year_diversity(merged)
    if merged:
        om = max(800, min(len(merged), 12000))
        doc_metrics["merged_length"] = LiteraryMetrics.length_score(
            merged,
            min_length=400,
            max_length=50000,
            optimal_min=800,
            optimal_max=om,
        )

    # ---- 跨章指标 ----
    cross_metrics: Dict[str, MetricResult] = {}
    if len(chapters_content) >= 2:
        cross_metrics["inter_chapter_repetition"] = \
            CrossChapterMetrics.inter_chapter_repetition(chapters_content)
        cross_metrics["style_consistency"] = \
            CrossChapterMetrics.style_consistency(chapters_content)
        cross_metrics["summary_sentence_ratio"] = \
            CrossChapterMetrics.summary_sentence_ratio(chapters_content)

    # ---- 质量门控 ----
    gate_result: Optional[QualityGateResult] = None
    if enable_quality_gate:
        gate_result = check_quality_gate(
            chapters_content,
            segment_scores=seg_scores,
            fact_scores=fact_scores_list,
            target_chars_per_chapter=target_chars_list,
            thresholds=quality_thresholds,
        )

    # ---- 摘要文本 ----
    summary_lines = [
        f"分章数: {len(records)}",
        f"加权综合分(段级): {aggregated:.2f}",
    ]
    for r in records:
        line = f"- 段 {r.segment_index}: 指标聚合 {aggregate_scores(r.metrics):.2f}"
        if r.evaluation:
            line += f" | Evaluator {r.evaluation.overall_score:.2f}"
        if r.fact_check is not None:
            line += f" | 事实检查 {'通过' if r.fact_check.is_factual else '待核'}"
            line += f" (FActScore {r.fact_check.factscore:.0%})"
        elif r.fact_check_skipped_reason:
            line += f" | 事实检查({r.fact_check_skipped_reason})"
        # 新内容评估信息
        if r.novel_content_info:
            nci = r.novel_content_info
            if "novel_content_ratio" in nci:
                line += f" | 新内容引入率 {nci['novel_content_ratio']:.0%}"
            if "novel_content_grounding" in nci:
                line += f" | 溯源率 {nci['novel_content_grounding']:.0%}"
        summary_lines.append(line)

    # 跨章指标摘要
    if cross_metrics:
        summary_lines.append("跨章指标:")
        for name, mr in cross_metrics.items():
            summary_lines.append(f"  {name}: {mr.value:.2f} ({mr.explanation})")

    # 质量门控摘要
    if gate_result:
        summary_lines.append(f"质量门控: {'通过' if gate_result.passed else '未通过'}")
        if gate_result.remediation:
            regen_ids = gate_result.remediation.chapters_to_regenerate
            summary_lines.append(f"  建议重新生成: 第 {', '.join(str(c+1) for c in regen_ids)} 章")
            for ch_idx, reasons in gate_result.remediation.reasons.items():
                for reason in reasons:
                    summary_lines.append(f"    第{ch_idx+1}章: {reason}")

    # ---- 构建 JSON ----
    raw: Dict[str, Any] = {
        "aggregated_score": aggregated,
        "segment_count": len(records),
        "segments": [
            {
                "segment_index": r.segment_index,
                "metrics": {
                    k: {"value": v.value, "max": v.max_value, "explanation": v.explanation}
                    for k, v in r.metrics.items()
                },
                "eval_overall": r.evaluation.overall_score if r.evaluation else None,
                "fact_is_factual": r.fact_check.is_factual if r.fact_check else None,
                "fact_score": r.fact_check.factscore if r.fact_check else None,
                "fact_skip": r.fact_check_skipped_reason,
                "entities": _extract_entities_info(long_form.chapters[i], r) if i < len(long_form.chapters) else {},
                "novel_content": r.novel_content_info,
            }
            for i, r in enumerate(records)
        ],
        "document": {k: {"value": v.value, "max": v.max_value} for k, v in doc_metrics.items()},
        "cross_chapter": {
            k: {"value": v.value, "max": v.max_value, "explanation": v.explanation}
            for k, v in cross_metrics.items()
        },
        "quality_gate": {
            "passed": gate_result.passed if gate_result else None,
            "overall_score": gate_result.overall_score if gate_result else None,
            "chapters_to_regenerate": (
                gate_result.remediation.chapters_to_regenerate
                if gate_result and gate_result.remediation else []
            ),
        },
    }

    return LongFormEvalResult(
        segments=records,
        document_metrics=doc_metrics,
        cross_chapter_metrics=cross_metrics,
        aggregated_score=aggregated,
        summary_text="\n".join(summary_lines),
        quality_gate=gate_result,
        raw_json_ready=raw,
    )


def long_form_eval_to_json(result: LongFormEvalResult) -> str:
    return json.dumps(result.raw_json_ready, ensure_ascii=False, indent=2)
