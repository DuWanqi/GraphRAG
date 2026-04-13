"""
长文分章结果的评估聚合：段级指标 + 可选段级事实检查 + 篇级轻量指标。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .metrics import MetricResult, calculate_all_metrics, aggregate_scores, LiteraryMetrics


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


@dataclass
class LongFormEvalResult:
    segments: List[SegmentEvalRecord]
    document_metrics: Dict[str, MetricResult]
    aggregated_score: float
    summary_text: str
    raw_json_ready: Dict[str, Any] = field(default_factory=dict)


def _metrics_for_segment(
    memoir_snippet: str,
    generated_snippet: str,
    length_hint: str,
    retrieval_entities: Optional[List[str]] = None,
    reference_year: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, MetricResult]:
    lo, hi = _parse_length_hint_range(length_hint)
    margin = max(80, (hi - lo) // 2)
    return calculate_all_metrics(
        memoir_snippet,
        generated_snippet,
        reference_entities=retrieval_entities or [],
        reference_year=reference_year,
        keywords=keywords,
        literary_length_min=max(50, lo - margin),
        literary_length_max=min(20000, hi + margin * 2),
        literary_optimal_min=max(50, lo),
        literary_optimal_max=max(lo + 1, hi),
        literary_paragraph_relaxed=True,
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
) -> LongFormEvalResult:
    """
    对分章生成结果跑通评估 pipeline（默认规则评估；事实检查可超时跳过单段）。
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

    records: List[SegmentEvalRecord] = []
    weights: List[float] = []
    seg_scores: List[float] = []

    for ch in long_form.chapters:
        gen_text = ch.generation.content
        w = max(1, len(gen_text))
        weights.append(float(w))

        ref_entities: List[str] = []
        for e in (ch.retrieval_result.entities or [])[:15]:
            name = e.get("name") or e.get("title") or ""
            if name:
                ref_entities.append(name)

        ctx = ch.retrieval_result.context
        hint = ch.length_hint or f"{max(80, len(gen_text) // 2)}-{max(100, len(gen_text) + 100)}字"
        metrics = _metrics_for_segment(
            ch.segment_text,
            gen_text,
            hint,
            retrieval_entities=ref_entities,
            reference_year=ctx.year,
            keywords=ctx.keywords,
        )

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

        seg_agg = aggregate_scores(metrics)
        if eval_result:
            seg_scores.append(eval_result.overall_score)
        else:
            seg_scores.append(seg_agg)

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
            )

        records.append(
            SegmentEvalRecord(
                segment_index=ch.segment_index,
                memoir_snippet=ch.segment_text[:200] + ("…" if len(ch.segment_text) > 200 else ""),
                generated_snippet=gen_text[:200] + ("…" if len(gen_text) > 200 else ""),
                evaluation=eval_result,
                metrics=metrics,
                fact_check=fc,
                fact_check_skipped_reason=skip_reason,
            )
        )

    wsum = sum(weights) or 1.0
    aggregated = sum(s * (weights[i] / wsum) for i, s in enumerate(seg_scores))

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
        elif r.fact_check_skipped_reason:
            line += f" | 事实检查({r.fact_check_skipped_reason})"
        summary_lines.append(line)

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
                "fact_skip": r.fact_check_skipped_reason,
            }
            for r in records
        ],
        "document": {k: {"value": v.value, "max": v.max_value} for k, v in doc_metrics.items()},
    }

    return LongFormEvalResult(
        segments=records,
        document_metrics=doc_metrics,
        aggregated_score=aggregated,
        summary_text="\n".join(summary_lines),
        raw_json_ready=raw,
    )


def long_form_eval_to_json(result: LongFormEvalResult) -> str:
    return json.dumps(result.raw_json_ready, ensure_ascii=False, indent=2)
