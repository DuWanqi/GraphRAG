"""分章编排 + 长文评估 E2E（纯数据结构 Mock，不加载检索/LLM）。
覆盖：段级指标、跨章指标、质量门控、修复建议。
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.long_form_eval import evaluate_long_form
from src.evaluation.quality_gate import check_quality_gate, QualityThresholds


@dataclass
class _Ctx:
    original_text: str = ""
    year: Optional[str] = None
    location: Optional[str] = None
    keywords: List[str] = field(default_factory=list)


@dataclass
class _RR:
    query: str
    context: _Ctx
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Any] = field(default_factory=list)


@dataclass
class _GR:
    content: str
    provider: str
    model: str


@dataclass
class _Ch:
    segment_index: int
    segment_text: str
    retrieval_result: _RR
    generation: _GR
    length_hint: str = ""


@dataclass
class _LF:
    chapters: List[_Ch]
    merged_content: str
    full_memoir_text: str
    segments: List[Any] = field(default_factory=list)


def _make_mock_long_form():
    ctx1 = _Ctx(original_text="seg1", year="1988", location="深圳", keywords=["特区"])
    ctx2 = _Ctx(original_text="seg2", year="1992", location="广州", keywords=["南巡"])
    rr1 = _RR(query="1988 深圳", context=ctx1, entities=[{"name": "深圳"}])
    rr2 = _RR(query="1992 广州", context=ctx2, entities=[{"name": "广州"}])
    return _LF(
        chapters=[
            _Ch(
                0,
                "回忆录第一段内容。" * 30,
                rr1,
                _GR(
                    content="那时候的深圳还是一片工地，1988年的华强北刚刚起步。当时到处是从内地来的年轻人。",
                    provider="mock",
                    model="mock",
                ),
                length_hint="100-200字",
            ),
            _Ch(
                1,
                "回忆录第二段内容。" * 30,
                rr2,
                _GR(
                    content="1992年春天南巡讲话之后，广州的街头巷尾都在议论改革。那年珠三角的工厂如雨后春笋。",
                    provider="mock",
                    model="mock",
                ),
                length_hint="100-200字",
            ),
        ],
        merged_content=(
            "那时候的深圳还是一片工地，1988年的华强北刚刚起步。当时到处是从内地来的年轻人。"
            "\n\n---\n\n"
            "1992年春天南巡讲话之后，广州的街头巷尾都在议论改革。那年珠三角的工厂如雨后春笋。"
        ),
        full_memoir_text="全文",
    )


def test_evaluate_long_form_pipeline_completes():
    """基础流程：段级指标 + 跨章指标 + 摘要 + JSON。"""
    lf = _make_mock_long_form()
    result = asyncio.run(
        evaluate_long_form(
            lf,
            llm_adapter=None,
            use_llm_eval=False,
            enable_fact_check=False,
            enable_quality_gate=True,
            max_atomic_facts_per_segment=4,
        )
    )
    assert result.aggregated_score >= 0
    assert len(result.segments) == 2
    assert "分章数" in result.summary_text
    assert result.raw_json_ready["segment_count"] == 2


def test_evaluate_long_form_has_cross_chapter_metrics():
    """评估结果应包含跨章指标。"""
    lf = _make_mock_long_form()
    result = asyncio.run(
        evaluate_long_form(
            lf,
            llm_adapter=None,
            use_llm_eval=False,
            enable_fact_check=False,
        )
    )
    assert "inter_chapter_repetition" in result.cross_chapter_metrics
    assert "style_consistency" in result.cross_chapter_metrics
    assert "summary_sentence_ratio" in result.cross_chapter_metrics
    # 两段内容完全不同，重复率应该很低 → 评分接近 1.0
    rep = result.cross_chapter_metrics["inter_chapter_repetition"]
    assert rep.value >= 0.5


def test_evaluate_long_form_has_quality_gate():
    """评估结果应包含质量门控。"""
    lf = _make_mock_long_form()
    result = asyncio.run(
        evaluate_long_form(
            lf,
            llm_adapter=None,
            use_llm_eval=False,
            enable_fact_check=False,
            enable_quality_gate=True,
        )
    )
    assert result.quality_gate is not None
    assert "quality_gate" in result.raw_json_ready
    assert "cross_chapter" in result.raw_json_ready
    assert "质量门控" in result.summary_text


def test_evaluate_long_form_json_has_fact_score():
    """JSON 输出应包含 fact_score 字段。"""
    lf = _make_mock_long_form()
    result = asyncio.run(
        evaluate_long_form(
            lf,
            llm_adapter=None,
            use_llm_eval=False,
            enable_fact_check=False,
        )
    )
    for seg_json in result.raw_json_ready["segments"]:
        assert "fact_score" in seg_json


# ---------------------------------------------------------------------------
# 质量门控独立测试
# ---------------------------------------------------------------------------

def test_quality_gate_pass():
    """正常内容应通过门控。"""
    chapters = [
        "1988年的深圳还是一片热土，华强北的电子元器件生意刚刚起步。",
        "1992年南巡之后整个珠三角迎来了新一轮的发展浪潮。",
    ]
    result = check_quality_gate(
        chapters,
        segment_scores=[7.5, 8.0],
        fact_scores=[0.9, 0.85],
        target_chars_per_chapter=[50, 50],
    )
    assert result.passed


def test_quality_gate_detects_repetition():
    """高重复内容应触发门控失败。"""
    repeated = "这是一段关于改革开放的历史背景描述，当时深圳特区刚刚成立。" * 10
    chapters = [repeated, repeated]
    result = check_quality_gate(chapters)
    # 完全相同的两章应有高重叠率
    assert any(ci.dimension == "repetition" for ci in result.cross_chapter_issues)


def test_quality_gate_detects_summary_sentences():
    """总结性语句过多应产生警告。"""
    text = "总之改革开放是伟大的。综上所述这个时期很重要。总的来说人民生活改善了。" * 5
    result = check_quality_gate([text])
    ch_result = result.chapter_results[0]
    assert any(iss.dimension == "summary" for iss in ch_result.issues)


def test_quality_gate_remediation_plan():
    """综合分过低的章节应出现在修复计划中。"""
    chapters = ["短内容", "另一段短内容"]
    result = check_quality_gate(
        chapters,
        segment_scores=[3.0, 8.0],
        thresholds=QualityThresholds(min_segment_score=5.0),
    )
    assert result.remediation is not None
    assert 0 in result.remediation.chapters_to_regenerate
