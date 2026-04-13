"""分章编排 + 长文评估 E2E（纯数据结构 Mock，不加载检索/LLM）。"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.long_form_eval import evaluate_long_form


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


def test_evaluate_long_form_pipeline_completes():
    ctx1 = _Ctx(original_text="seg1", year="1988", location="深圳", keywords=["特区"])
    ctx2 = _Ctx(original_text="seg2", year="1992", location="广州", keywords=["南巡"])
    rr1 = _RR(query="1988 深圳", context=ctx1, entities=[{"name": "深圳"}])
    rr2 = _RR(query="1992 广州", context=ctx2, entities=[{"name": "广州"}])
    lf = _LF(
        chapters=[
            _Ch(
                0,
                "回忆录第一段内容。" * 30,
                rr1,
                _GR(
                    content="历史背景第一段，关于1988年与深圳特区。",
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
                    content="历史背景第二段，关于1992年改革氛围。",
                    provider="mock",
                    model="mock",
                ),
                length_hint="100-200字",
            ),
        ],
        merged_content="历史背景第一段...\n\n---\n\n历史背景第二段...",
        full_memoir_text="全文",
    )
    result = asyncio.run(
        evaluate_long_form(
            lf,
            llm_adapter=None,
            use_llm_eval=False,
            enable_fact_check=False,
            max_atomic_facts_per_segment=4,
        )
    )
    assert result.aggregated_score >= 0
    assert len(result.segments) == 2
    assert "分章数" in result.summary_text
    assert result.raw_json_ready["segment_count"] == 2
