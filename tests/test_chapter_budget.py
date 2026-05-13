"""章节字数预算单测 + 分段元数据兼容性。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.memoir_segmenter import MemoirSegment, SegmentMeta
from src.generation.chapter_budget import (
    allocate_segment_budgets,
    allocate_segment_budgets_uniform,
    legacy_maps_for_single_segment,
)


def test_allocate_uniform_per_chapter():
    segs = [
        MemoirSegment(0, "a" * 400),
        MemoirSegment(1, "b" * 600),
    ]
    budgets = allocate_segment_budgets_uniform(segs, 500, 900)
    assert len(budgets) == 2
    assert budgets[0].length_hint == budgets[1].length_hint
    assert budgets[0].target_chars == budgets[1].target_chars


def test_allocate_proportional():
    segs = [
        MemoirSegment(0, "a" * 400),
        MemoirSegment(1, "b" * 600),
    ]
    budgets = allocate_segment_budgets(segs, "400-800")
    assert len(budgets) == 2
    assert budgets[0].target_chars < budgets[1].target_chars
    assert "字" in budgets[0].length_hint
    assert budgets[0].max_tokens >= 256


def test_legacy_maps():
    h, m = legacy_maps_for_single_segment("200-400")
    assert "200" in h
    assert m == 800


def test_allocate_with_meta():
    """带元数据的 MemoirSegment 应正常工作。"""
    meta = SegmentMeta(
        detected_years=("1972",),
        detected_locations=("陕北",),
        temporal_label="1972",
        split_reason="temporal_boundary",
    )
    segs = [
        MemoirSegment(0, "a" * 400, meta=meta),
        MemoirSegment(1, "b" * 600, meta=None),
    ]
    budgets = allocate_segment_budgets(segs, "400-800")
    assert len(budgets) == 2
    assert budgets[0].target_chars > 0
