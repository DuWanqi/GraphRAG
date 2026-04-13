"""章节字数预算单测。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.memoir_segmenter import MemoirSegment
from src.generation.chapter_budget import allocate_segment_budgets, legacy_maps_for_single_segment


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
