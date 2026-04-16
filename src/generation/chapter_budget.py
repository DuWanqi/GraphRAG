"""
按 UI length_bucket 将「全书生成目标字数」分配到各分段，并给出 length_hint 与 max_tokens。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .memoir_segmenter import MemoirSegment

# 与 web/app.py 的 bucket 键一致；数值为生成正文总目标（中位）及上下限比例
_BUCKET_TOTAL_CENTER: dict[str, int] = {
    "200-400": 300,
    "400-800": 600,
    "800-1200": 1000,
    "1200+": 2000,
}

_DEFAULT_BUCKET = "400-800"


@dataclass(frozen=True)
class SegmentBudget:
    length_hint: str
    max_tokens: int
    target_chars: int


def _hint_from_target(low: int, high: int) -> str:
    low = max(50, int(low))
    high = max(low + 1, int(high))
    return f"{low}-{high}字"


_BUCKET_EXPANSION: dict[str, float] = {
    "200-400": 0.8,
    "400-800": 1.2,
    "800-1200": 1.6,
    "1200+": 2.0,
}


def allocate_segment_budgets(
    segments: List[MemoirSegment],
    length_bucket: str,
) -> List[SegmentBudget]:
    """
    按各段原文字数和扩展系数，分配每段生成目标字数与 max_tokens。

    润色改写模式下，每章的目标字数 = 原文字数 × 扩展系数。
    扩展系数由 length_bucket 决定：
      - "200-400": 0.8× (精简)
      - "400-800": 1.2× (略扩写)
      - "800-1200": 1.6× (丰富扩写)
      - "1200+": 2.0× (大幅扩写)

    对于只有 1 段的情况（单段模式），回退到原来的按 bucket 总量分配。
    """
    if not segments:
        return []

    if len(segments) == 1:
        return _allocate_single_segment(segments, length_bucket)

    bucket = length_bucket if length_bucket in _BUCKET_EXPANSION else _DEFAULT_BUCKET
    expansion = _BUCKET_EXPANSION[bucket]

    out: List[SegmentBudget] = []
    for seg in segments:
        src_len = max(80, len(seg.text))
        center = max(150, int(src_len * expansion))
        low = max(100, int(center * 0.85))
        high = max(low + 50, int(center * 1.15))
        hint = _hint_from_target(low, high)
        max_tokens = min(8000, max(512, int(high * 2.2)))
        out.append(
            SegmentBudget(
                length_hint=hint,
                max_tokens=max_tokens,
                target_chars=center,
            )
        )
    return out


def _allocate_single_segment(
    segments: List[MemoirSegment],
    length_bucket: str,
) -> List[SegmentBudget]:
    """单段模式：按 bucket 总量分配（兼容旧行为）。"""
    bucket = length_bucket if length_bucket in _BUCKET_TOTAL_CENTER else _DEFAULT_BUCKET
    total_center = _BUCKET_TOTAL_CENTER[bucket]
    weights = [max(1, len(s.text)) for s in segments]
    wsum = sum(weights)

    out: List[SegmentBudget] = []
    for seg, w in zip(segments, weights):
        share = w / wsum
        center = max(80, int(total_center * share))
        low = max(50, int(center * 0.85))
        high = max(low + 10, int(center * 1.15))
        hint = _hint_from_target(low, high)
        max_tokens = min(8000, max(256, int(high * 2.2)))
        out.append(
            SegmentBudget(
                length_hint=hint,
                max_tokens=max_tokens,
                target_chars=center,
            )
        )
    return out


def legacy_maps_for_single_segment(length_bucket: str) -> Tuple[str, int]:
    """与 web/app 中单段模式一致的 hint / max_tokens（整篇一次生成）。"""
    length_hint_map = {
        "200-400": "200-400字",
        "400-800": "400-800字",
        "800-1200": "800-1200字",
        "1200+": "1200字以上",
    }
    max_tokens_map = {
        "200-400": 800,
        "400-800": 1400,
        "800-1200": 2200,
        "1200+": 3200,
    }
    b = length_bucket if length_bucket in length_hint_map else _DEFAULT_BUCKET
    return length_hint_map[b], max_tokens_map[b]
