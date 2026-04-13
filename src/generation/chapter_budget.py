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


def allocate_segment_budgets(
    segments: List[MemoirSegment],
    length_bucket: str,
) -> List[SegmentBudget]:
    """
    按各段字符数占全书比例，分配每段生成目标字数与 max_tokens。

    max_tokens 按「目标上限字数的约 2.2 倍」估算（中文保守），并设上下界。
    """
    if not segments:
        return []

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
        # tokens：略宽裕，避免截断
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
