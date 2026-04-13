"""
回忆录长文分段：结构优先 + 长度约束，无 LLM 依赖。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# 章节/小节标题行（行首）
_CHAPTER_LINE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千万零〇○0-9]+[章节回部编篇集]|"
    r"[一二三四五六七八九十]+、|"
    r"[（(]?[一二三四五六七八九十]+[）)]、?)\s*",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MemoirSegment:
    """一段待检索/生成的回忆录切片。"""

    index: int
    text: str


def _split_sentences(text: str) -> List[str]:
    """按中文句末切分，保留分隔符在句尾。"""
    if not text.strip():
        return []
    parts: List[str] = []
    buf: List[str] = []
    for ch in text:
        buf.append(ch)
        if ch in "。！？\n":
            s = "".join(buf).strip()
            if s:
                parts.append(s)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts if parts else [text.strip()]


def _structural_blocks(text: str) -> List[str]:
    """
    优先按空行分段；若块内有多行且某行像标题，可进一步拆开。
    """
    text = text.strip()
    if not text:
        return []
    raw_blocks = re.split(r"\n\s*\n+", text)
    blocks: List[str] = []
    for b in raw_blocks:
        b = b.strip()
        if not b:
            continue
        lines = b.split("\n")
        if len(lines) <= 1:
            blocks.append(b)
            continue
        # 多行块：若第二行起有标题行，按标题再切
        current: List[str] = []
        for line in lines:
            if _CHAPTER_LINE.match(line) and current:
                joined = "\n".join(current).strip()
                if joined:
                    blocks.append(joined)
                current = [line]
            else:
                current.append(line)
        if current:
            joined = "\n".join(current).strip()
            if joined:
                blocks.append(joined)
    return blocks if blocks else [text]


def _split_oversized(block: str, target_max_chars: int) -> List[str]:
    """将超长块按句子切到不超过 target_max_chars（硬上限略放宽）。"""
    if len(block) <= target_max_chars:
        return [block]
    sents = _split_sentences(block)
    if len(sents) <= 1:
        # 无句读：按字符硬切
        out: List[str] = []
        i = 0
        while i < len(block):
            out.append(block[i : i + target_max_chars])
            i += target_max_chars
        return out

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        add_len = len(s) + (1 if cur else 0)
        if cur and cur_len + add_len > target_max_chars:
            chunks.append("".join(cur))
            cur = [s]
            cur_len = len(s)
        else:
            if cur:
                cur_len += add_len
            else:
                cur_len = len(s)
            cur.append(s)
    if cur:
        chunks.append("".join(cur))
    return chunks


def _merge_short(blocks: List[str], target_min_chars: int, target_max_chars: int) -> List[str]:
    """合并过短块直到达到 target_min 或无法合并。"""
    if not blocks:
        return []
    merged: List[str] = []
    buf = blocks[0]
    for b in blocks[1:]:
        if len(buf) < target_min_chars:
            if len(buf) + len(b) + 1 <= target_max_chars * 2:
                buf = f"{buf}\n\n{b}"
            else:
                merged.append(buf)
                buf = b
        else:
            merged.append(buf)
            buf = b
    merged.append(buf)
    return merged


def segment_memoir(
    text: str,
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> List[MemoirSegment]:
    """
    将回忆录分为多段，便于分章检索与生成。

    - 先按空行 / 章节标题拆块；超长块按句切分；过短块合并。
    - 全文极短（低于 target_min）时返回单段。
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) < target_min_chars:
        return [MemoirSegment(0, text)]

    blocks = _structural_blocks(text)
    expanded: List[str] = []
    for b in blocks:
        expanded.extend(_split_oversized(b, target_max_chars))

    merged = _merge_short(expanded, target_min_chars, target_max_chars)

    # 最后兜底：仍有超长的再切一刀
    final: List[str] = []
    for m in merged:
        if len(m) > target_max_chars * 2:
            final.extend(_split_oversized(m, target_max_chars))
        else:
            final.append(m)

    return [MemoirSegment(i, seg.strip()) for i, seg in enumerate(final) if seg.strip()]
