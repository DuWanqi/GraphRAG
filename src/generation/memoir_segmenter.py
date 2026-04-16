"""
回忆录长文分段：时间边界优先 + 结构辅助 + 长度约束，无 LLM 依赖。

分段策略（优先级从高到低）：
1. 显式时间边界：段落以新年份开头时，强制切分
2. 结构边界：空行、章节标题
3. 长度约束：超长块按句切分，过短块合并（但不跨时间边界合并）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 正则模式
# ---------------------------------------------------------------------------

# 章节/小节标题行（行首）
_CHAPTER_LINE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千万零〇○0-9]+[章节回部编篇集]|"
    r"[一二三四五六七八九十]+、|"
    r"[（(]?[一二三四五六七八九十]+[）)]、?)\s*",
    re.MULTILINE,
)

# 匹配四位年份（阿拉伯 + 中文大写）
_YEAR_PATTERN = re.compile(
    r"(?:一九[〇零○一二三四五六七八九]{2}|二[〇零○][〇零○一二三四五六七八九]{2}|(?:19|20)\d{2})"
)

# 段首年份标记——用于判断一个文本块是否以新年份开头
_LEADING_YEAR = re.compile(
    r"^\s*(?:一九[〇零○一二三四五六七八九]{2}|二[〇零○][〇零○一二三四五六七八九]{2}|(?:19|20)\d{2})\s*年"
)

# 中文数字到阿拉伯数字映射
_CN_DIGIT: Dict[str, str] = {
    "零": "0", "〇": "0", "○": "0",
    "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
    "六": "6", "七": "7", "八": "8", "九": "9",
}

# 地名关键词
_LOCATION_KEYWORDS = [
    "北京", "上海", "广州", "深圳", "天津", "重庆", "南京", "武汉", "成都",
    "杭州", "西安", "长沙", "沈阳", "哈尔滨", "大连", "青岛", "厦门", "苏州",
    "东北", "华北", "华南", "西南", "西北", "华东", "陕北", "延安", "香港",
    "台湾", "新疆", "西藏", "内蒙古", "广东", "浙江", "江苏", "福建", "四川",
]

# 人物称谓
_PERSON_PATTERNS = re.compile(
    r"(?:老[A-Za-z\u4e00-\u9fa5]|[A-Za-z\u4e00-\u9fa5]老师|"
    r"[A-Za-z\u4e00-\u9fa5]{2,3}(?:同志|队长|书记|厂长|经理|校长|主任))"
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentMeta:
    """分段元数据：记录该段的时间、地点、人物等结构化信息。"""

    detected_years: Tuple[str, ...] = ()
    detected_locations: Tuple[str, ...] = ()
    detected_figures: Tuple[str, ...] = ()
    temporal_label: str = ""          # e.g. "1972-1977"
    split_reason: str = ""            # 切分原因

    def to_dict(self) -> Dict[str, object]:
        return {
            "years": list(self.detected_years),
            "locations": list(self.detected_locations),
            "figures": list(self.detected_figures),
            "temporal_label": self.temporal_label,
            "split_reason": self.split_reason,
        }


@dataclass(frozen=True)
class MemoirSegment:
    """一段待检索/生成的回忆录切片。"""

    index: int
    text: str
    meta: Optional[SegmentMeta] = None


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


def _merge_short(
    blocks: List[str],
    target_min_chars: int,
    target_max_chars: int,
    *,
    respect_temporal: bool = True,
) -> List[str]:
    """合并过短块，但 **不跨时间边界** 合并（respect_temporal=True 时）。"""
    if not blocks:
        return []
    merged: List[str] = []
    buf = blocks[0]
    for b in blocks[1:]:
        # 如果下一个块以新年份开头，即使当前块很短也不合并
        crosses_time = respect_temporal and _LEADING_YEAR.match(b)
        if len(buf) < target_min_chars and not crosses_time:
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


# ---------------------------------------------------------------------------
# 元数据提取
# ---------------------------------------------------------------------------

def _cn_year_to_arabic(raw: str) -> str:
    """将中文年份字符串转为四位阿拉伯数字，如 '一九七二' → '1972'。"""
    digits = ""
    for ch in raw:
        if ch in _CN_DIGIT:
            digits += _CN_DIGIT[ch]
        elif ch.isdigit():
            digits += ch
    return digits if len(digits) == 4 else raw


def extract_years(text: str) -> List[str]:
    """从文本中提取所有四位年份（去重、排序、阿拉伯数字形式）。"""
    raw_matches = _YEAR_PATTERN.findall(text)
    seen: dict[str, None] = {}
    for m in raw_matches:
        arabic = _cn_year_to_arabic(m)
        if arabic not in seen:
            seen[arabic] = None
    return list(seen.keys())


def _extract_locations(text: str) -> List[str]:
    """从文本中提取地名关键词（去重、保持首次出现顺序）。"""
    found: dict[str, None] = {}
    for loc in _LOCATION_KEYWORDS:
        if loc in text and loc not in found:
            found[loc] = None
    return list(found.keys())


def _extract_figures(text: str) -> List[str]:
    """从文本中提取人物称谓（去重）。"""
    matches = _PERSON_PATTERNS.findall(text)
    seen: dict[str, None] = {}
    for m in matches:
        if m not in seen:
            seen[m] = None
    return list(seen.keys())


def _build_temporal_label(years: List[str]) -> str:
    """根据年份列表生成时间范围标签，如 '1972-1977'。"""
    if not years:
        return ""
    int_years = []
    for y in years:
        try:
            int_years.append(int(y))
        except ValueError:
            pass
    if not int_years:
        return years[0]
    lo, hi = min(int_years), max(int_years)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _build_meta(text: str, split_reason: str = "") -> SegmentMeta:
    """为一个文本块构建元数据。"""
    years = extract_years(text)
    locations = _extract_locations(text)
    figures = _extract_figures(text)
    return SegmentMeta(
        detected_years=tuple(years),
        detected_locations=tuple(locations),
        detected_figures=tuple(figures),
        temporal_label=_build_temporal_label(years),
        split_reason=split_reason,
    )


# ---------------------------------------------------------------------------
# 分段质量校验
# ---------------------------------------------------------------------------

@dataclass
class SegmentationIssue:
    """分段质量问题。"""
    severity: str            # "warning" | "error"
    segment_index: int
    message: str


@dataclass
class SegmentationReport:
    """分段校验报告——向调用方说明分段质量与潜在风险。"""

    segment_count: int
    total_chars: int
    issues: List[SegmentationIssue]
    segment_summaries: List[Dict[str, object]]

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def to_text(self) -> str:
        lines = [
            f"分段数: {self.segment_count}，总字数: {self.total_chars}",
            f"校验结果: {'通过' if self.passed else '存在问题'}",
        ]
        for s in self.segment_summaries:
            lines.append(
                f"  段{s['index']}: {s['chars']}字 | 时间={s['temporal_label'] or '无'} "
                f"| 地点={','.join(s['locations']) if s['locations'] else '无'} "
                f"| 切分原因={s['split_reason'] or '-'}"
            )
        if self.issues:
            lines.append("问题:")
            for iss in self.issues:
                lines.append(f"  [{iss.severity}] 段{iss.segment_index}: {iss.message}")
        return "\n".join(lines)


def validate_segmentation(
    segments: List[MemoirSegment],
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> SegmentationReport:
    """
    校验分段结果，返回结构化报告。

    检查维度:
    - 每段长度是否在合理范围
    - 是否存在跨时间边界的段（同一段覆盖多个不相干年代）
    - 段索引连续性
    """
    issues: List[SegmentationIssue] = []
    summaries: List[Dict[str, object]] = []
    total_chars = sum(len(s.text) for s in segments)

    for s in segments:
        meta = s.meta or _build_meta(s.text)
        summaries.append({
            "index": s.index,
            "chars": len(s.text),
            "temporal_label": meta.temporal_label,
            "locations": list(meta.detected_locations),
            "split_reason": meta.split_reason,
        })
        # 长度检查
        if len(s.text) < target_min_chars * 0.5:
            issues.append(SegmentationIssue("warning", s.index,
                          f"段落过短 ({len(s.text)}字 < {target_min_chars // 2}字下限)"))
        if len(s.text) > target_max_chars * 3:
            issues.append(SegmentationIssue("error", s.index,
                          f"段落过长 ({len(s.text)}字 > {target_max_chars * 3}字上限)"))
        # 跨年代检查
        years = meta.detected_years
        if len(years) >= 2:
            try:
                span = int(years[-1]) - int(years[0])
                if span > 15:
                    issues.append(SegmentationIssue("warning", s.index,
                                  f"单段跨 {span} 年 ({years[0]}→{years[-1]})，建议拆分"))
            except ValueError:
                pass

    # 索引连续性
    for j, s in enumerate(segments):
        if s.index != j:
            issues.append(SegmentationIssue("error", j,
                          f"索引不连续: 期望 {j}，实际 {s.index}"))

    return SegmentationReport(
        segment_count=len(segments),
        total_chars=total_chars,
        issues=issues,
        segment_summaries=summaries,
    )


# ---------------------------------------------------------------------------
# 核心分段入口
# ---------------------------------------------------------------------------

def segment_memoir(
    text: str,
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> List[MemoirSegment]:
    """
    将回忆录分为多段，便于分章检索与生成。

    策略:
    1. 按空行 / 章节标题拆出结构块
    2. 超长块按句切分
    3. 过短块合并——但不跨时间边界合并
    4. 为每段附加元数据 (SegmentMeta)
    5. 全文极短（低于 target_min）时返回单段
    """
    text = (text or "").strip()
    if not text:
        return []

    if len(text) < target_min_chars:
        return [MemoirSegment(0, text, meta=_build_meta(text, "single_short"))]

    blocks = _structural_blocks(text)
    expanded: List[str] = []
    for b in blocks:
        expanded.extend(_split_oversized(b, target_max_chars))

    merged = _merge_short(expanded, target_min_chars, target_max_chars, respect_temporal=True)

    # 最后兜底：仍有超长的再切一刀
    final: List[str] = []
    for m in merged:
        if len(m) > target_max_chars * 2:
            final.extend(_split_oversized(m, target_max_chars))
        else:
            final.append(m)

    # 推断每段的切分原因
    segments: List[MemoirSegment] = []
    for i, seg_text in enumerate(final):
        seg_text = seg_text.strip()
        if not seg_text:
            continue
        reason = _infer_split_reason(seg_text, i, final)
        meta = _build_meta(seg_text, reason)
        segments.append(MemoirSegment(i, seg_text, meta=meta))

    # 修正索引（因为可能 skip 了空段）
    return [MemoirSegment(j, s.text, meta=s.meta) for j, s in enumerate(segments)]


def _infer_split_reason(block: str, idx: int, all_blocks: List[str]) -> str:
    """推断这个块是因为什么原因被切分出来的。"""
    if idx == 0:
        return "document_start"
    if _LEADING_YEAR.match(block):
        return "temporal_boundary"
    if _CHAPTER_LINE.match(block):
        return "chapter_heading"
    # 前一个块和当前块之间原本有空行（在 _structural_blocks 中切分）
    return "paragraph_break"
