"""
跨章上下文管理器：在分章生成过程中维护章间状态，
确保叙事衔接、避免内容重复、保持风格统一。

设计原则：
- 无 LLM 依赖（纯规则提取 + 字符串拼装）
- 每生成完一章后更新，供下一章 prompt 使用
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .memoir_segmenter import extract_years


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ChapterPosition:
    """当前章在全文中的位置。"""
    index: int
    total: int

    @property
    def is_opening(self) -> bool:
        return self.index == 0

    @property
    def is_closing(self) -> bool:
        return self.index == self.total - 1

    @property
    def role(self) -> str:
        if self.is_opening:
            return "opening"
        if self.is_closing:
            return "closing"
        return "body"

    @property
    def role_instruction(self) -> str:
        if self.is_opening:
            return "本章是全文开篇，请自然引入时代背景，为后续叙事铺垫基调。"
        if self.is_closing:
            return "本章是全文收尾，请在呈现历史背景时带有收束感与回望意味。"
        return "本章位于全文中段，请与前文自然衔接，推进叙事节奏。"


@dataclass
class ChapterRecord:
    """一章生成完成后的记录摘要。"""
    index: int
    brief: str                      # ≤60 字内容概要
    entities_mentioned: List[str]   # 已提及的实体
    time_period: str                # 该章涵盖的年代
    key_phrases: List[str]          # 2-gram / 3-gram 高频短语


# ---------------------------------------------------------------------------
# 上下文管理器
# ---------------------------------------------------------------------------

class ChapterContext:
    """
    在分章生成过程中追踪已生成内容，提供：
    1. 前文概要 → 注入 prompt，让 LLM 知道前面写了什么
    2. 已出现要点 → 注入 prompt，明确要求 LLM 不重复
    3. 章节位置指令 → 开头/中间/结尾不同措辞
    """

    def __init__(self, total_chapters: int):
        self.total_chapters = total_chapters
        self._records: List[ChapterRecord] = []
        self._all_entities: Set[str] = set()
        self._all_key_phrases: List[str] = []

    # ---- 生成后调用 ----

    def record_chapter(
        self,
        index: int,
        content: str,
        entities: Optional[List[str]] = None,
    ) -> None:
        """记录刚生成完毕的章节，用于后续章节的上下文。"""
        brief = _extract_brief(content)
        time_period = _extract_time_period(content)
        key_phrases = _extract_key_phrases(content)
        ent_list = entities or []

        self._records.append(ChapterRecord(
            index=index,
            brief=brief,
            entities_mentioned=ent_list,
            time_period=time_period,
            key_phrases=key_phrases,
        ))
        self._all_entities.update(ent_list)
        self._all_key_phrases.extend(key_phrases)

    # ---- 生成前调用 ----

    def get_position(self, index: int) -> ChapterPosition:
        return ChapterPosition(index, self.total_chapters)

    def build_prompt_section(self, current_index: int) -> str:
        """
        构建插入 prompt 的跨章上下文段落。
        如果是第一章（无前文），返回空字符串。
        """
        position = self.get_position(current_index)
        parts: List[str] = []

        # 1) 前文概要
        if self._records:
            parts.append("## 前文已生成内容概要（请勿重复）")
            for rec in self._records[-3:]:          # 最近 3 章
                parts.append(f"- 第{rec.index + 1}章 ({rec.time_period}): {rec.brief}")

        # 2) 反重复要点
        recent_phrases = self._all_key_phrases[-12:]
        if recent_phrases:
            parts.append("")
            parts.append("以下要点已在前文出现，请勿再次展开或总结：")
            parts.append("、".join(dict.fromkeys(recent_phrases)))  # 去重保序

        # 3) 章节位置指令
        parts.append("")
        parts.append(position.role_instruction)

        return "\n".join(parts) if parts else ""

    # ---- 生成后校验 ----

    def detect_repetition_with_previous(
        self,
        new_content: str,
        *,
        ngram_n: int = 6,
        threshold: float = 0.15,
    ) -> Optional[str]:
        """
        检测新生成章节与前面章节的 n-gram 重叠率。

        Returns:
            重复警告文本，若无问题返回 None。
        """
        if not self._records:
            return None

        new_ngrams = _char_ngrams(new_content, ngram_n)
        if not new_ngrams:
            return None

        for rec in self._records:
            # 用 brief 太短，用完整 key_phrases 也不够；
            # 这里只能对比 n-gram，需要存原文片段。
            # 折中：对比 key_phrases 覆盖率
            pass

        # 实际实现：与之前所有 key_phrases 做交集
        prev_phrases_set = set(self._all_key_phrases)
        current_phrases = _extract_key_phrases(new_content)
        if not prev_phrases_set or not current_phrases:
            return None

        overlap = len(set(current_phrases) & prev_phrases_set)
        ratio = overlap / max(len(current_phrases), 1)
        if ratio >= threshold:
            return (
                f"与前文要点重叠率 {ratio:.0%} (阈值 {threshold:.0%})，"
                f"重叠要点: {', '.join(list(set(current_phrases) & prev_phrases_set)[:5])}"
            )
        return None


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _extract_brief(content: str, max_len: int = 60) -> str:
    """提取内容的前 1-2 句作为概要。"""
    content = content.strip()
    # 按句号/问号/感叹号切
    sentences = re.split(r"[。！？]", content)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return content[:max_len]
    brief = sentences[0]
    if len(brief) < 20 and len(sentences) > 1:
        brief += "。" + sentences[1]
    if len(brief) > max_len:
        brief = brief[:max_len] + "…"
    return brief


def _extract_time_period(content: str) -> str:
    """从内容中提取年代范围。"""
    years = extract_years(content)
    if not years:
        return ""
    try:
        int_years = sorted(int(y) for y in years)
        if int_years[0] == int_years[-1]:
            return str(int_years[0])
        return f"{int_years[0]}-{int_years[-1]}"
    except ValueError:
        return years[0]


def _extract_key_phrases(content: str, top_n: int = 8) -> List[str]:
    """
    提取内容中的高频 2-3 字短语作为要点标识。
    用于跨章去重检测。
    """
    # 去掉标点，只保留中文
    clean = re.sub(r"[^\u4e00-\u9fa5]", "", content)
    if len(clean) < 4:
        return []

    counter: Counter[str] = Counter()
    for n in (3, 2):
        for i in range(len(clean) - n + 1):
            gram = clean[i:i + n]
            # 过滤纯虚词
            if gram in ("的是", "了的", "在了", "是的", "和的", "不是"):
                continue
            counter[gram] += 1

    # 取频次 ≥ 2 且排名前 top_n 的
    return [phrase for phrase, cnt in counter.most_common(top_n * 2)
            if cnt >= 2][:top_n]


def _char_ngrams(text: str, n: int) -> Set[str]:
    """字符级 n-gram 集合。"""
    clean = re.sub(r"\s+", "", text)
    if len(clean) < n:
        return set()
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}
