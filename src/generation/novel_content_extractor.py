"""
Novel Content Extractor - 新内容提取器

从 RAG 检索结果中区分"输入已有的对齐内容"和"输入未提及的新知识"。
纯规则实现，无 LLM 依赖。

用途：
1. 在生成时，将 RAG 内容分类后注入 prompt，明确标注哪些是新知识
2. 在评估时，作为 ground truth 判断生成内容是否引入了新知识
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Set

from ..retrieval import RetrievalResult


def _localize_entity_name(name: str, description: str) -> str:
    """
    若实体名为英文/拼音，尝试从描述中提取对应的中文名称。
    返回中文名（如找到）或原名。
    """
    if not name:
        return name
    # 已经是中文名（含至少一个中文字符），直接返回
    if re.search(r'[\u4e00-\u9fff]', name):
        return name
    # 从描述前 300 字符中提取括号内的中文名
    # 常见模式: "Deng Xiaoping ... (邓小平)" 或 "... also known as 美国 ..."
    head = description[:300] if description else ""
    # 模式1: 括号内中文 (中文名) 或 （中文名）
    m = re.search(r'[（(]([一-龥]{2,10})[）)]', head)
    if m:
        return m.group(1)
    # 模式2: "also known as" / "commonly known as" 后的中文
    m = re.search(r'(?:also|commonly)\s+known\s+as\s+["\']?([一-龥]{2,10})', head)
    if m:
        return m.group(1)
    # 模式3: 描述开头直接有"中文名 (English)" — 如 "邓小平（Deng Xiaoping）"
    m = re.search(r'^([一-龥]{2,10})', head)
    if m:
        return m.group(1)
    # 无法提取，返回原名
    return name


def _truncate_for_prompt(text: str, max_chars: int = 200) -> str:
    """将 RAG 描述截断到可控长度；兼容英文（仅用中文「。」分割会失效）。"""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    min_keep = max(24, max_chars // 5)
    best_end = -1
    for sep in ("。", "！", "？", ". ", ".\n", "\n\n", "\n"):
        j = window.rfind(sep)
        if j >= min_keep:
            if sep == ". ":
                best_end = max(best_end, j + 1)
            else:
                best_end = max(best_end, j + len(sep))
    if best_end > 0:
        return text[:best_end].rstrip()
    return window.rstrip() + "…"


@dataclass
class NovelContentBrief:
    """
    RAG 检索结果的分类摘要
    
    Attributes:
        novel_entities: RAG 实体中 memoir_text 未提及的（新知识）
        novel_relationships: RAG 关系中 memoir_text 未提及的（新知识）
        novel_snippets: 社区报告/text_units 中的新信息片段
        aligned_entities: RAG 实体中 memoir_text 已提及的（对齐内容）
        aligned_relationships: RAG 关系中 memoir_text 已提及的（对齐内容）
        summary: 一句话概括有哪些新知识可用
    """
    novel_entities: List[Dict[str, Any]] = field(default_factory=list)
    novel_relationships: List[Dict[str, Any]] = field(default_factory=list)
    novel_snippets: List[str] = field(default_factory=list)
    aligned_entities: List[Dict[str, Any]] = field(default_factory=list)
    aligned_relationships: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    
    @property
    def has_novel_content(self) -> bool:
        """是否有新知识可用"""
        return bool(self.novel_entities or self.novel_relationships or self.novel_snippets)
    
    @property
    def novel_entity_names(self) -> List[str]:
        """新实体名称列表（用于评估）"""
        return [e.get("name", e.get("title", "")) for e in self.novel_entities]
    
    @property
    def aligned_entity_names(self) -> List[str]:
        """对齐实体名称列表"""
        return [e.get("name", e.get("title", "")) for e in self.aligned_entities]
    
    def format_for_prompt(self) -> Dict[str, str]:
        """
        格式化为 prompt 注入用的两个区块

        Returns:
            {"aligned_context": str, "novel_context": str}
        """
        aligned_parts = []
        novel_parts = []
        entity_names: List[str] = []

        # 对齐内容：已在原文中提及的实体
        if self.aligned_entities:
            aligned_parts.append("相关实体：")
            for entity in self.aligned_entities[:5]:
                name = entity.get("name", entity.get("title", ""))
                desc = entity.get("description", "")[:150]
                aligned_parts.append(f"- {name}: {desc}")

        # 新知识：原文未提及的实体（白名单式格式）
        if self.novel_entities:
            novel_parts.append("【可用的新知识实体白名单】")
            novel_parts.append("以下是检索到的、原文未提及的实体。你必须使用其中文名称（而非英文原名）融入叙事，不得添加其他实体。\n")

            # 先列出实体名称清单（中文化）
            for e in self.novel_entities[:8]:
                raw_name = e.get("name", e.get("title", ""))
                raw_desc = e.get("description", "") or ""
                entity_names.append(_localize_entity_name(raw_name, raw_desc))
            novel_parts.append("✓ 允许使用的实体名称（中文）：" + "、".join(entity_names))
            novel_parts.append("")

            # 再列出详细描述
            for i, entity in enumerate(self.novel_entities[:8], 1):
                raw_name = entity.get("name", entity.get("title", ""))
                raw_desc = entity.get("description", "") or ""
                display_name = _localize_entity_name(raw_name, raw_desc)
                desc = _truncate_for_prompt(raw_desc, 200)

                novel_parts.append(f"{i}. {display_name} ({entity.get('type', 'ENTITY')})")
                novel_parts.append(f"   {desc}")

        # 新知识：原文未提及的关系/事件
        if self.novel_relationships:
            if not self.novel_entities:
                novel_parts.append("【可用的新知识白名单】")
                novel_parts.append("以下是检索到的关系和事件。你只能使用这些内容，不得添加其他实体或事实。\n")
            else:
                novel_parts.append("\n【可用的关系和事件】")

            start_idx = len(self.novel_entities[:8]) + 1
            for i, rel in enumerate(self.novel_relationships[:5], start_idx):
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_desc = rel.get("description", "") or ""
                desc = _truncate_for_prompt(rel_desc, 200)
                # 中文化 source/target
                source_disp = _localize_entity_name(source, rel_desc)
                target_disp = _localize_entity_name(target, rel_desc)

                novel_parts.append(f"{i}. {source_disp} → {target_disp}")
                novel_parts.append(f"   {desc}")

        # 新知识：背景片段
        if self.novel_snippets:
            novel_parts.append("\n补充背景（可选择性引用）：")
            for snippet in self.novel_snippets[:8]:  # 增加到 8 个片段
                novel_parts.append(f"- {snippet[:400]}")  # 增加到 400 字符

        # 添加使用规则
        if self.novel_entities or self.novel_relationships:
            novel_parts.append("\n" + "="*60)
            novel_parts.append("【严格使用规则 - 必须遵守】")
            novel_parts.append("="*60)
            novel_parts.append("\n✓ 允许的操作：")
            novel_parts.append("  • 从上述白名单中选择 2-4 个与叙事最相关的实体或事件（至少使用 2 个）")
            novel_parts.append("  • 改写上述内容的表述方式，调整语序，使其自然融入叙事")
            novel_parts.append("  • 将这些背景知识作为人物对话、环境描写或叙事者见闻的一部分")

            novel_parts.append("\n✗ 严格禁止的操作（违反将导致生成失败）：")
            novel_parts.append("  • 添加白名单中未列出的任何实体（人名、地名、机构名、事件名）")
            novel_parts.append("  • 添加上述内容中未提及的具体数据、数字、政策名称、时间节点")
            novel_parts.append("  • 推断或编造上述内容中未提及的因果关系、影响或评价")
            novel_parts.append("  • 使用你的训练知识中的历史事实（如果未在上述白名单中提供）")

            novel_parts.append("\n⚠️  重要提醒：")
            novel_parts.append("  如果你在生成中提到了任何实体（人名、地名、机构、事件），")
            novel_parts.append("  该实体必须出现在上述白名单中，或者出现在原文中。")
            novel_parts.append("  不在白名单中的实体一律不得使用。")
            novel_parts.append("")
            novel_parts.append("  【语言要求 - 严格禁止英文】")
            novel_parts.append("  生成的文本必须是纯中文。所有实体名称必须使用中文表达。")
            novel_parts.append("  严格禁止在生成文本中出现英文实体名（如 DENG XIAOPING、SHENZHEN CITY 等）。")
            novel_parts.append("  必须使用对应的中文名称（如 邓小平、深圳 等）。")

        aligned_ctx = "\n".join(aligned_parts) if aligned_parts else "（无）"
        novel_ctx = "\n".join(novel_parts) if novel_parts else "（无可用新知识）"

        return {
            "aligned_context": aligned_ctx,
            "novel_context": novel_ctx,
        }


def extract_novel_content(
    memoir_text: str,
    retrieval_result: RetrievalResult,
    *,
    fuzzy_match: bool = True,
) -> NovelContentBrief:
    """
    从 RAG 检索结果中提取新内容
    
    Args:
        memoir_text: 回忆录原文
        retrieval_result: RAG 检索结果
        fuzzy_match: 是否使用模糊匹配（考虑中英文变体、简称等）
    
    Returns:
        NovelContentBrief: 分类后的内容摘要
    """
    # 预处理：提取 memoir_text 中的关键词（用于匹配）
    memoir_keywords = _extract_keywords(memoir_text)
    memoir_text_normalized = _normalize_text(memoir_text)
    
    brief = NovelContentBrief()
    
    # 1. 分类实体
    for entity in retrieval_result.entities:
        name = entity.get("name", entity.get("title", ""))
        if not name:
            continue
        
        is_aligned = _is_mentioned(name, memoir_text_normalized, memoir_keywords, fuzzy_match)
        
        if is_aligned:
            brief.aligned_entities.append(entity)
        else:
            brief.novel_entities.append(entity)
    
    # 2. 分类关系
    for rel in retrieval_result.relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        desc = rel.get("description", "")
        
        # 如果 source 或 target 在原文中提及，则视为对齐内容
        source_mentioned = _is_mentioned(source, memoir_text_normalized, memoir_keywords, fuzzy_match)
        target_mentioned = _is_mentioned(target, memoir_text_normalized, memoir_keywords, fuzzy_match)
        
        if source_mentioned or target_mentioned:
            brief.aligned_relationships.append(rel)
        else:
            # 检查描述中是否包含原文关键词
            desc_related = any(kw in desc for kw in memoir_keywords if len(kw) >= 2)
            if desc_related:
                brief.aligned_relationships.append(rel)
            else:
                brief.novel_relationships.append(rel)
    
    # 3. 提取新信息片段（从社区报告和 text_units）
    # 社区报告通常包含更高层次的背景信息
    for comm in retrieval_result.communities[:5]:  # 增加到 5 个
        summary = comm.get("summary", "")
        if summary and len(summary) > 50:
            # 检查是否包含原文未提及的新信息
            if not _has_significant_overlap(summary, memoir_text_normalized):
                brief.novel_snippets.append(summary[:400])  # 增加到 400 字
    
    # text_units 作为补充（增加到 10 个）
    for text_unit in retrieval_result.text_units[:10]:
        if text_unit and len(text_unit) > 50:
            if not _has_significant_overlap(text_unit, memoir_text_normalized):
                brief.novel_snippets.append(text_unit[:400])  # 增加到 400 字
    
    # 4. 生成摘要
    brief.summary = _generate_summary(brief)
    
    return brief


def _normalize_text(text: str) -> str:
    """文本归一化：去除空格、标点，转大写（用于匹配）"""
    # 保留中文、英文、数字
    text = re.sub(r'[^一-龥a-zA-Z0-9]', '', text)
    return text.upper()


def _extract_keywords(text: str, min_len: int = 2) -> Set[str]:
    """
    提取文本中的关键词（用于快速匹配）
    
    包括：
    - 中文词（2+ 字）
    - 英文词（3+ 字母）
    - 年份（4 位数字）
    """
    keywords = set()
    
    # 中文词（2-10 字）
    chinese_words = re.findall(r'[一-龥]{2,10}', text)
    keywords.update(w for w in chinese_words if len(w) >= min_len)
    
    # 英文词（3+ 字母）
    english_words = re.findall(r'[A-Za-z]{3,}', text)
    keywords.update(w.upper() for w in english_words)
    
    # 年份
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    keywords.update(years)
    
    return keywords


def _is_mentioned(
    entity_name: str,
    memoir_text_normalized: str,
    memoir_keywords: Set[str],
    fuzzy_match: bool,
) -> bool:
    """
    判断实体是否在原文中提及
    
    匹配策略：
    1. 精确匹配：实体名直接出现在原文中
    2. 模糊匹配（可选）：
       - 中英文变体（如"深圳" vs "SHENZHEN"）
       - 简称（如"改革开放" vs "改革"）
       - 部分匹配（实体名的主要部分出现在原文中）
    """
    if not entity_name:
        return False
    
    entity_normalized = _normalize_text(entity_name)
    
    # 1. 精确匹配
    if entity_normalized in memoir_text_normalized:
        return True
    
    if not fuzzy_match:
        return False
    
    # 2. 模糊匹配
    
    # 2a. 关键词匹配（实体名的任何关键词出现在原文中）
    entity_keywords = _extract_keywords(entity_name, min_len=2)
    if entity_keywords & memoir_keywords:
        return True
    
    # 2b. 部分匹配（实体名的主要部分）
    # 例如："中华人民共和国" 的主要部分是 "中华" "人民" "共和国"
    if len(entity_normalized) >= 4:
        # 取前 4 个字符作为主要部分
        main_part = entity_normalized[:4]
        if main_part in memoir_text_normalized:
            return True
    
    return False


def _has_significant_overlap(snippet: str, memoir_text_normalized: str) -> bool:
    """
    判断片段是否与原文有显著重叠（用于过滤重复内容）
    
    策略：提取片段中的关键词，如果超过 80% 出现在原文中，则视为重叠
    （降低阈值，允许更多历史背景信息通过）
    """
    snippet_keywords = _extract_keywords(snippet, min_len=3)
    if not snippet_keywords:
        return False
    
    overlap_count = sum(1 for kw in snippet_keywords if _normalize_text(kw) in memoir_text_normalized)
    overlap_ratio = overlap_count / len(snippet_keywords)
    
    # 提高阈值到 80%，允许更多历史背景通过
    return overlap_ratio > 0.8


def _generate_summary(brief: NovelContentBrief) -> str:
    """生成一句话摘要"""
    parts = []
    
    if brief.novel_entities:
        parts.append(f"{len(brief.novel_entities)}个新实体")
    if brief.novel_relationships:
        parts.append(f"{len(brief.novel_relationships)}个新关系")
    if brief.novel_snippets:
        parts.append(f"{len(brief.novel_snippets)}段新背景")
    
    if not parts:
        return "无可用新知识"
    
    return f"可用新知识：{', '.join(parts)}"
