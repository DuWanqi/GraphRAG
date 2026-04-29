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
        
        # 对齐内容：已在原文中提及的实体
        if self.aligned_entities:
            aligned_parts.append("相关实体：")
            for entity in self.aligned_entities[:5]:
                name = entity.get("name", entity.get("title", ""))
                desc = entity.get("description", "")[:150]
                aligned_parts.append(f"- {name}: {desc}")
        
        # 新知识：原文未提及的实体
        if self.novel_entities:
            novel_parts.append("新增实体（原文未提及）：")
            for entity in self.novel_entities[:8]:
                name = entity.get("name", entity.get("title", ""))
                desc = entity.get("description", "")[:150]
                novel_parts.append(f"- {name}: {desc}")
        
        # 新知识：原文未提及的关系/事件
        if self.novel_relationships:
            novel_parts.append("\n新增事件关联（原文未提及）：")
            for rel in self.novel_relationships[:5]:
                source = rel.get("source", "")
                target = rel.get("target", "")
                desc = rel.get("description", "")[:150]
                novel_parts.append(f"- {source} → {target}: {desc}")
        
        # 新知识：背景片段
        if self.novel_snippets:
            novel_parts.append("\n新增背景信息（原文未提及）：")
            for snippet in self.novel_snippets[:3]:
                novel_parts.append(f"- {snippet[:200]}")
        
        return {
            "aligned_context": "\n".join(aligned_parts) if aligned_parts else "（无）",
            "novel_context": "\n".join(novel_parts) if novel_parts else "（无可用新知识）",
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
    for comm in retrieval_result.communities[:3]:
        summary = comm.get("summary", "")
        if summary and len(summary) > 50:
            # 检查是否包含原文未提及的新信息
            if not _has_significant_overlap(summary, memoir_text_normalized):
                brief.novel_snippets.append(summary[:300])
    
    # text_units 作为补充
    for text_unit in retrieval_result.text_units[:3]:
        if text_unit and len(text_unit) > 50:
            if not _has_significant_overlap(text_unit, memoir_text_normalized):
                brief.novel_snippets.append(text_unit[:300])
    
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
    
    策略：提取片段中的关键词，如果超过 50% 出现在原文中，则视为重叠
    """
    snippet_keywords = _extract_keywords(snippet, min_len=3)
    if not snippet_keywords:
        return False
    
    overlap_count = sum(1 for kw in snippet_keywords if _normalize_text(kw) in memoir_text_normalized)
    overlap_ratio = overlap_count / len(snippet_keywords)
    
    return overlap_ratio > 0.5


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
