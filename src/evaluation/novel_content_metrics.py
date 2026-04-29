"""
Novel Content Metrics - 新内容评估指标

评估生成文本中"新内容"的引入情况：
1. novel_content_ratio - 新内容引入率（使用了多少 RAG 提供的新知识）
2. novel_content_grounding - 新内容溯源率（新内容是否有 RAG 来源支撑，防幻觉）
3. expansion_depth - 扩展深度（shallow/moderate/deep）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from .metrics import MetricResult


@dataclass
class NovelContentAnalysis:
    """新内容分析结果"""
    novel_entities_used: List[str]          # 生成文本中使用的新实体
    novel_entities_available: List[str]     # RAG 提供的新实体
    new_facts_in_output: List[str]          # 生成文本中的新事实陈述
    grounded_facts: List[str]               # 有 RAG 来源支撑的新事实
    ungrounded_facts: List[str]             # 无 RAG 来源支撑的新事实（疑似幻觉）
    
    @property
    def novel_content_ratio(self) -> float:
        """新内容引入率"""
        if not self.novel_entities_available:
            return 0.0
        return len(self.novel_entities_used) / len(self.novel_entities_available)
    
    @property
    def novel_content_grounding(self) -> float:
        """新内容溯源率"""
        if not self.new_facts_in_output:
            return 1.0  # 没有新事实，默认为完全有据
        return len(self.grounded_facts) / len(self.new_facts_in_output)
    
    @property
    def expansion_depth(self) -> str:
        """扩展深度"""
        used_count = len(self.novel_entities_used)
        if used_count == 0:
            return "shallow"
        elif used_count <= 2:
            return "moderate"
        else:
            return "deep"


def analyze_novel_content(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,  # NovelContentBrief
) -> NovelContentAnalysis:
    """
    分析生成文本中的新内容使用情况
    
    Args:
        memoir_text: 回忆录原文
        generated_text: 生成的文本
        novel_content_brief: NovelContentBrief 对象（来自 extract_novel_content）
    
    Returns:
        NovelContentAnalysis: 分析结果
    """
    # 1. 检查哪些新实体被使用了
    novel_entities_available = novel_content_brief.novel_entity_names
    novel_entities_used = []
    
    for entity_name in novel_entities_available:
        if entity_name and _is_mentioned_in_text(entity_name, generated_text):
            novel_entities_used.append(entity_name)
    
    # 2. 提取生成文本中的新事实陈述（不在原文中的）
    new_facts_in_output = _extract_new_facts(memoir_text, generated_text)
    
    # 3. 检查新事实是否有 RAG 来源支撑
    grounded_facts = []
    ungrounded_facts = []
    
    # 构建 RAG 来源文本（用于匹配）
    rag_source_text = _build_rag_source_text(novel_content_brief)
    
    for fact in new_facts_in_output:
        if _is_grounded_in_rag(fact, rag_source_text, novel_entities_available):
            grounded_facts.append(fact)
        else:
            ungrounded_facts.append(fact)
    
    return NovelContentAnalysis(
        novel_entities_used=novel_entities_used,
        novel_entities_available=novel_entities_available,
        new_facts_in_output=new_facts_in_output,
        grounded_facts=grounded_facts,
        ungrounded_facts=ungrounded_facts,
    )


def novel_content_ratio_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
) -> MetricResult:
    """
    新内容引入率指标
    
    衡量生成文本使用了多少 RAG 提供的新知识
    """
    analysis = analyze_novel_content(memoir_text, generated_text, novel_content_brief)
    
    ratio = analysis.novel_content_ratio
    used = len(analysis.novel_entities_used)
    available = len(analysis.novel_entities_available)
    
    if available == 0:
        explanation = "RAG 未提供新实体"
    else:
        explanation = f"使用了 {used}/{available} 个新实体 ({ratio:.0%})"
        if used > 0:
            explanation += f"，包括：{', '.join(analysis.novel_entities_used[:3])}"
    
    return MetricResult(
        name="novel_content_ratio",
        value=ratio,
        max_value=1.0,
        explanation=explanation,
    )


def novel_content_grounding_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
) -> MetricResult:
    """
    新内容溯源率指标（防幻觉）
    
    衡量生成文本中的新事实是否有 RAG 来源支撑
    """
    analysis = analyze_novel_content(memoir_text, generated_text, novel_content_brief)
    
    grounding = analysis.novel_content_grounding
    grounded = len(analysis.grounded_facts)
    total = len(analysis.new_facts_in_output)
    
    if total == 0:
        explanation = "未检测到新事实陈述"
    else:
        explanation = f"{grounded}/{total} 个新事实有 RAG 来源支撑 ({grounding:.0%})"
        if analysis.ungrounded_facts:
            explanation += f"，{len(analysis.ungrounded_facts)} 个疑似幻觉"
    
    return MetricResult(
        name="novel_content_grounding",
        value=grounding,
        max_value=1.0,
        explanation=explanation,
    )


def expansion_depth_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
) -> MetricResult:
    """
    扩展深度指标
    
    衡量生成文本相对于输入的信息增量
    """
    analysis = analyze_novel_content(memoir_text, generated_text, novel_content_brief)
    
    depth = analysis.expansion_depth
    used = len(analysis.novel_entities_used)
    
    depth_scores = {
        "shallow": 0.3,
        "moderate": 0.7,
        "deep": 1.0,
    }
    
    depth_labels = {
        "shallow": "浅层（仅润色，未引入新知识）",
        "moderate": f"中等（引入 {used} 个新事实）",
        "deep": f"深度（引入 {used} 个新事实并有叙事整合）",
    }
    
    return MetricResult(
        name="expansion_depth",
        value=depth_scores[depth],
        max_value=1.0,
        explanation=depth_labels[depth],
    )


# ============================================================================
# 内部辅助函数
# ============================================================================

def _is_mentioned_in_text(entity_name: str, text: str) -> bool:
    """检查实体是否在文本中提及（模糊匹配）"""
    if not entity_name or not text:
        return False
    
    # 归一化
    entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity_name).upper()
    text_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', text).upper()
    
    # 精确匹配
    if entity_normalized in text_normalized:
        return True
    
    # 部分匹配（实体名的主要部分）
    if len(entity_normalized) >= 4:
        main_part = entity_normalized[:4]
        if main_part in text_normalized:
            return True
    
    return False


def _extract_new_facts(memoir_text: str, generated_text: str) -> List[str]:
    """
    提取生成文本中的新事实陈述（不在原文中的）
    
    策略：
    1. 提取生成文本中的实体名（人名、地名、组织名）
    2. 提取生成文本中的年份
    3. 过滤掉原文中已有的
    """
    new_facts = []
    
    # 提取实体（中文 2-10 字）
    generated_entities = set(re.findall(r'[一-龥]{2,10}', generated_text))
    memoir_entities = set(re.findall(r'[一-龥]{2,10}', memoir_text))
    
    new_entities = generated_entities - memoir_entities
    new_facts.extend(list(new_entities)[:20])  # 限制数量
    
    # 提取年份
    generated_years = set(re.findall(r'\b(19|20)\d{2}\b', generated_text))
    memoir_years = set(re.findall(r'\b(19|20)\d{2}\b', memoir_text))
    
    new_years = generated_years - memoir_years
    new_facts.extend(list(new_years))
    
    return new_facts


def _build_rag_source_text(novel_content_brief: Any) -> str:
    """构建 RAG 来源文本（用于匹配）"""
    parts = []
    
    # 实体描述
    for entity in novel_content_brief.novel_entities:
        name = entity.get("name", entity.get("title", ""))
        desc = entity.get("description", "")
        parts.append(f"{name} {desc}")
    
    # 关系描述
    for rel in novel_content_brief.novel_relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        desc = rel.get("description", "")
        parts.append(f"{source} {target} {desc}")
    
    # 背景片段
    parts.extend(novel_content_brief.novel_snippets)
    
    return " ".join(parts)


def _is_grounded_in_rag(fact: str, rag_source_text: str, novel_entities: List[str]) -> bool:
    """
    检查事实是否在 RAG 来源中有支撑
    
    策略：
    1. 如果事实是新实体名，检查是否在 novel_entities 中
    2. 如果事实是年份，检查是否在 RAG 来源文本中
    3. 如果事实是其他词，检查是否在 RAG 来源文本中（模糊匹配）
    """
    if not fact:
        return False
    
    # 归一化
    fact_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', fact).upper()
    rag_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', rag_source_text).upper()
    
    # 1. 检查是否在 novel_entities 中
    for entity in novel_entities:
        entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity).upper()
        if fact_normalized == entity_normalized or fact_normalized in entity_normalized:
            return True
    
    # 2. 检查是否在 RAG 来源文本中
    if fact_normalized in rag_normalized:
        return True
    
    # 3. 部分匹配（至少 3 个字符）
    if len(fact_normalized) >= 3:
        if fact_normalized[:3] in rag_normalized:
            return True
    
    return False
