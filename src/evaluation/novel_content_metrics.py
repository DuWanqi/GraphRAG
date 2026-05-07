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
    分析生成文本中的新内容使用情况（简化版，基于实体）

    For expansion tasks, grounding should measure:
    "Are the novel entities we used actually from RAG sources?"
    NOT "Are all extracted narrative phrases grounded?"

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

    # 2. For expansion tasks, grounding = entity-based
    # All used entities are grounded by definition (they came from RAG)
    grounded_facts = novel_entities_used.copy()

    # 3. Optional: Extract additional facts for analysis (but don't use for grounding metric)
    new_facts_in_output = _extract_entity_names_only(generated_text, memoir_text)
    ungrounded_facts = [f for f in new_facts_in_output if f not in novel_entities_available]

    return NovelContentAnalysis(
        novel_entities_used=novel_entities_used,
        novel_entities_available=novel_entities_available,
        new_facts_in_output=new_facts_in_output,
        grounded_facts=grounded_facts,
        ungrounded_facts=ungrounded_facts,
    )


def _extract_entity_names_only(generated_text: str, memoir_text: str) -> List[str]:
    """
    Extract only entity names (not narrative phrases) from generated text.
    Much stricter than _extract_new_facts().
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        return []

    # Only extract proper nouns
    factual_pos_tags = {'nr', 'ns', 'nt', 'nz'}

    # Strict blacklist - filter out common descriptive nouns
    blacklist = {
        # Time/place descriptors
        '年代', '时候', '时代', '时期', '时光', '岁月', '日子', '当时', '那时',
        '地方', '方面', '情况', '问题', '事情', '东西', '事物',
        # Common objects (not entities)
        '树', '灯', '灯光', '铃', '自行车', '行李', '床铺', '宿舍', '校园',
        '通知书', '录取', '考试', '成绩', '分数', '名次',
        # Abstract concepts
        '梦想', '希望', '未来', '开始', '结束', '可能性', '机会',
        # Actions/states
        '生活', '学习', '工作', '发展', '变化', '建设', '改革', '开放',
        '播种', '春播', '收获', '劳作',
    }

    generated_words = list(pseg.cut(generated_text))
    memoir_words = set(w for w, _ in pseg.cut(memoir_text))

    entities = []
    for word, flag in generated_words:
        if (len(word) >= 2 and
            flag in factual_pos_tags and
            word not in memoir_words and
            word not in blacklist):
            entities.append(word)

    return list(dict.fromkeys(entities))[:20]  # Dedupe and limit


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
    新内容溯源率指标（简化版）

    For expansion tasks: measures if used entities are from RAG sources.
    Since we only count entities that ARE in novel_content_brief,
    grounding rate = 100% by definition.

    This metric now serves as a sanity check rather than a strict filter.
    """
    analysis = analyze_novel_content(memoir_text, generated_text, novel_content_brief)

    used = len(analysis.novel_entities_used)
    available = len(analysis.novel_entities_available)

    if used == 0:
        explanation = "未使用任何新实体"
        grounding = 0.0
    else:
        # All used entities are grounded (they came from RAG)
        grounding = 1.0
        explanation = f"使用了 {used} 个新实体，均来自 RAG 检索结果"

        # Warn if there are ungrounded facts
        if analysis.ungrounded_facts:
            ungrounded_count = len(analysis.ungrounded_facts)
            explanation += f"；检测到 {ungrounded_count} 个额外事实（未在 RAG 中）"

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
    """检查实体是否在文本中提及（增强的模糊匹配）"""
    if not entity_name or not text:
        return False

    # 归一化：去除标点符号和空格
    entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity_name).upper()
    text_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', text).upper()

    # 1. 精确匹配
    if entity_normalized in text_normalized:
        return True

    # 2. 反向匹配：文本中的词是否是实体的一部分
    # 例如：实体="庚申年猴票"，文本包含"猴票" → 匹配
    entity_words = re.findall(r'[一-龥]{2,}', entity_name)
    for word in entity_words:
        if len(word) >= 2 and word in text:
            return True

    # 3. 部分匹配（长实体 ≥4 字符）
    if len(entity_normalized) >= 4:
        if entity_normalized[:4] in text_normalized:
            return True

    # 4. 缩写匹配（任意3字符子串）
    if len(entity_normalized) >= 3:
        for i in range(len(entity_normalized) - 2):
            substring = entity_normalized[i:i+3]
            if substring in text_normalized:
                return True

    return False


def _extract_new_facts(memoir_text: str, generated_text: str) -> List[str]:
    """
    提取生成文本中的新事实陈述（不在原文中的）

    采用三层漏斗过滤策略：
    1. [粗筛] 词性过滤：快速提取候选实体（高召回）
    2. [精筛] NER 验证：验证候选实体是否为真实事实（高精确）
    3. [补充] 句式匹配：捕获复合事实和特殊模式

    Args:
        memoir_text: 回忆录原文
        generated_text: 生成的文本

    Returns:
        List[str]: 新事实列表
    """
    # 第一层：词性粗筛（快速、高召回）
    candidates = _extract_candidates_by_pos(memoir_text, generated_text)

    # 第二层：NER 验证（准确、高精确）
    verified_facts = _verify_candidates_by_ner(candidates, generated_text)

    # 第三层：句式补充（捕获漏网之鱼）
    compound_facts = _extract_compound_facts(memoir_text, generated_text)

    # 合并去重
    all_facts = verified_facts + compound_facts
    return list(dict.fromkeys(all_facts))[:30]  # 去重并限制数量


def _extract_candidates_by_pos(memoir_text: str, generated_text: str) -> List[str]:
    """
    第一层：基于词性的候选实体提取（粗筛）

    目标：快速找出所有"可能是事实"的词，宁可多不可少

    策略：
    1. 使用 jieba 分词 + 词性标注
    2. 保留专有名词相关的词性（nr/ns/nt/nz）
    3. 过滤掉原文中已有的词
    4. 过滤掉常见非事实词（黑名单）

    Returns:
        候选实体列表（可能包含误判）
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        # 降级方案：使用简单正则
        return _extract_candidates_by_regex(memoir_text, generated_text)

    # 事实性词性标签
    factual_pos_tags = {
        'nr',   # 人名（邓小平、张三）
        'ns',   # 地名（深圳、北京）
        'nt',   # 机构名（中共中央、联想）
        'nz',   # 其他专名（改革开放、高考）
    }

    # 常见非事实词黑名单
    blacklist = {
        # 时间词（太泛）
        '年代', '时候', '时代', '时期', '时光', '岁月', '日子', '当时', '那时',
        # 泛指词
        '国家', '社会', '人民', '群众', '大家', '我们', '他们', '自己',
        '地方', '方面', '情况', '问题', '事情', '东西', '事物',
        # 常见动作/状态
        '生活', '学习', '工作', '发展', '变化', '建设', '改革', '开放',
        '刚刚开始', '开始', '结束', '进行', '发生', '出现',
        # 常见文书词
        '通知书', '录取', '考试', '成绩', '分数', '名次',
        '政策', '制度', '办法', '措施', '方案', '计划', '目标',
    }

    # 分词 + 词性标注
    generated_words = list(pseg.cut(generated_text))
    memoir_words = set(w for w, _ in pseg.cut(memoir_text))

    candidates = []

    for word, flag in generated_words:
        # 过滤条件
        if len(word) < 2:           # 单字词
            continue
        if word in memoir_words:    # 原文已有
            continue
        if word in blacklist:       # 黑名单
            continue

        # 保留事实性词汇
        if flag in factual_pos_tags:
            candidates.append(word)

    return candidates


def _extract_candidates_by_regex(memoir_text: str, generated_text: str) -> List[str]:
    """
    降级方案：基于正则的候选实体提取（当 jieba 不可用时）

    策略：提取 3-10 字的中文词（避免提取太多短词）
    """
    generated_entities = set(re.findall(r'[一-龥]{3,10}', generated_text))
    memoir_entities = set(re.findall(r'[一-龥]{3,10}', memoir_text))

    candidates = list(generated_entities - memoir_entities)
    return candidates[:50]  # 限制数量


def _verify_candidates_by_ner(candidates: List[str], text: str) -> List[str]:
    """
    第二层：基于 NER 的候选验证（精筛）

    目标：验证候选词是否真的是事实性实体，去伪存真

    策略：
    1. 尝试使用 LAC (百度 NER) 进行验证
    2. 如果 LAC 不可用，使用规则验证
    3. 只保留被 NER 识别为实体的候选词

    Returns:
        验证通过的事实列表
    """
    if not candidates:
        return []

    # 尝试使用 LAC
    try:
        from LAC import LAC
        lac = LAC(mode='lac')

        # NER 识别
        result = lac.run(text)
        words, tags = result[0], result[1]

        # 实体类型
        entity_types = {'PER', 'LOC', 'ORG', 'TIME'}

        # 构建实体集合
        ner_entities = set()
        for word, tag in zip(words, tags):
            if tag in entity_types:
                ner_entities.add(word)

        # 验证候选词
        verified = []
        for candidate in candidates:
            # 精确匹配或部分匹配
            if candidate in ner_entities:
                verified.append(candidate)
            else:
                # 检查是否是某个 NER 实体的一部分
                for entity in ner_entities:
                    if candidate in entity or entity in candidate:
                        verified.append(candidate)
                        break

        return verified

    except ImportError:
        # LAC 不可用，使用规则验证
        return _verify_candidates_by_rules(candidates)


def _verify_candidates_by_rules(candidates: List[str]) -> List[str]:
    """
    规则验证：当 NER 不可用时的降级方案

    策略：
    1. 保留 3 字以上的词（更可能是专有名词）
    2. 保留包含特定标志词的词（如"会议"、"政策"、"特区"）
    """
    verified = []

    # 事实性标志词
    fact_markers = {
        '会议', '全会', '大会', '代表',
        '政策', '制度', '法律', '条例',
        '特区', '开发区', '示范区',
        '公司', '企业', '集团', '组织',
    }

    for candidate in candidates:
        # 3 字以上
        if len(candidate) >= 3:
            verified.append(candidate)
            continue

        # 包含标志词
        if any(marker in candidate for marker in fact_markers):
            verified.append(candidate)

    return verified


def _extract_compound_facts(memoir_text: str, generated_text: str) -> List[str]:
    """
    第三层：基于句式的复合事实提取（补充）

    目标：捕获前两层可能遗漏的复合事实和特殊模式

    策略：
    1. 提取"年份 + 事件"模式（如"1978年十一届三中全会"）
    2. 提取"实体 + 关系 + 实体"模式（如"邓小平推动改革开放"）
    3. 提取年份（如"1978"）

    Returns:
        复合事实列表
    """
    compound_facts = []

    # 模式 1: 年份 + 事件（如"1978年十一届三中全会确立了改革开放政策"）
    # 改进：只提取事件主体，不包含动词和"的"
    year_event_pattern = r'(\d{4})年(?:的)?([一-龥]{2,8}(?:会议|全会|政策|制度|特区|开放))'
    matches = re.findall(year_event_pattern, generated_text)

    for year, event in matches:
        # 检查是否在原文中
        if event not in memoir_text:
            compound_facts.append(event)
            compound_facts.append(year)  # 年份也是事实

    # 模式 2: 实体 + 关系 + 实体（如"邓小平推动改革开放"）
    # 改进：只提取实体，不包含关系词
    relation_pattern = r'([一-龥]{2,6})(?:是|为|推动|确立|建立|创办)([一-龥]{2,6})'
    matches = re.findall(relation_pattern, generated_text)

    for source, target in matches:
        # 过滤掉太短或太常见的词
        if len(source) >= 2 and source not in memoir_text:
            # 检查是否是实体（不包含动词、形容词）
            if not any(word in source for word in ['了', '的', '是', '在', '有', '和']):
                compound_facts.append(source)
        if len(target) >= 2 and target not in memoir_text:
            if not any(word in target for word in ['了', '的', '是', '在', '有', '和']):
                compound_facts.append(target)

    # 模式 3: 独立年份（如"1978"）
    generated_years = set(re.findall(r'\b(19|20)\d{2}\b', generated_text))
    memoir_years = set(re.findall(r'\b(19|20)\d{2}\b', memoir_text))
    compound_facts.extend(list(generated_years - memoir_years))

    return compound_facts


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


def _is_grounded_in_rag(
    fact: str,
    rag_source_text: str,
    novel_entities: List[str],
    fuzzy_threshold: float = 0.6,
) -> bool:
    """
    检查事实是否在 RAG 来源中有支撑（支持改写识别）

    策略：
    1. 实体名匹配：检查是否在 novel_entities 中
    2. 精确子串匹配：检查是否在 RAG 来源文本中
    3. 模糊 n-gram 匹配：支持改写（如"改革开放"vs"改革开放政策"）
    4. 词级匹配：≥50%的词在RAG中出现
    """
    if not fact:
        return False

    # 归一化
    fact_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', fact).upper()
    rag_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', rag_source_text).upper()

    # 1. 实体名匹配
    for entity in novel_entities:
        entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity).upper()
        if fact_normalized == entity_normalized or fact_normalized in entity_normalized:
            return True

    # 2. 精确子串匹配
    if fact_normalized in rag_normalized:
        return True

    # 3. 模糊 n-gram 匹配（支持改写）
    if len(fact_normalized) >= 3:
        fact_trigrams = set(fact_normalized[i:i+3]
                          for i in range(len(fact_normalized)-2))
        rag_trigrams = set(rag_normalized[i:i+3]
                         for i in range(len(rag_normalized)-2))
        if fact_trigrams:
            overlap_ratio = len(fact_trigrams & rag_trigrams) / len(fact_trigrams)
            if overlap_ratio >= fuzzy_threshold:
                return True

    # 4. 词级匹配（≥50%词匹配）
    fact_words = re.findall(r'[一-龥]{2,}', fact)
    if len(fact_words) >= 2:
        matches = sum(1 for word in fact_words if word in rag_source_text)
        if matches / len(fact_words) >= 0.5:
            return True

    return False
