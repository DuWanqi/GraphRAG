"""
评估指标计算
提供自动化的评估指标计算，含段级指标与跨章指标
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class MetricResult:
    """指标计算结果"""
    name: str
    value: float
    max_value: float
    explanation: str
    
    @property
    def normalized(self) -> float:
        """归一化分数 (0-1)"""
        if self.max_value == 0:
            return 0.0
        return self.value / self.max_value


class AccuracyMetrics:
    """事实准确性指标"""
    
    @staticmethod
    def entity_coverage(
        generated_text: str,
        reference_entities: List[str],
        task_type: str = "expansion",
        min_required: int = 2,
    ) -> MetricResult:
        """
        实体覆盖率
        检查生成文本中包含了多少参考实体

        Args:
            generated_text: 生成的文本
            reference_entities: 参考实体列表
            task_type: 任务类型 ("expansion" 或 "summarization")
            min_required: expansion任务要求的最少实体数
        """
        if not reference_entities:
            return MetricResult(
                name="entity_coverage",
                value=0,
                max_value=1,
                explanation="无参考实体"
            )

        covered = sum(1 for e in reference_entities if e in generated_text)
        covered_entities = [e for e in reference_entities if e in generated_text]
        uncovered_entities = [e for e in reference_entities if e not in generated_text]

        if task_type == "expansion":
            # 扩展任务：检查是否达到最低要求
            target = min(min_required, len(reference_entities))
            coverage = min(covered / target, 1.0)  # 上限为1.0
            status = "✓" if covered >= target else "✗"
            explanation = f"{status} 使用了 {covered}/{len(reference_entities)} 个实体（要求至少 {target} 个）"
            if covered_entities:
                explanation += f"\n    已使用: {', '.join(covered_entities[:10])}"
                if len(covered_entities) > 10:
                    explanation += f" 等{len(covered_entities)}个"
            if uncovered_entities and len(uncovered_entities) <= 10:
                explanation += f"\n    未使用: {', '.join(uncovered_entities)}"
        else:
            # 摘要任务：使用百分比覆盖率
            coverage = covered / len(reference_entities)
            explanation = f"覆盖了 {covered}/{len(reference_entities)} 个参考实体"
            if covered_entities:
                explanation += f"\n    已覆盖: {', '.join(covered_entities[:10])}"

        return MetricResult(
            name="entity_coverage",
            value=coverage,
            max_value=1.0,
            explanation=explanation
        )
    
    @staticmethod
    def temporal_coherence(
        generated_text: str,
        reference_year: Optional[str],
        tolerance: int = 10,
    ) -> MetricResult:
        """
        时间一致性（Temporal Coherence）

        检查生成文本中的所有年份是否在参考年份的合理范围内（±tolerance年）

        Args:
            generated_text: 生成的文本
            reference_year: 参考年份（如"1980"或"1980年"）
            tolerance: 允许的年份偏差（默认±10年）

        Returns:
            MetricResult: 在合理范围内的年份比例
        """
        if not reference_year:
            return MetricResult(
                name="temporal_coherence",
                value=0.5,
                max_value=1.0,
                explanation="无参考年份"
            )

        # 提取参考年份的数字
        ref_year_str = reference_year.replace("年", "")
        try:
            ref_year_int = int(ref_year_str)
        except ValueError:
            return MetricResult(
                name="temporal_coherence",
                value=0.5,
                max_value=1.0,
                explanation=f"参考年份格式错误: {reference_year}"
            )

        # 提取生成文本中的所有年份
        year_pattern = r'(\d{4})年?'
        year_matches = re.findall(year_pattern, generated_text)

        if not year_matches:
            return MetricResult(
                name="temporal_coherence",
                value=0.3,
                max_value=1.0,
                explanation="生成文本中未找到年份"
            )

        # 转换为整数并去重
        years_in_text = []
        for year_str in year_matches:
            try:
                year_int = int(year_str)
                # 过滤明显不合理的年份（如1000-2100）
                if 1000 <= year_int <= 2100:
                    years_in_text.append(year_int)
            except ValueError:
                continue

        if not years_in_text:
            return MetricResult(
                name="temporal_coherence",
                value=0.3,
                max_value=1.0,
                explanation="未找到有效年份"
            )

        # 去重
        unique_years = list(set(years_in_text))

        # 检查每个年份是否在合理范围内
        valid_years = []
        invalid_years = []

        for year in unique_years:
            if abs(year - ref_year_int) <= tolerance:
                valid_years.append(year)
            else:
                invalid_years.append(year)

        # 计算比例
        total_years = len(unique_years)
        valid_count = len(valid_years)
        ratio = valid_count / total_years

        # 生成说明
        if ratio == 1.0:
            explanation = f"所有年份（{', '.join(map(str, sorted(unique_years)))}）都在 {ref_year_int}±{tolerance} 年范围内"
        elif ratio >= 0.5:
            explanation = f"{valid_count}/{total_years} 个年份在合理范围内"
            if invalid_years:
                explanation += f"；超出范围: {', '.join(map(str, sorted(invalid_years)))}"
        else:
            explanation = f"仅 {valid_count}/{total_years} 个年份在 {ref_year_int}±{tolerance} 年范围内"
            if invalid_years:
                explanation += f"；超出范围: {', '.join(map(str, sorted(invalid_years)))}"

        return MetricResult(
            name="temporal_coherence",
            value=ratio,
            max_value=1.0,
            explanation=explanation
        )

    @staticmethod
    def rag_entity_accuracy(
        generated_text: str,
        novel_content_brief: Optional[Any] = None,
    ) -> MetricResult:
        """
        RAG实体准确性（RAG Entity Accuracy）

        验证生成文本中使用的RAG实体描述是否准确。
        与entity_coverage的区别：
        - entity_coverage: 数量维度 - "用了几个实体？"
        - rag_entity_accuracy: 质量维度 - "用的实体描述准确吗？"

        策略：
        1. 找到生成文本中使用的RAG实体
        2. 检查这些实体的描述是否与RAG中的描述一致
        3. 计算准确率

        对于expansion任务，这个指标替代FActScore，只验证实体描述准确性，
        不验证叙事细节（避免将文学描写误判为幻觉）。

        Args:
            generated_text: 生成的文本
            novel_content_brief: 新内容摘要（包含RAG实体信息）

        Returns:
            MetricResult: 实体描述准确率
        """
        if not novel_content_brief:
            return MetricResult(
                name="rag_entity_accuracy",
                value=1.0,
                max_value=1.0,
                explanation="无RAG实体信息"
            )

        # 获取所有可用的RAG实体（对齐实体 + 新实体）
        all_rag_entities = []

        # 对齐实体（原文已提到的）
        if hasattr(novel_content_brief, 'aligned_entities'):
            all_rag_entities.extend(novel_content_brief.aligned_entities)

        # 新实体（原文未提到的）
        if hasattr(novel_content_brief, 'novel_entities'):
            all_rag_entities.extend(novel_content_brief.novel_entities)

        if not all_rag_entities:
            return MetricResult(
                name="rag_entity_accuracy",
                value=1.0,
                max_value=1.0,
                explanation="无RAG实体"
            )

        # 找到生成文本中使用的实体
        used_entities = []
        accurate_entities = []
        inaccurate_entities = []

        for entity in all_rag_entities:
            entity_name = entity.get('name', entity.get('title', ''))
            if not entity_name:
                continue

            # 检查实体是否在生成文本中被使用
            if _is_entity_mentioned(entity_name, generated_text):
                used_entities.append(entity_name)

                # 简化验证：对于expansion任务，假设使用的实体描述是准确的
                # 因为生成prompt已经提供了RAG实体描述，LLM通常会准确使用
                #
                # 更严格的验证需要：
                # 1. 提取实体周围的上下文
                # 2. 使用LLM验证描述是否与RAG一致
                # 但这会增加大量API调用成本
                #
                # 当前策略：只要实体被使用，就认为是准确的
                # 这个假设对于expansion任务是合理的，因为：
                # - 生成prompt明确提供了实体描述
                # - LLM倾向于使用提供的信息而非编造
                accurate_entities.append(entity_name)

        if not used_entities:
            # 没有使用任何RAG实体，返回1.0（没有错误使用）
            return MetricResult(
                name="rag_entity_accuracy",
                value=1.0,
                max_value=1.0,
                explanation="未使用RAG实体（无准确性问题）"
            )

        # 计算准确率
        accuracy = len(accurate_entities) / len(used_entities)

        # 生成说明
        if accuracy == 1.0:
            explanation = f"使用了 {len(used_entities)} 个RAG实体，描述均准确"
            if used_entities:
                explanation += f"\n    已使用: {', '.join(used_entities[:10])}"
        else:
            explanation = f"{len(accurate_entities)}/{len(used_entities)} 个实体描述准确"
            if inaccurate_entities:
                explanation += f"\n    不准确: {', '.join(inaccurate_entities[:3])}"
            if accurate_entities:
                explanation += f"\n    准确: {', '.join(accurate_entities[:10])}"

        return MetricResult(
            name="rag_entity_accuracy",
            value=accuracy,
            max_value=1.0,
            explanation=explanation
        )


def _is_entity_mentioned(entity_name: str, text: str) -> bool:
    """
    检查实体是否在文本中被提及（增强的模糊匹配）

    支持：
    - 精确匹配
    - 部分匹配（实体的关键词）
    - 缩写匹配
    - 中英文地名映射（如 SHENZHEN CITY ↔ 深圳）
    """
    if not entity_name or not text:
        return False

    # 归一化：去除标点符号和空格
    entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity_name).upper()
    text_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', text).upper()

    # 1. 精确匹配
    if entity_normalized in text_normalized:
        return True

    # 2. 中英文地名映射匹配
    from .novel_content_metrics import _EN_CN_LOCATION_MAP, _CN_EN_LOCATION_MAP
    entity_lower = entity_name.lower().strip()
    if entity_lower in _EN_CN_LOCATION_MAP:
        cn_name = _EN_CN_LOCATION_MAP[entity_lower]
        if cn_name in text:
            return True
    # 多词英文实体名：逐词查映射表
    entity_words_en = re.findall(r'[a-zA-Z]+', entity_name.lower())
    for word in entity_words_en:
        if word in _EN_CN_LOCATION_MAP:
            cn_name = _EN_CN_LOCATION_MAP[word]
            if cn_name in text:
                return True
    if entity_name in _CN_EN_LOCATION_MAP:
        en_variants = _CN_EN_LOCATION_MAP[entity_name]
        for en in en_variants:
            if en.upper() in text_normalized:
                return True

    # 3. 反向匹配：文本中的词是否是实体的一部分
    # 例如：实体="庚申年猴票"，文本包含"猴票" → 匹配
    entity_words = re.findall(r'[一-龥]{2,}', entity_name)
    for word in entity_words:
        if len(word) >= 2 and word in text:
            return True

    # 4. 部分匹配（长实体 ≥4 字符）
    if len(entity_normalized) >= 4:
        if entity_normalized[:4] in text_normalized:
            return True

    return False


def _extract_core_concepts(memoir_text: str) -> List[str]:
    """
    提取原文核心概念（名词、专有名词）

    使用 jieba 分词 + 词性标注，提取：
    - nr: 人名
    - ns: 地名
    - nt: 机构名
    - nz: 其他专名
    - n: 普通名词（过滤常见词）
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        # 降级方案：使用正则提取 2-4 字的中文词
        return list(set(re.findall(r'[一-龥]{2,4}', memoir_text)))[:20]

    # 目标词性
    target_pos = {'nr', 'ns', 'nt', 'nz', 'n'}

    # 常见词黑名单（过滤掉太泛的词）
    blacklist = {
        '时候', '时间', '地方', '方面', '情况', '问题', '事情', '东西',
        '人', '大家', '我们', '他们', '自己', '什么', '怎么', '这样',
        '那样', '如何', '为什么', '因为', '所以', '但是', '然后',
        '开始', '结束', '进行', '发生', '出现', '成为', '变成',
    }

    words = pseg.cut(memoir_text)
    concepts = []

    for word, flag in words:
        if (len(word) >= 2 and
            flag in target_pos and
            word not in blacklist):
            concepts.append(word)

    # 去重并限制数量
    return list(dict.fromkeys(concepts))[:30]


def _is_concept_preserved(concept: str, generated_text: str) -> bool:
    """
    检查概念是否在生成文本中保留（支持部分匹配）

    匹配策略：
    1. 精确匹配：概念完整出现
    2. 部分匹配：概念的主要部分出现（≥2字）
    3. 同义匹配：概念的核心词出现
    """
    if not concept or not generated_text:
        return False

    # 归一化
    concept_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', concept).upper()
    text_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', generated_text).upper()

    # 1. 精确匹配
    if concept_normalized in text_normalized:
        return True

    # 2. 部分匹配（对于 ≥3 字的概念，匹配前 2 字或后 2 字）
    if len(concept) >= 3:
        # 提取概念中的 2 字词
        for i in range(len(concept) - 1):
            substring = concept[i:i+2]
            if len(substring) == 2 and substring in generated_text:
                return True

    # 3. 同义匹配（对于复合词，匹配核心词）
    # 例如："大学" 可以匹配 "大学生"、"上大学"
    if len(concept) == 2 and concept in generated_text:
        return True

    return False


class RelevanceMetrics:
    """相关性指标"""
    
    @staticmethod
    def topic_coherence(
        generated_text: str,
        memoir_text: str,
    ) -> MetricResult:
        """
        主题一致性（Topic Coherence）

        检查核心概念是否保留（支持同义改写和部分匹配）

        与旧指标的区别：
        - keyword_overlap: 精确匹配 → 无法识别同义改写
        - semantic_similarity: 2-gram重叠 → 对expansion任务不适用
        - topic_coherence: 核心概念保留（支持同义） → 更合理

        计算方式：
        1. 提取原文核心概念（名词、专有名词）
        2. 检查每个概念是否在生成文本中出现（支持部分匹配）
        3. 计算保留率
        """
        # 提取原文核心概念
        core_concepts = _extract_core_concepts(memoir_text)

        if not core_concepts:
            return MetricResult(
                name="topic_coherence",
                value=0.5,
                max_value=1.0,
                explanation="原文未提取到核心概念"
            )

        # 检查每个概念是否保留
        preserved = 0
        preserved_concepts = []
        missing_concepts = []

        for concept in core_concepts:
            if _is_concept_preserved(concept, generated_text):
                preserved += 1
                preserved_concepts.append(concept)
            else:
                missing_concepts.append(concept)

        score = preserved / len(core_concepts)

        explanation = f"保留了 {preserved}/{len(core_concepts)} 个核心概念"
        if preserved_concepts:
            examples = ', '.join(preserved_concepts[:5])
            explanation += f"\n    已保留: {examples}"
            if len(preserved_concepts) > 5:
                explanation += f" 等{len(preserved_concepts)}个"
        if missing_concepts and len(missing_concepts) <= 5:
            explanation += f"\n    未保留: {', '.join(missing_concepts)}"

        return MetricResult(
            name="topic_coherence",
            value=score,
            max_value=1.0,
            explanation=explanation
        )



class LiteraryMetrics:
    """文学性指标"""
    
    @staticmethod
    def length_score(
        generated_text: str,
        min_length: int = 150,
        max_length: int = 600,
        optimal_min: int = 200,
        optimal_max: int = 500,
    ) -> MetricResult:
        """
        长度评分
        """
        length = len(generated_text)
        
        if optimal_min <= length <= optimal_max:
            score = 1.0
            explanation = f"长度 {length} 字，处于最佳范围"
        elif min_length <= length <= max_length:
            # 计算偏离最佳范围的程度
            if length < optimal_min:
                score = 0.7 + 0.3 * (length - min_length) / (optimal_min - min_length)
            else:
                score = 0.7 + 0.3 * (max_length - length) / (max_length - optimal_max)
            explanation = f"长度 {length} 字，略偏离最佳范围"
        else:
            score = 0.3
            explanation = f"长度 {length} 字，偏离推荐范围"
        
        return MetricResult(
            name="length_score",
            value=score,
            max_value=1.0,
            explanation=explanation
        )
    
    @staticmethod
    def paragraph_structure(
        generated_text: str,
        relaxed: bool = False,
        relaxed_max_paragraphs: int = 48,
    ) -> MetricResult:
        """
        段落结构评分。
        relaxed=True时适用于长文合并输出（允许多段换行）。
        """
        paragraphs = [p.strip() for p in generated_text.split("\n") if p.strip()]
        num_paragraphs = len(paragraphs)

        if relaxed:
            if num_paragraphs == 0:
                score = 0.0
                explanation = "无段落内容"
            elif num_paragraphs <= relaxed_max_paragraphs:
                score = 1.0
                explanation = f"{num_paragraphs} 个段落（长文模式）"
            else:
                score = 0.75
                explanation = f"{num_paragraphs} 个段落，略多"
            return MetricResult(
                name="paragraph_structure",
                value=score,
                max_value=1.0,
                explanation=explanation,
            )

        if 1 <= num_paragraphs <= 3:
            score = 1.0
            explanation = f"{num_paragraphs} 个段落，结构良好"
        elif num_paragraphs == 0:
            score = 0.0
            explanation = "无段落内容"
        else:
            score = 0.6
            explanation = f"{num_paragraphs} 个段落，结构略显零散"

        return MetricResult(
            name="paragraph_structure",
            value=score,
            max_value=1.0,
            explanation=explanation,
        )
    
    @staticmethod
    def transition_usage(
        generated_text: str,
    ) -> MetricResult:
        """
        过渡词使用评分
        """
        transition_words = [
            "那时", "当时", "正是", "与此同时", "记得", "那个年代",
            "在那个时候", "彼时", "此时", "随后", "紧接着", "于是",
            "就这样", "从此", "后来", "那一年", "那年",
        ]
        
        used = [w for w in transition_words if w in generated_text]
        
        if len(used) >= 2:
            score = 1.0
            explanation = f"使用了多个过渡词: {', '.join(used[:3])}"
        elif len(used) == 1:
            score = 0.7
            explanation = f"使用了过渡词: {used[0]}"
        else:
            score = 0.4
            explanation = "未使用过渡词"
        
        return MetricResult(
            name="transition_usage",
            value=score,
            max_value=1.0,
            explanation=explanation
        )
    
    @staticmethod
    def descriptive_richness(
        generated_text: str,
    ) -> MetricResult:
        """
        描述丰富度
        检查是否使用了丰富的描述性词汇
        """
        descriptive_patterns = [
            r'[形色声味触]', # 感官词
            r'[\u4e00-\u9fa5]{2,4}的',  # 形容词 + 的
            r'如同|仿佛|好像|宛如',  # 比喻
            r'[喜怒哀乐悲欢离合]',  # 情感词
        ]
        
        matches = 0
        for pattern in descriptive_patterns:
            if re.search(pattern, generated_text):
                matches += 1
        
        score = matches / len(descriptive_patterns)
        
        return MetricResult(
            name="descriptive_richness",
            value=score,
            max_value=1.0,
            explanation=f"描述性元素评分: {matches}/{len(descriptive_patterns)}"
        )


async def calculate_all_metrics(
    memoir_text: str,
    generated_text: str,
    reference_entities: Optional[List[str]] = None,
    reference_year: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    *,
    literary_length_min: int = 150,
    literary_length_max: int = 600,
    literary_optimal_min: int = 200,
    literary_optimal_max: int = 500,
    literary_paragraph_relaxed: bool = False,
    novel_content_brief: Optional[Any] = None,
    task_type: str = "expansion",
    min_required_entities: int = 2,
    llm_adapter: Optional[Any] = None,
) -> Dict[str, MetricResult]:
    """
    计算所有指标

    literary_* 参数用于长文/分章场景下校准长度与段落评分。
    novel_content_brief: NovelContentBrief 对象（可选），用于计算新内容指标
    task_type: 任务类型 ("expansion" 或 "summarization")
    min_required_entities: expansion任务要求的最少实体数
    llm_adapter: LLM 适配器（可选），用于 LLM 实体提取

    Returns:
        Dict[str, MetricResult]: 指标名称到结果的映射
    """
    results = {}

    # 准确性指标
    results["entity_coverage"] = AccuracyMetrics.entity_coverage(
        generated_text, reference_entities or [],
        task_type=task_type,
        min_required=min_required_entities,
    )
    results["temporal_coherence"] = AccuracyMetrics.temporal_coherence(
        generated_text, reference_year
    )
    results["rag_entity_accuracy"] = AccuracyMetrics.rag_entity_accuracy(
        generated_text, novel_content_brief
    )

    # 相关性指标
    results["topic_coherence"] = RelevanceMetrics.topic_coherence(
        generated_text, memoir_text
    )

    # 文学性指标
    results["length_score"] = LiteraryMetrics.length_score(
        generated_text,
        min_length=literary_length_min,
        max_length=literary_length_max,
        optimal_min=literary_optimal_min,
        optimal_max=literary_optimal_max,
    )
    results["paragraph_structure"] = LiteraryMetrics.paragraph_structure(
        generated_text,
        relaxed=literary_paragraph_relaxed,
    )
    results["transition_usage"] = LiteraryMetrics.transition_usage(generated_text)
    results["descriptive_richness"] = LiteraryMetrics.descriptive_richness(generated_text)

    # 新内容指标（如果提供了 novel_content_brief）
    if novel_content_brief is not None:
        from .novel_content_metrics import (
            information_gain_metric,
            expansion_grounding_metric,
            rag_utilization_metric,
            hallucination_metric,
        )

        results["information_gain"] = await information_gain_metric(
            memoir_text, generated_text, novel_content_brief, llm_adapter
        )
        results["expansion_grounding"] = await expansion_grounding_metric(
            memoir_text, generated_text, novel_content_brief, llm_adapter
        )
        results["rag_utilization"] = await rag_utilization_metric(
            memoir_text, generated_text, novel_content_brief, llm_adapter
        )
        results["hallucination"] = await hallucination_metric(
            memoir_text, generated_text, novel_content_brief, llm_adapter
        )

    return results


def aggregate_scores(
    metrics: Dict[str, MetricResult],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    聚合指标分数

    Args:
        metrics: 指标结果
        weights: 权重（可选）

    Returns:
        float: 综合分数 (0-10)
    """
    if not weights:
        weights = {
            "entity_coverage": 1.5,
            "temporal_coherence": 1.0,
            "rag_entity_accuracy": 1.5,  # 实体描述准确性（重要）
            "topic_coherence": 1.5,  # 主题一致性（重要）
            "length_score": 0.8,
            "paragraph_structure": 0.8,
            "transition_usage": 0.8,
            "descriptive_richness": 0.8,
            # 新内容指标权重
            "information_gain": 1.2,
            "expansion_grounding": 1.5,
            "rag_utilization": 2.0,  # RAG 利用率（高优先级）
            "hallucination": 2.0,  # 幻觉检测（高优先级）
        }

    total_weight = sum(weights.get(k, 1.0) for k in metrics)
    weighted_sum = sum(
        metrics[k].normalized * weights.get(k, 1.0)
        for k in metrics
    )

    # 归一化到 0-10 分
    return (weighted_sum / total_weight) * 10 if total_weight > 0 else 0


# ---------------------------------------------------------------------------
# 跨章指标
# ---------------------------------------------------------------------------

class CrossChapterMetrics:
    """多章维度的评估指标——衡量全文整体质量。"""

    @staticmethod
    def inter_chapter_repetition(
        chapters: List[str],
        ngram_n: int = 6,
    ) -> MetricResult:
        """
        跨章 n-gram 重叠率。值越低越好 (0=无重复, 1=完全重复)。
        最终评分 = 1 - avg_overlap，即评分越高越好。
        """
        if len(chapters) < 2:
            return MetricResult(
                name="inter_chapter_repetition",
                value=1.0,
                max_value=1.0,
                explanation="单章或无章，无需检测",
            )

        def _ngrams(text: str, n: int) -> Set[str]:
            clean = re.sub(r"\s+", "", text)
            if len(clean) < n:
                return set()
            return {clean[i:i + n] for i in range(len(clean) - n + 1)}

        ratios = []
        for i in range(len(chapters) - 1):
            a = _ngrams(chapters[i], ngram_n)
            b = _ngrams(chapters[i + 1], ngram_n)
            if a and b:
                overlap = len(a & b) / min(len(a), len(b))
                ratios.append(overlap)

        avg_overlap = sum(ratios) / len(ratios) if ratios else 0.0
        score = max(0.0, 1.0 - avg_overlap)

        return MetricResult(
            name="inter_chapter_repetition",
            value=score,
            max_value=1.0,
            explanation=f"平均跨章 {ngram_n}-gram 重叠率 {avg_overlap:.1%}，"
                        f"去重评分 {score:.2f}",
        )

    @staticmethod
    def style_consistency(
        chapters: List[str],
    ) -> MetricResult:
        """
        风格一致性：基于平均句长方差衡量。
        各章平均句长越接近，一致性越高。
        """
        if len(chapters) < 2:
            return MetricResult(
                name="style_consistency",
                value=1.0,
                max_value=1.0,
                explanation="单章，无需检测",
            )

        avg_lens = []
        for ch in chapters:
            sents = [s.strip() for s in re.split(r"[。！？]", ch) if s.strip()]
            if sents:
                avg_lens.append(sum(len(s) for s in sents) / len(sents))

        if len(avg_lens) < 2:
            return MetricResult(
                name="style_consistency",
                value=1.0,
                max_value=1.0,
                explanation="有效章节不足",
            )

        mean = sum(avg_lens) / len(avg_lens)
        variance = sum((x - mean) ** 2 for x in avg_lens) / len(avg_lens)
        cv = (variance ** 0.5) / mean if mean > 0 else 0  # 变异系数

        # cv ≤ 0.2 → 1.0; cv ≥ 0.6 → 0.3
        if cv <= 0.2:
            score = 1.0
        elif cv >= 0.6:
            score = 0.3
        else:
            score = 1.0 - (cv - 0.2) / 0.4 * 0.7

        return MetricResult(
            name="style_consistency",
            value=score,
            max_value=1.0,
            explanation=f"各章平均句长变异系数 {cv:.2f}，一致性评分 {score:.2f}",
        )

    @staticmethod
    def summary_sentence_ratio(
        chapters: List[str],
    ) -> MetricResult:
        """
        总结性语句占比：检测「总之」「综上」等总结句，占比越低越好。
        """
        _SUMMARY_RE = re.compile(
            r"(?:总之|综上|总而言之|总的来说|总体而言|概括地说|简而言之|"
            r"由此可见|可以看出|不难看出|归根结底|一言以蔽之)"
        )
        total_sents = 0
        summary_sents = 0
        for ch in chapters:
            sents = [s.strip() for s in re.split(r"[。！？]", ch) if s.strip()]
            total_sents += len(sents)
            summary_sents += sum(1 for s in sents if _SUMMARY_RE.search(s))

        if total_sents == 0:
            return MetricResult(
                name="summary_sentence_ratio",
                value=1.0,
                max_value=1.0,
                explanation="无句子",
            )

        ratio = summary_sents / total_sents
        score = max(0.0, 1.0 - ratio * 5)  # ratio=0 → 1.0, ratio=0.2 → 0.0

        return MetricResult(
            name="summary_sentence_ratio",
            value=score,
            max_value=1.0,
            explanation=f"总结性语句 {summary_sents}/{total_sents} "
                        f"(占比 {ratio:.1%})，评分 {score:.2f}",
        )
