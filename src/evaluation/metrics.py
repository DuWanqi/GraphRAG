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
    def temporal_coherence(
        generated_text: str,
        memoir_text: str,
    ) -> MetricResult:
        """
        时间一致性（Temporal Coherence）

        提取原文中所有年份作为白名单，检查生成文本中的年份是否全部在白名单内。
        score = 合规年份数 / 生成年份总数；无生成年份时 score = 1.0。
        门控由 quality_gate.py 负责（score < 1.0 触发 error）。
        """
        year_pattern = r'(\d{4})'

        def _extract_years(text: str) -> set:
            return {int(y) for y in re.findall(year_pattern, text) if 1000 <= int(y) <= 2100}

        memoir_years = _extract_years(memoir_text)
        gen_years = _extract_years(generated_text)

        if not gen_years:
            return MetricResult(
                name="temporal_coherence",
                value=1.0,
                max_value=1.0,
                explanation="生成文本中未出现年份",
            )

        valid = gen_years & memoir_years
        invalid = gen_years - memoir_years
        score = len(valid) / len(gen_years)

        if invalid:
            explanation = (
                f"生成年份 {sorted(gen_years)} 中，"
                f"{sorted(invalid)} 不在原文年份白名单 {sorted(memoir_years)} 内"
            )
        else:
            explanation = f"生成年份 {sorted(gen_years)} 均在原文年份白名单内"

        return MetricResult(
            name="temporal_coherence",
            value=score,
            max_value=1.0,
            explanation=explanation,
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


_LITERARY_RUBRIC_PROMPT = """\
你是一位文学评估专家，请对以下回忆录扩写文本进行评估。

【待评文本】
{text}

请按照以下两个维度打分（0.0-1.0），并以 JSON 格式返回，不要输出任何其他内容：

1. descriptive_richness（描述丰富度）
   - 0.0-0.3：几乎无感官描写、情感表达或比喻修辞，叙述平铺直叙
   - 0.4-0.6：有少量描述性词汇，但整体较平淡
   - 0.7-0.9：感官描写、情感表达或比喻修辞较丰富
   - 1.0：感官描写、情感表达、比喻修辞、细节具体度四类均丰富

2. transition_usage（叙事过渡）
   - 0.0-0.3：叙事跳跃，缺乏时间/场景过渡
   - 0.4-0.6：有基本过渡词，但衔接不够自然
   - 0.7-0.9：过渡自然，叙事连贯
   - 1.0：时间过渡、场景切换、叙事连贯性均优秀

返回格式（仅 JSON，不含 markdown 代码块）：
{{"descriptive_richness": 0.8, "transition_usage": 0.7}}
"""


async def literary_rubric_eval(
    generated_text: str,
    llm_adapter: Any,
) -> Dict[str, float]:
    """
    用单次 LLM 推理对 descriptive_richness 和 transition_usage 按 rubric 打分。
    解析失败时返回 fallback（基于原有关键词规则）。
    """
    import json as _json

    fallback = {
        "descriptive_richness": LiteraryMetrics.descriptive_richness(generated_text).normalized,
        "transition_usage": LiteraryMetrics.transition_usage(generated_text).normalized,
    }

    if llm_adapter is None:
        return fallback

    try:
        prompt = _LITERARY_RUBRIC_PROMPT.format(text=generated_text[:2000])
        response = await llm_adapter.generate(prompt)
        raw = response.strip()
        # 去除可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        scores = _json.loads(raw)
        result = {}
        for key in ("descriptive_richness", "transition_usage"):
            val = float(scores.get(key, fallback[key]))
            result[key] = max(0.0, min(1.0, val))
        return result
    except Exception:
        return fallback


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
    novel_content_brief: Optional[Any] = None,
    task_type: str = "expansion",
    min_required_entities: int = 2,
    llm_adapter: Optional[Any] = None,
) -> Dict[str, MetricResult]:
    """
    计算所有指标

    literary_* 参数用于长文/分章场景下校准长度评分。
    novel_content_brief: NovelContentBrief 对象（可选），用于计算新内容指标
    llm_adapter: LLM 适配器（可选），用于 LLM 实体提取和文学性 rubric 评分

    Returns:
        Dict[str, MetricResult]: 指标名称到结果的映射
    """
    results = {}

    # 准确性指标：用原文提取年份白名单
    results["temporal_coherence"] = AccuracyMetrics.temporal_coherence(
        generated_text, memoir_text
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

    # 文学性 rubric：单次 LLM 推理评 descriptive_richness + transition_usage
    rubric_scores = await literary_rubric_eval(generated_text, llm_adapter)
    results["transition_usage"] = MetricResult(
        name="transition_usage",
        value=rubric_scores["transition_usage"],
        max_value=1.0,
        explanation="LLM rubric 评分（叙事过渡自然度）",
    )
    results["descriptive_richness"] = MetricResult(
        name="descriptive_richness",
        value=rubric_scores["descriptive_richness"],
        max_value=1.0,
        explanation="LLM rubric 评分（感官/情感/比喻/细节丰富度）",
    )

    # 新内容指标（如果提供了 novel_content_brief）
    if novel_content_brief is not None:
        from .novel_content_metrics import (
            rag_utilization_metric,
            hallucination_metric,
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
            "temporal_coherence": 1.0,
            "topic_coherence": 1.5,       # 主题一致性（重要）
            "length_score": 0.8,
            "transition_usage": 0.8,
            "descriptive_richness": 0.8,
            "rag_utilization": 2.0,        # RAG 利用率（高优先级）
            "hallucination": 2.0,          # 幻觉检测（高优先级）
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
