"""
评估指标计算
提供自动化的评估指标计算
"""

from typing import List, Dict, Any, Optional
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
    ) -> MetricResult:
        """
        实体覆盖率
        检查生成文本中包含了多少参考实体
        """
        if not reference_entities:
            return MetricResult(
                name="entity_coverage",
                value=0,
                max_value=1,
                explanation="无参考实体"
            )
        
        covered = sum(1 for e in reference_entities if e in generated_text)
        coverage = covered / len(reference_entities)
        
        return MetricResult(
            name="entity_coverage",
            value=coverage,
            max_value=1.0,
            explanation=f"覆盖了 {covered}/{len(reference_entities)} 个参考实体"
        )
    
    @staticmethod
    def time_consistency(
        generated_text: str,
        reference_year: Optional[str],
    ) -> MetricResult:
        """
        时间一致性
        检查生成的年份是否与参考年份一致
        """
        if not reference_year:
            return MetricResult(
                name="time_consistency",
                value=0.5,
                max_value=1.0,
                explanation="无参考年份"
            )
        
        # 提取生成文本中的年份
        year_pattern = r'(\d{4})年?|(\d{2})年'
        years_in_text = re.findall(year_pattern, generated_text)
        
        if not years_in_text:
            return MetricResult(
                name="time_consistency",
                value=0.3,
                max_value=1.0,
                explanation="生成文本中未找到年份"
            )
        
        # 检查是否包含参考年份
        ref_year = reference_year.replace("年", "")
        
        for year_match in years_in_text:
            year = year_match[0] or year_match[1]
            # 处理两位数年份
            if len(year) == 2:
                year = ("19" if int(year) > 50 else "20") + year
            
            if year == ref_year:
                return MetricResult(
                    name="time_consistency",
                    value=1.0,
                    max_value=1.0,
                    explanation=f"包含参考年份 {reference_year}"
                )
        
        return MetricResult(
            name="time_consistency",
            value=0.5,
            max_value=1.0,
            explanation=f"未找到参考年份 {reference_year}"
        )


class RelevanceMetrics:
    """相关性指标"""
    
    @staticmethod
    def keyword_overlap(
        generated_text: str,
        memoir_text: str,
        keywords: Optional[List[str]] = None,
    ) -> MetricResult:
        """
        关键词重叠度
        """
        if keywords:
            matched = sum(1 for k in keywords if k in generated_text)
            score = matched / len(keywords) if keywords else 0
            return MetricResult(
                name="keyword_overlap",
                value=score,
                max_value=1.0,
                explanation=f"匹配了 {matched}/{len(keywords)} 个关键词"
            )
        
        # 如果没有提供关键词，计算字符重叠
        memoir_chars = set(memoir_text)
        generated_chars = set(generated_text)
        overlap = len(memoir_chars & generated_chars)
        total = len(memoir_chars | generated_chars)
        
        score = overlap / total if total > 0 else 0
        
        return MetricResult(
            name="keyword_overlap",
            value=score,
            max_value=1.0,
            explanation=f"字符重叠度 {score:.2%}"
        )
    
    @staticmethod
    def semantic_similarity(
        generated_text: str,
        memoir_text: str,
    ) -> MetricResult:
        """
        语义相似度（简化版，基于词频）
        如需精确计算，应使用embedding模型
        """
        # 简单的词袋模型
        def get_words(text: str) -> Dict[str, int]:
            words = re.findall(r'[\u4e00-\u9fa5]+', text)
            word_freq = {}
            for word in words:
                for char in word:
                    word_freq[char] = word_freq.get(char, 0) + 1
            return word_freq
        
        memoir_freq = get_words(memoir_text)
        generated_freq = get_words(generated_text)
        
        # 计算余弦相似度
        all_chars = set(memoir_freq.keys()) | set(generated_freq.keys())
        
        dot_product = sum(
            memoir_freq.get(c, 0) * generated_freq.get(c, 0)
            for c in all_chars
        )
        
        norm_memoir = sum(v**2 for v in memoir_freq.values()) ** 0.5
        norm_generated = sum(v**2 for v in generated_freq.values()) ** 0.5
        
        if norm_memoir == 0 or norm_generated == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_memoir * norm_generated)
        
        return MetricResult(
            name="semantic_similarity",
            value=similarity,
            max_value=1.0,
            explanation=f"语义相似度 {similarity:.2%}"
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
    ) -> MetricResult:
        """
        段落结构评分
        """
        paragraphs = [p.strip() for p in generated_text.split("\n") if p.strip()]
        num_paragraphs = len(paragraphs)
        
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


def calculate_all_metrics(
    memoir_text: str,
    generated_text: str,
    reference_entities: Optional[List[str]] = None,
    reference_year: Optional[str] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, MetricResult]:
    """
    计算所有指标
    
    Returns:
        Dict[str, MetricResult]: 指标名称到结果的映射
    """
    results = {}
    
    # 准确性指标
    results["entity_coverage"] = AccuracyMetrics.entity_coverage(
        generated_text, reference_entities or []
    )
    results["time_consistency"] = AccuracyMetrics.time_consistency(
        generated_text, reference_year
    )
    
    # 相关性指标
    results["keyword_overlap"] = RelevanceMetrics.keyword_overlap(
        generated_text, memoir_text, keywords
    )
    results["semantic_similarity"] = RelevanceMetrics.semantic_similarity(
        generated_text, memoir_text
    )
    
    # 文学性指标
    results["length_score"] = LiteraryMetrics.length_score(generated_text)
    results["paragraph_structure"] = LiteraryMetrics.paragraph_structure(generated_text)
    results["transition_usage"] = LiteraryMetrics.transition_usage(generated_text)
    results["descriptive_richness"] = LiteraryMetrics.descriptive_richness(generated_text)
    
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
            "time_consistency": 1.0,
            "keyword_overlap": 1.0,
            "semantic_similarity": 1.0,
            "length_score": 0.8,
            "paragraph_structure": 0.8,
            "transition_usage": 0.8,
            "descriptive_richness": 0.8,
        }
    
    total_weight = sum(weights.get(k, 1.0) for k in metrics)
    weighted_sum = sum(
        metrics[k].normalized * weights.get(k, 1.0)
        for k in metrics
    )
    
    # 归一化到 0-10 分
    return (weighted_sum / total_weight) * 10 if total_weight > 0 else 0
