"""
评估模块
实现事实准确性、相关性和文学性的评估
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import json

from ..llm import LLMAdapter, create_llm_adapter
from ..retrieval import RetrievalResult


class EvaluationDimension(Enum):
    """评估维度"""
    ACCURACY = "accuracy"          # 事实准确性
    RELEVANCE = "relevance"        # 相关性
    LITERARY = "literary"          # 文学性
    COHERENCE = "coherence"        # 连贯性
    OVERALL = "overall"            # 综合评分


@dataclass
class DimensionScore:
    """单个维度的评分"""
    dimension: EvaluationDimension
    score: float  # 0-10 分
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """评估结果"""
    scores: Dict[str, DimensionScore]
    overall_score: float
    summary: str
    suggestions: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "scores": {
                k: {
                    "dimension": v.dimension.value,
                    "score": v.score,
                    "explanation": v.explanation,
                }
                for k, v in self.scores.items()
            },
            "overall_score": self.overall_score,
            "summary": self.summary,
            "suggestions": self.suggestions,
        }
    
    def get_score(self, dimension: str) -> float:
        """获取指定维度的评分"""
        if dimension in self.scores:
            return self.scores[dimension].score
        return 0.0


class Evaluator:
    """
    评估器
    
    支持三个维度的评估：
    1. 事实准确性 - 生成的历史背景是否准确
    2. 相关性 - 与回忆录内容的关联程度
    3. 文学性 - 文本的文学表达质量
    """
    
    # 评估提示词
    EVALUATION_PROMPT = """你是一位专业的文本评估专家。请对以下生成的历史背景文本进行评估。

## 原始回忆录
{memoir_text}

## 生成的历史背景
{generated_text}

## 参考历史信息（如有）
{reference_info}

请从以下三个维度进行评估，每个维度打分0-10分：

### 1. 事实准确性 (Accuracy)
评估生成的历史信息是否准确：
- 时间、地点、人物是否正确
- 历史事件描述是否符合史实
- 是否存在明显的事实错误

### 2. 相关性 (Relevance)
评估生成内容与回忆录的关联程度：
- 历史背景是否与回忆录的时间地点相符
- 是否能够补充和增强回忆录的内容
- 主题是否一致

### 3. 文学性 (Literary)
评估文本的文学表达质量：
- 语言是否优美流畅
- 是否具有感染力和画面感
- 与回忆录的风格是否融合

请以JSON格式返回评估结果：
{{
    "accuracy": {{
        "score": <0-10>,
        "explanation": "<评分理由>"
    }},
    "relevance": {{
        "score": <0-10>,
        "explanation": "<评分理由>"
    }},
    "literary": {{
        "score": <0-10>,
        "explanation": "<评分理由>"
    }},
    "overall": {{
        "score": <0-10>,
        "explanation": "<综合评价>"
    }},
    "suggestions": ["<改进建议1>", "<改进建议2>"]
}}

只返回JSON，不要其他内容。"""

    # 简化版评估（不需要LLM）
    SIMPLE_EVALUATION_CRITERIA = {
        "accuracy": {
            "time_match": 2.0,      # 时间信息匹配
            "location_match": 2.0,   # 地点信息匹配
            "no_contradiction": 3.0, # 无明显矛盾
            "source_based": 3.0,     # 基于可靠来源
        },
        "relevance": {
            "topic_match": 3.0,      # 主题匹配
            "context_fit": 3.0,      # 上下文契合
            "entity_overlap": 2.0,   # 实体重叠
            "semantic_sim": 2.0,     # 语义相似度
        },
        "literary": {
            "fluency": 3.0,          # 流畅性
            "expressiveness": 3.0,   # 表现力
            "style_match": 2.0,      # 风格匹配
            "length_appropriate": 2.0, # 长度适当
        },
    }
    
    def __init__(self, llm_adapter: Optional[LLMAdapter] = None):
        """
        初始化评估器
        
        Args:
            llm_adapter: LLM适配器（用于深度评估）
        """
        self.llm_adapter = llm_adapter
    
    async def evaluate(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult] = None,
        use_llm: bool = True,
    ) -> EvaluationResult:
        """
        评估生成的历史背景文本
        
        Args:
            memoir_text: 原始回忆录文本
            generated_text: 生成的历史背景文本
            retrieval_result: 检索结果（可选，用于准确性评估）
            use_llm: 是否使用LLM进行评估
            
        Returns:
            EvaluationResult: 评估结果
        """
        if use_llm and self.llm_adapter:
            return await self._evaluate_with_llm(
                memoir_text, generated_text, retrieval_result
            )
        else:
            return self._evaluate_simple(
                memoir_text, generated_text, retrieval_result
            )
    
    def evaluate_sync(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> EvaluationResult:
        """同步版本的评估"""
        return asyncio.run(
            self.evaluate(memoir_text, generated_text, retrieval_result, use_llm=False)
        )
    
    async def _evaluate_with_llm(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult],
    ) -> EvaluationResult:
        """使用LLM进行深度评估"""
        reference_info = ""
        if retrieval_result:
            reference_info = retrieval_result.get_context_text()
        
        prompt = self.EVALUATION_PROMPT.format(
            memoir_text=memoir_text,
            generated_text=generated_text,
            reference_info=reference_info or "无参考信息",
        )
        
        try:
            response = await self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="你是一位专业的文本质量评估专家。请客观、公正地评估文本质量。",
                temperature=0.1,
                max_tokens=1000,
            )
            
            # 解析JSON响应
            result_data = json.loads(response.content)
            
            scores = {}
            for dim in ["accuracy", "relevance", "literary"]:
                if dim in result_data:
                    scores[dim] = DimensionScore(
                        dimension=EvaluationDimension(dim),
                        score=float(result_data[dim].get("score", 0)),
                        explanation=result_data[dim].get("explanation", ""),
                    )
            
            overall_data = result_data.get("overall", {})
            overall_score = float(overall_data.get("score", 0))
            
            return EvaluationResult(
                scores=scores,
                overall_score=overall_score,
                summary=overall_data.get("explanation", ""),
                suggestions=result_data.get("suggestions", []),
                raw_response=response.content,
            )
            
        except Exception as e:
            # LLM评估失败，回退到简单评估
            return self._evaluate_simple(memoir_text, generated_text, retrieval_result)
    
    def _evaluate_simple(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult],
    ) -> EvaluationResult:
        """简单评估（不使用LLM）"""
        scores = {}
        
        # 评估准确性
        accuracy_score = self._evaluate_accuracy_simple(
            generated_text, retrieval_result
        )
        scores["accuracy"] = accuracy_score
        
        # 评估相关性
        relevance_score = self._evaluate_relevance_simple(
            memoir_text, generated_text, retrieval_result
        )
        scores["relevance"] = relevance_score
        
        # 评估文学性
        literary_score = self._evaluate_literary_simple(
            memoir_text, generated_text
        )
        scores["literary"] = literary_score
        
        # 计算综合评分
        overall = (
            accuracy_score.score * 0.4 +
            relevance_score.score * 0.3 +
            literary_score.score * 0.3
        )
        
        return EvaluationResult(
            scores=scores,
            overall_score=round(overall, 2),
            summary="基于规则的自动评估结果",
            suggestions=self._generate_suggestions(scores),
        )
    
    def _evaluate_accuracy_simple(
        self,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult],
    ) -> DimensionScore:
        """简单的准确性评估"""
        score = 5.0  # 基础分
        explanations = []
        
        if retrieval_result:
            # 检查是否使用了检索到的实体
            entity_count = 0
            for entity in retrieval_result.entities[:5]:
                name = entity.get("name", "")
                if name and name in generated_text:
                    entity_count += 1
            
            if entity_count > 0:
                score += min(entity_count, 3)
                explanations.append(f"引用了{entity_count}个检索到的实体")
            
            # 检查时间一致性
            context = retrieval_result.context
            if context.year and context.year in generated_text:
                score += 1
                explanations.append("时间信息一致")
            
            if context.location and context.location in generated_text:
                score += 1
                explanations.append("地点信息一致")
        
        return DimensionScore(
            dimension=EvaluationDimension.ACCURACY,
            score=min(score, 10.0),
            explanation="; ".join(explanations) if explanations else "基于内容分析的评估",
        )
    
    def _evaluate_relevance_simple(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult],
    ) -> DimensionScore:
        """简单的相关性评估"""
        score = 5.0
        explanations = []
        
        # 计算词汇重叠
        memoir_words = set(memoir_text)
        generated_words = set(generated_text)
        overlap = len(memoir_words & generated_words)
        
        if overlap > 10:
            score += 2
            explanations.append("词汇重叠度较高")
        elif overlap > 5:
            score += 1
            explanations.append("词汇有一定重叠")
        
        # 检查主题一致性（通过关键词）
        if retrieval_result:
            context = retrieval_result.context
            keyword_match = 0
            for keyword in context.keywords:
                if keyword in generated_text:
                    keyword_match += 1
            
            if keyword_match > 0:
                score += min(keyword_match, 3)
                explanations.append(f"包含{keyword_match}个主题关键词")
        
        return DimensionScore(
            dimension=EvaluationDimension.RELEVANCE,
            score=min(score, 10.0),
            explanation="; ".join(explanations) if explanations else "基于内容分析的评估",
        )
    
    def _evaluate_literary_simple(
        self,
        memoir_text: str,
        generated_text: str,
    ) -> DimensionScore:
        """简单的文学性评估"""
        score = 5.0
        explanations = []
        
        # 检查长度适当性
        length = len(generated_text)
        if 200 <= length <= 500:
            score += 2
            explanations.append("长度适当")
        elif 100 <= length <= 800:
            score += 1
            explanations.append("长度可接受")
        
        # 检查段落结构
        paragraphs = [p for p in generated_text.split("\n") if p.strip()]
        if 1 <= len(paragraphs) <= 3:
            score += 1
            explanations.append("段落结构良好")
        
        # 检查是否有过渡词
        transition_words = ["那时", "当时", "正是", "与此同时", "记得", "那个年代"]
        transition_count = sum(1 for w in transition_words if w in generated_text)
        if transition_count > 0:
            score += min(transition_count, 2)
            explanations.append("使用了过渡词")
        
        return DimensionScore(
            dimension=EvaluationDimension.LITERARY,
            score=min(score, 10.0),
            explanation="; ".join(explanations) if explanations else "基于规则的评估",
        )
    
    def _generate_suggestions(
        self,
        scores: Dict[str, DimensionScore]
    ) -> List[str]:
        """根据评分生成改进建议"""
        suggestions = []
        
        if scores.get("accuracy") and scores["accuracy"].score < 7:
            suggestions.append("建议增加更多可验证的历史事实")
        
        if scores.get("relevance") and scores["relevance"].score < 7:
            suggestions.append("建议加强与回忆录主题的关联")
        
        if scores.get("literary") and scores["literary"].score < 7:
            suggestions.append("建议使用更多文学性的表达方式")
        
        return suggestions


class BatchEvaluator:
    """
    批量评估器
    用于评估多个生成结果，支持多LLM对比评估
    """
    
    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator
    
    async def evaluate_batch(
        self,
        memoir_text: str,
        generated_texts: Dict[str, str],
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> Dict[str, EvaluationResult]:
        """
        批量评估多个生成结果
        
        Args:
            memoir_text: 原始回忆录
            generated_texts: 生成文本字典 {provider: text}
            retrieval_result: 检索结果
            
        Returns:
            Dict[str, EvaluationResult]: 各提供商的评估结果
        """
        results = {}
        
        for provider, text in generated_texts.items():
            result = await self.evaluator.evaluate(
                memoir_text=memoir_text,
                generated_text=text,
                retrieval_result=retrieval_result,
            )
            results[provider] = result
        
        return results
    
    def generate_comparison_report(
        self,
        results: Dict[str, EvaluationResult]
    ) -> str:
        """生成对比报告"""
        report = ["# 多模型评估对比报告\n"]
        
        # 汇总表
        report.append("## 评分汇总\n")
        report.append("| 模型 | 准确性 | 相关性 | 文学性 | 综合 |")
        report.append("|------|--------|--------|--------|------|")
        
        for provider, result in results.items():
            accuracy = result.get_score("accuracy")
            relevance = result.get_score("relevance")
            literary = result.get_score("literary")
            overall = result.overall_score
            report.append(
                f"| {provider} | {accuracy:.1f} | {relevance:.1f} | {literary:.1f} | {overall:.1f} |"
            )
        
        # 详细分析
        report.append("\n## 详细分析\n")
        
        for provider, result in results.items():
            report.append(f"### {provider.upper()}\n")
            report.append(f"**综合评分**: {result.overall_score}/10\n")
            report.append(f"**评价**: {result.summary}\n")
            
            if result.suggestions:
                report.append("**改进建议**:")
                for suggestion in result.suggestions:
                    report.append(f"- {suggestion}")
            
            report.append("")
        
        # 最佳选择
        if results:
            best_provider = max(results.items(), key=lambda x: x[1].overall_score)
            report.append(f"\n## 推荐\n")
            report.append(f"综合评分最高的模型是 **{best_provider[0]}**，评分 {best_provider[1].overall_score}/10")
        
        return "\n".join(report)
