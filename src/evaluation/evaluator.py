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
from .factscore_adapter import FActScoreChecker, FactCheckResult
from .safe_checker import SAFEFactChecker, SAFECheckResult


class EvaluationDimension(Enum):
    """评估维度"""
    ACCURACY = "accuracy"          # 事实准确性
    RELEVANCE = "relevance"        # 相关性
    LITERARY = "literary"          # 文学性
    COHERENCE = "coherence"        # 连贯性
    COMPLIANCE = "compliance"      # 合规性
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
    fact_check: Optional[FactCheckResult] = None
    safe_check: Optional[SAFECheckResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
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
        if self.fact_check:
            result["fact_check"] = self.fact_check.to_dict()
        if self.safe_check:
            result["safe_check"] = self.safe_check.to_dict()
        return result
    
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
    
    # 评估提示词（合并 Judge 与 Compliance，单次 LLM 调用完成四维度打分）
    EVALUATION_PROMPT = """你是一位专业的文本评估专家。请对以下生成的历史背景文本进行多维度评估，并在同一次回复中完成合规性检查。

## 原始回忆录
{memoir_text}

## 生成的历史背景
{generated_text}

## 参考历史信息（如有）
{reference_info}

请从以下四个维度进行评估，每个维度打分0-10分：

### 1. 事实准确性 (Accuracy)
- 时间、地点、人物是否正确
- 历史事件描述是否符合史实
- 是否存在明显的事实错误

### 2. 相关性 (Relevance)
- 历史背景是否与回忆录的时间地点相符
- 是否能够补充和增强回忆录的内容
- 主题是否一致

### 3. 文学性 (Literary)
- 语言是否优美流畅
- 是否具有感染力和画面感
- 与回忆录的风格是否融合

### 4. 合规性 (Compliance, 10 表示完全合规)
严格检查文本是否存在下列问题；发现任何一项应明显扣分：
- 是否泄露敏感个人信息（身份证号、电话、地址等）
- 是否包含未经证实的谣言或不实信息
- 是否涉及政治敏感内容的不当表述
- 是否存在歧视性或侮辱性言论

请以JSON格式返回评估结果：
{{
    "accuracy": {{"score": <0-10>, "explanation": "<评分理由>"}},
    "relevance": {{"score": <0-10>, "explanation": "<评分理由>"}},
    "literary": {{"score": <0-10>, "explanation": "<评分理由>"}},
    "compliance": {{"score": <0-10>, "issues": ["<问题1>", "<问题2>"], "explanation": "<合规说明>"}},
    "overall": {{"score": <0-10>, "explanation": "<综合评价>"}},
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
    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
    ):
        """
        初始化评估器

        Args:
            llm_adapter: LLM适配器（用于深度评估）
            google_api_key: Google API 密钥（用于 SAFE 网络搜索验证，可选）
            google_cse_id: Google Custom Search Engine ID（可选）
        """
        self.llm_adapter = llm_adapter
        # 使用 FActScoreChecker 替代原来的 FactChecker
        self.fact_checker = FActScoreChecker(llm_adapter)
        self.safe_checker = SAFEFactChecker(
            llm_adapter=llm_adapter,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
        )
    
    async def evaluate(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult] = None,
        use_llm: bool = True,
        enable_fact_check: bool = True,
        enable_safe_check: bool = False,
        use_safe_search: bool = False,
        enable_llm_judge: bool = True,
        batch_size: int = 5,
    ) -> EvaluationResult:
        """
        评估生成的历史背景文本

        Args:
            memoir_text: 原始回忆录文本
            generated_text: 生成的历史背景文本
            retrieval_result: 检索结果（可选，用于准确性评估）
            use_llm: 是否使用LLM进行评估
            enable_fact_check: 是否启用事实性检查（基于知识库）
            enable_safe_check: 是否启用独立知识验证（SAFE，不依赖知识库）
            use_safe_search: SAFE 验证是否使用网络搜索（否则使用 LLM 自身知识）
            enable_llm_judge: 是否启用 LLM-as-a-Judge（相关性/文学性/合规性）

        Returns:
            EvaluationResult: 评估结果
        """
        if (enable_llm_judge or enable_fact_check or enable_safe_check) and not self.llm_adapter:
            raise RuntimeError(
                "评估依赖 LLM-as-a-Judge / 事实检查器，但未配置 LLM 适配器。"
                "请提供有效的 llm_provider。"
            )

        if enable_llm_judge:
            # Judge + Compliance 合并成单次 LLM 调用，scores["compliance"] 来自 _evaluate_with_llm
            result = await self._evaluate_with_llm(
                memoir_text, generated_text, retrieval_result
            )
            # 规则层合规检查（无 LLM 调用），与 LLM 分合并取较低者
            rule_compliance = self._compliance_rule_score(generated_text)
            llm_compliance = result.scores.get("compliance")
            result.scores["compliance"] = self._merge_compliance_scores(
                llm_compliance, rule_compliance
            )
            compliance_score = result.scores["compliance"]
        else:
            result = EvaluationResult(scores={}, overall_score=0.0)

        if enable_fact_check:
            fact_check_result = await self.fact_checker.check(
                memoir_text=memoir_text,
                generated_text=generated_text,
                retrieval_result=retrieval_result,
                use_llm=use_llm and self.llm_adapter is not None,
                batch_size=batch_size,
            )
            result.fact_check = fact_check_result

            if not fact_check_result.is_factual:
                result.suggestions.insert(
                    0,
                    f"⚠️ 事实性警告：{fact_check_result.summary}"
                )

        # SAFE 独立知识验证
        if enable_safe_check:
            kb_factscore = None
            kb_supported = None
            kb_total = None
            shared_atomic_facts = None
            if result.fact_check:
                kb_factscore = result.fact_check.factscore
                kb_supported = result.fact_check.supported_facts
                kb_total = result.fact_check.total_facts
                # 复用 FActScore 已分解好的原子事实，避免重复 LLM 调用
                shared_atomic_facts = result.fact_check.atomic_facts or None

            safe_result = await self.safe_checker.check(
                generated_text=generated_text,
                memoir_text=memoir_text,
                atomic_facts=shared_atomic_facts,
                use_search=use_safe_search,
                kb_factscore=kb_factscore,
                kb_supported_facts=kb_supported,
                kb_total_facts=kb_total,
                batch_size=batch_size,
            )
            result.safe_check = safe_result

        if enable_llm_judge and compliance_score.score < 8:
            result.suggestions.append(
                f"⚠️ 合规性警告：{compliance_score.explanation}"
            )

        return result
    
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
            
            # 解析JSON响应（LLM 可能返回 ```json ... ``` 包裹的内容）
            import re
            raw = response.content.strip()
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                raise ValueError(f"LLM 返回内容中未找到 JSON: {raw[:200]}")
            result_data = json.loads(json_match.group(0))
            
            scores = {}
            for dim in ["accuracy", "relevance", "literary"]:
                if dim in result_data:
                    scores[dim] = DimensionScore(
                        dimension=EvaluationDimension(dim),
                        score=float(result_data[dim].get("score", 0)),
                        explanation=result_data[dim].get("explanation", ""),
                    )

            # 合并到同一个 prompt 的合规性评分
            if "compliance" in result_data:
                comp = result_data["compliance"]
                comp_issues = comp.get("issues") or []
                comp_explanation = comp.get("explanation", "")
                if comp_issues:
                    comp_explanation = (
                        "; ".join(comp_issues) if not comp_explanation
                        else f"{comp_explanation}; {'; '.join(comp_issues)}"
                    )
                scores["compliance"] = DimensionScore(
                    dimension=EvaluationDimension.COMPLIANCE,
                    score=float(comp.get("score", 10)),
                    explanation=comp_explanation or "未发现合规性问题",
                    details={"issues": comp_issues},
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
            raise RuntimeError(f"LLM-as-a-Judge 评估失败: {e}") from e
    
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
    
    def _compliance_rule_score(self, generated_text: str) -> DimensionScore:
        """
        规则层合规性评估（无 LLM 调用）。

        与合并进 Judge prompt 的 LLM 合规判定配合使用：
        - 正则命中的是硬违规（身份证、手机号、邮箱），LLM 可能漏判
        - 最终合规分取 min(规则分, LLM 分)，issues 合并
        """
        import re as _re

        issues: List[str] = []
        score = 10.0

        id_pattern = _re.compile(r'\d{17}[\dXx]')
        phone_pattern = _re.compile(r'1[3-9]\d{9}')
        email_pattern = _re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

        if id_pattern.search(generated_text):
            issues.append("包含疑似身份证号")
            score -= 5.0
        if phone_pattern.search(generated_text):
            issues.append("包含疑似手机号码")
            score -= 3.0
        if email_pattern.search(generated_text):
            issues.append("包含疑似电子邮箱")
            score -= 2.0

        score = max(0.0, score)
        explanation = "; ".join(issues) if issues else "未发现合规性问题"

        return DimensionScore(
            dimension=EvaluationDimension.COMPLIANCE,
            score=score,
            explanation=explanation,
            details={"issues": issues},
        )

    @staticmethod
    def _merge_compliance_scores(
        llm_score: Optional[DimensionScore],
        rule_score: DimensionScore,
    ) -> DimensionScore:
        """合并 LLM 合规分与规则合规分：分数取较低者，issues 合并去重"""
        if llm_score is None:
            return rule_score

        rule_issues = rule_score.details.get("issues", []) if rule_score.details else []
        llm_issues = llm_score.details.get("issues", []) if llm_score.details else []
        merged_issues: List[str] = []
        for item in rule_issues + llm_issues:
            if item and item not in merged_issues:
                merged_issues.append(item)

        merged_score = min(llm_score.score, rule_score.score)
        if merged_issues:
            explanation = "; ".join(merged_issues)
        else:
            explanation = llm_score.explanation or rule_score.explanation

        return DimensionScore(
            dimension=EvaluationDimension.COMPLIANCE,
            score=merged_score,
            explanation=explanation,
            details={"issues": merged_issues},
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
