"""
质量门控 (Quality Gate)：对分章生成结果执行可配置的阈值检查，
输出通过/不通过判定及可执行的修复建议。

使用场景：
- 生成完成后，调用 check_quality_gate() 判断是否达到交付标准
- 对未通过的章节，返回 RemediationPlan 指导自动或人工修复
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# 阈值配置
# ---------------------------------------------------------------------------

@dataclass
class QualityThresholds:
    """可配置的质量阈值。"""
    min_segment_score: float = 5.0        # 段级综合分下限 (0-10)
    max_cross_repetition: float = 0.20    # 跨章 n-gram 重叠率上限
    min_fact_score: float = 0.60          # FActScore 最低比率
    min_length_ratio: float = 0.40        # 实际字数 / 目标字数 下限
    max_length_ratio: float = 2.5         # 实际字数 / 目标字数 上限
    max_summary_sentence_ratio: float = 0.30  # 总结性语句占比上限
    min_semantic_similarity: Optional[float] = None  # 语义相似度下限（可选）
    min_expansion_grounding: float = 0.40  # 扩展内容溯源率下限
    min_entity_coverage: float = 0.80     # 实体覆盖率下限
    min_rag_utilization: float = 0.6      # RAG 利用率下限（≥2个实体）

    @classmethod
    def for_expansion_task(cls) -> "QualityThresholds":
        """为扩展任务优化的阈值配置"""
        return cls(
            min_segment_score=5.0,
            max_cross_repetition=0.20,
            min_fact_score=0.60,
            min_semantic_similarity=0.15,  # 扩展任务语义相似度要求较低
            min_expansion_grounding=0.40,  # 更现实的溯源率要求
            min_entity_coverage=0.80,  # 基于新的指标定义
            min_length_ratio=0.40,
            max_length_ratio=2.5,
            max_summary_sentence_ratio=0.30,
            min_rag_utilization=0.6,  # 要求至少使用2个RAG实体
        )


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class ChapterIssue:
    """单章质量问题。"""
    dimension: str       # length | repetition | fact | style | summary
    severity: str        # "warning" | "error"
    message: str
    suggestion: str      # 可执行的修复建议


@dataclass
class ChapterGateResult:
    """单章门控结果。"""
    chapter_index: int
    passed: bool
    issues: List[ChapterIssue] = field(default_factory=list)


@dataclass
class CrossChapterIssue:
    """跨章质量问题。"""
    dimension: str
    severity: str
    chapters_involved: List[int]
    message: str
    suggestion: str


@dataclass
class RemediationPlan:
    """修复计划：告诉调用方需要对哪些章做什么操作。"""
    chapters_to_regenerate: List[int]
    reasons: Dict[int, List[str]]          # chapter_index → 需要修复的问题
    prompt_adjustments: Dict[int, str]     # chapter_index → 建议的 prompt 补充指令


@dataclass
class QualityGateResult:
    """整体门控结果。"""
    passed: bool
    overall_score: float
    chapter_results: List[ChapterGateResult]
    cross_chapter_issues: List[CrossChapterIssue]
    remediation: Optional[RemediationPlan]

    def to_text(self) -> str:
        lines = [
            f"质量门控: {'通过' if self.passed else '未通过'}  综合分: {self.overall_score:.2f}",
        ]
        for cr in self.chapter_results:
            status = "✓" if cr.passed else "✗"
            lines.append(f"  第{cr.chapter_index + 1}章 {status}")
            for iss in cr.issues:
                lines.append(f"    [{iss.severity}] {iss.dimension}: {iss.message}")
                lines.append(f"      → 建议: {iss.suggestion}")
        if self.cross_chapter_issues:
            lines.append("  跨章问题:")
            for ci in self.cross_chapter_issues:
                lines.append(f"    [{ci.severity}] {ci.dimension}: {ci.message}")
                lines.append(f"      → 建议: {ci.suggestion}")
        if self.remediation and self.remediation.chapters_to_regenerate:
            lines.append(f"  需重新生成: 第 {', '.join(str(c+1) for c in self.remediation.chapters_to_regenerate)} 章")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 检测函数
# ---------------------------------------------------------------------------

# 总结性语句模式
_SUMMARY_PATTERNS = re.compile(
    r"(?:总之|综上|总而言之|总的来说|总体而言|回顾|纵观|概括地说|简而言之|"
    r"由此可见|可以看出|不难看出|归根结底|一言以蔽之)"
)

# 重复性套话模式
_BOILERPLATE_PATTERNS = re.compile(
    r"(?:在这个大背景下|在这样的时代背景下|在这种情况下|在这一时期|"
    r"正是在这样的|值得一提的是|不可否认的是|众所周知)"
)

# 章末抒情感悟/人生哲理式收尾模式（"软性总结"）
_EPILOGUE_PATTERNS = re.compile(
    r"(?:这(?:段|些)(?:日子|岁月|经历|时光).*?(?:教会|让我|使我|改变|影响|塑造|成就|成长|明白|懂得|铭记|难忘)|"
    r"(?:那|这)是.*?(?:最(?:美好|难忘|珍贵|宝贵|深刻|温暖)的|一生中)|"
    r"(?:见证|承载|记录)了.*?(?:成长|变迁|变化|岁月|青春|我的)|"
    r"(?:在我|在那|从此).*?(?:心中|心底|记忆里|脑海中).*?(?:留下|刻下|种下|埋下|烙下|扎根)|"
    r"(?:多年以后|许多年后|后来我才).*?(?:才(?:明白|懂得|知道|理解)|回想|回忆|想起)|"
    r"(?:也许|或许).*?正是.*?(?:成就|塑造|造就|奠定)|"
    r"(?:这一切|所有这些).*?(?:构成|汇成|编织成|成为).*?(?:底色|篇章|画卷|一部分))"
)


def _detect_summary_sentences(text: str) -> float:
    """检测文本中总结性语句的比率。"""
    sentences = [s.strip() for s in re.split(r"[。！？]", text) if s.strip()]
    if not sentences:
        return 0.0
    summary_count = sum(1 for s in sentences if _SUMMARY_PATTERNS.search(s))
    return summary_count / len(sentences)


def _detect_boilerplate(text: str) -> List[str]:
    """检测套话。"""
    return _BOILERPLATE_PATTERNS.findall(text)


def _detect_epilogue(text: str, tail_sentences: int = 3) -> Optional[str]:
    """
    检测章节末尾是否有抒情性感悟/哲理式收尾（"软性总结"）。

    只检查末尾 tail_sentences 句；若匹配则返回匹配到的原句片段，
    否则返回 None。
    """
    sentences = [s.strip() for s in re.split(r"[。！？]", text) if s.strip()]
    if not sentences:
        return None
    for s in sentences[-tail_sentences:]:
        m = _EPILOGUE_PATTERNS.search(s)
        if m:
            return s[:60]
    return None


def _cross_chapter_ngram_overlap(
    chapters: List[str],
    n: int = 6,
) -> List[Dict[str, Any]]:
    """
    计算每对相邻章节的 n-gram 重叠率。

    Returns:
        列表，每项包含 (i, j, overlap_ratio, sample_overlaps)
    """
    def _ngrams(text: str, ng: int) -> Set[str]:
        clean = re.sub(r"\s+", "", text)
        if len(clean) < ng:
            return set()
        return {clean[k:k + ng] for k in range(len(clean) - ng + 1)}

    results = []
    for i in range(len(chapters) - 1):
        ng_a = _ngrams(chapters[i], n)
        ng_b = _ngrams(chapters[i + 1], n)
        if not ng_a or not ng_b:
            results.append({"i": i, "j": i + 1, "ratio": 0.0, "samples": []})
            continue
        overlap = ng_a & ng_b
        ratio = len(overlap) / min(len(ng_a), len(ng_b))
        samples = sorted(overlap, key=len, reverse=True)[:5]
        results.append({"i": i, "j": i + 1, "ratio": ratio, "samples": samples})
    return results


# ---------------------------------------------------------------------------
# 门控主函数
# ---------------------------------------------------------------------------

def check_quality_gate(
    chapters_content: List[str],
    *,
    segment_scores: Optional[List[float]] = None,
    fact_scores: Optional[List[Optional[float]]] = None,
    target_chars_per_chapter: Optional[List[int]] = None,
    segment_metrics: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[QualityThresholds] = None,
) -> QualityGateResult:
    """
    对分章生成结果执行质量门控检查。

    Args:
        chapters_content: 每章生成的文本列表
        segment_scores: 每章的评估综合分 (0-10)，可选
        fact_scores: 每章的 FActScore (0-1)，可选
        target_chars_per_chapter: 每章目标字数，可选
        segment_metrics: 每章的详细指标字典，可选
        thresholds: 阈值配置
    """
    th = thresholds or QualityThresholds()
    n = len(chapters_content)

    chapter_results: List[ChapterGateResult] = []
    cross_issues: List[CrossChapterIssue] = []

    # ---- 1) 逐章检查 ----
    for idx, content in enumerate(chapters_content):
        issues: List[ChapterIssue] = []

        # 1a) 字数检查
        if target_chars_per_chapter and idx < len(target_chars_per_chapter):
            target = target_chars_per_chapter[idx]
            actual = len(content)
            ratio = actual / max(target, 1)
            if ratio < th.min_length_ratio:
                issues.append(ChapterIssue(
                    "length", "error",
                    f"字数 {actual} 远低于目标 {target} (比率 {ratio:.1%})",
                    f"增加 length_hint 或降低 temperature 以获得更充分的输出",
                ))
            elif ratio > th.max_length_ratio:
                issues.append(ChapterIssue(
                    "length", "warning",
                    f"字数 {actual} 远超目标 {target} (比率 {ratio:.1%})",
                    f"缩短 max_tokens 或在 prompt 中强调字数限制",
                ))

        # 1b) 综合分检查
        if segment_scores and idx < len(segment_scores):
            score = segment_scores[idx]
            if score < th.min_segment_score:
                issues.append(ChapterIssue(
                    "score", "error",
                    f"综合评分 {score:.1f} 低于阈值 {th.min_segment_score}",
                    "检查检索结果质量或切换 LLM provider",
                ))

        # 1c) 事实检查
        if fact_scores and idx < len(fact_scores) and fact_scores[idx] is not None:
            fs = fact_scores[idx]
            if fs < th.min_fact_score:
                issues.append(ChapterIssue(
                    "fact", "error",
                    f"FActScore {fs:.1%} 低于阈值 {th.min_fact_score:.0%}",
                    "检索到的背景信息不足或 LLM 产生了幻觉，建议增加 top_k 或切换检索模式",
                ))

        # 1d) RAG 利用率检查
        if segment_metrics and idx < len(segment_metrics):
            metrics = segment_metrics[idx]
            if "rag_utilization" in metrics:
                rag_util = metrics["rag_utilization"]
                if hasattr(rag_util, "value"):
                    rag_util_value = rag_util.value
                else:
                    rag_util_value = rag_util

                if rag_util_value < th.min_rag_utilization:
                    issues.append(ChapterIssue(
                        "rag_utilization", "error",
                        f"RAG 利用率 {rag_util_value:.1%} 低于阈值 {th.min_rag_utilization:.0%}",
                        "增加 top_k 以获取更多相关实体，或在 prompt 中明确要求使用检索到的背景知识",
                    ))

        # 注：幻觉检测指标仅用于评分，不设置门控

        # 1f) 总结性语句检查
        summary_ratio = _detect_summary_sentences(content)
        if summary_ratio > th.max_summary_sentence_ratio:
            issues.append(ChapterIssue(
                "summary", "warning",
                f"总结性语句占比 {summary_ratio:.0%} 偏高",
                "在 prompt 中明确要求「不要总结，只做叙事性描写」",
            ))

        # 1g) 套话检查
        boilerplate = _detect_boilerplate(content)
        if len(boilerplate) >= 2:
            issues.append(ChapterIssue(
                "style", "warning",
                f"检测到 {len(boilerplate)} 处套话: {', '.join(boilerplate[:3])}",
                "在 prompt 中要求「避免使用空泛的过渡语」",
            ))

        # 1h) 非末章软性感悟结尾检查
        is_last_chapter = (idx == n - 1)
        if not is_last_chapter:
            epilogue_hit = _detect_epilogue(content)
            if epilogue_hit:
                issues.append(ChapterIssue(
                    "epilogue", "warning",
                    f"非末章出现感悟式收尾: 「{epilogue_hit}」",
                    "在 prompt 中要求章节在叙事自然节点戛然而止，不要添加抒情性感悟",
                ))

        passed = not any(iss.severity == "error" for iss in issues)
        chapter_results.append(ChapterGateResult(idx, passed, issues))

    # ---- 2) 跨章检查 ----
    if n >= 2:
        overlaps = _cross_chapter_ngram_overlap(chapters_content, n=6)
        for ov in overlaps:
            if ov["ratio"] > th.max_cross_repetition:
                cross_issues.append(CrossChapterIssue(
                    "repetition", "error",
                    [ov["i"], ov["j"]],
                    f"第{ov['i']+1}章与第{ov['j']+1}章 6-gram 重叠率 {ov['ratio']:.1%}",
                    "在后一章 prompt 中注入前一章概要并要求不重复",
                ))

    # ---- 3) 汇总 ----
    all_passed = all(cr.passed for cr in chapter_results) and \
                 not any(ci.severity == "error" for ci in cross_issues)

    # 综合分
    if segment_scores:
        valid_scores = [s for s in segment_scores if s is not None]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    else:
        overall = 0.0

    # 修复计划
    remediation = _build_remediation(chapter_results, cross_issues)

    return QualityGateResult(
        passed=all_passed,
        overall_score=overall,
        chapter_results=chapter_results,
        cross_chapter_issues=cross_issues,
        remediation=remediation,
    )


def _build_remediation(
    chapter_results: List[ChapterGateResult],
    cross_issues: List[CrossChapterIssue],
) -> Optional[RemediationPlan]:
    """根据检查结果生成修复计划。"""
    to_regen: List[int] = []
    reasons: Dict[int, List[str]] = {}
    adjustments: Dict[int, str] = {}

    for cr in chapter_results:
        if not cr.passed:
            to_regen.append(cr.chapter_index)
            reasons[cr.chapter_index] = [iss.message for iss in cr.issues if iss.severity == "error"]
            # 拼接所有建议作为 prompt 调整
            suggestions = [iss.suggestion for iss in cr.issues if iss.severity == "error"]
            if suggestions:
                adjustments[cr.chapter_index] = "；".join(suggestions)

    for ci in cross_issues:
        if ci.severity == "error":
            # 建议重新生成后一章
            later = max(ci.chapters_involved)
            if later not in to_regen:
                to_regen.append(later)
            reasons.setdefault(later, []).append(ci.message)
            adjustments.setdefault(later, "")
            adjustments[later] += f"；{ci.suggestion}"

    if not to_regen:
        return None

    return RemediationPlan(
        chapters_to_regenerate=sorted(set(to_regen)),
        reasons=reasons,
        prompt_adjustments=adjustments,
    )
