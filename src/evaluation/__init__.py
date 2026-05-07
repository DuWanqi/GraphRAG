"""评估模块"""
from .evaluator import (
    Evaluator,
    EvaluationResult,
    DimensionScore,
    EvaluationDimension,
    BatchEvaluator,
)
from .metrics import (
    AccuracyMetrics,
    RelevanceMetrics,
    LiteraryMetrics,
    CrossChapterMetrics,
    MetricResult,
    calculate_all_metrics,
    aggregate_scores,
)
from .novel_content_metrics import (
    NovelContentAnalysis,
    analyze_novel_content,
    information_gain_metric,
    expansion_grounding_metric,
)
from .factscore_adapter import (
    FActScoreChecker,
    FactCheckResult,
)
from .safe_checker import (
    SAFEFactChecker,
    SAFECheckResult,
)
from .retrieval_benchmark import (
    RetrievalBenchmark,
    TestCase,
    RetrievalMetrics,
    BenchmarkResult,
    run_benchmark,
    evaluate_retrieval_quality,
)
from .long_form_eval import (
    evaluate_long_form,
    LongFormEvalResult,
    SegmentEvalRecord,
    long_form_eval_to_json,
    document_year_diversity,
)
from .quality_gate import (
    QualityGateResult,
    QualityThresholds,
    ChapterGateResult,
    ChapterIssue,
    CrossChapterIssue,
    RemediationPlan,
    check_quality_gate,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "DimensionScore",
    "EvaluationDimension",
    "BatchEvaluator",
    "AccuracyMetrics",
    "RelevanceMetrics",
    "LiteraryMetrics",
    "CrossChapterMetrics",
    "MetricResult",
    "calculate_all_metrics",
    "aggregate_scores",
    "NovelContentAnalysis",
    "analyze_novel_content",
    "information_gain_metric",
    "expansion_grounding_metric",
    "FActScoreChecker",
    "FactCheckResult",
    "SAFEFactChecker",
    "SAFECheckResult",
    "RetrievalBenchmark",
    "TestCase",
    "RetrievalMetrics",
    "BenchmarkResult",
    "run_benchmark",
    "evaluate_retrieval_quality",
    "evaluate_long_form",
    "LongFormEvalResult",
    "SegmentEvalRecord",
    "long_form_eval_to_json",
    "document_year_diversity",
    "QualityGateResult",
    "QualityThresholds",
    "ChapterGateResult",
    "ChapterIssue",
    "CrossChapterIssue",
    "RemediationPlan",
    "check_quality_gate",
]
