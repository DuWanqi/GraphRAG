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
from .factscore_adapter import (
    FActScoreChecker,
    FactCheckResult,
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
    "FActScoreChecker",
    "FactCheckResult",
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
