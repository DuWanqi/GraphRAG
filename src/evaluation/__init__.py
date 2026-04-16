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
    MetricResult,
    calculate_all_metrics,
    aggregate_scores,
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

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "DimensionScore",
    "EvaluationDimension",
    "BatchEvaluator",
    "AccuracyMetrics",
    "RelevanceMetrics",
    "LiteraryMetrics",
    "MetricResult",
    "calculate_all_metrics",
    "aggregate_scores",
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
]
