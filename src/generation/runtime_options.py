"""
生成链路运行参数：统一字数档位映射、长文超时预算与评估参数。
"""

from __future__ import annotations

from typing import Any, Dict

from .chapter_budget import legacy_maps_for_single_segment


def single_segment_generation_config(length_bucket: str) -> Dict[str, Any]:
    """
    单段模式生成参数（与 length_bucket 对齐）。
    """
    length_hint, max_tokens = legacy_maps_for_single_segment(length_bucket)
    return {
        "length_hint": length_hint,
        "max_tokens": max_tokens,
    }


def estimate_long_form_generation_timeout(chapter_count: int) -> float:
    """
    长文分章生成超时预算（秒）。
    """
    n = max(1, int(chapter_count))
    return 60.0 + 120.0 * n


def estimate_long_form_evaluation_timeout(chapter_count: int) -> float:
    """
    长文评估（段级 + 可选事实检查）超时预算（秒）。
    """
    n = max(1, int(chapter_count))
    return 120.0 + 30.0 * n


def build_long_form_eval_options(
    *,
    llm_adapter: Any,
    use_rule_decompose: bool,
    enable_fact_check: bool = True,
    use_llm_eval: bool = False,
    max_atomic_facts_per_segment: int = 12,
    fact_check_timeout_per_segment: float = 45.0,
    batch_size: int = 5,
) -> Dict[str, Any]:
    """
    统一 evaluate_long_form 的常用参数，避免 Web/API 分散硬编码。
    """
    return {
        "llm_adapter": llm_adapter,
        "use_llm_eval": use_llm_eval,
        "enable_fact_check": enable_fact_check,
        "max_atomic_facts_per_segment": max_atomic_facts_per_segment,
        "fact_check_timeout_per_segment": fact_check_timeout_per_segment,
        "use_rule_decompose": use_rule_decompose,
        "batch_size": batch_size,
    }
