"""
记忆图谱 Web 应用
基于 Gradio 构建的用户界面
"""

import asyncio
import copy
import csv
import json
import logging
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import html
import gradio as gr

from src.config import get_settings
from src.config.workspace_logging import setup_workspace_logging
from src.llm import (
    create_llm_adapter,
    get_available_providers,
    get_provider_models,
    LLMRouter,
)
from src.indexing import GraphBuilder
from src.retrieval import MemoirRetriever, PlainVectorRAGRetriever, PLAIN_VECTOR_RAG_MODE
from src.generation import (
    LiteraryGenerator,
    PromptTemplates,
    segment_memoir,
    run_long_form_generation,
    regenerate_chapters,
    single_segment_generation_config_from_range,
    estimate_long_form_generation_timeout,
    estimate_long_form_evaluation_timeout,
    build_long_form_eval_options,
)
from src.evaluation import (
    Evaluator,
    FActScoreChecker,
    SAFECheckResult,
    aggregate_scores,
    calculate_all_metrics,
    evaluate_retrieval_quality,
    evaluate_long_form,
    long_form_eval_to_json,
    QualityThresholds,
)


setup_workspace_logging()


logger = logging.getLogger(__name__)

# 全局变量
settings = get_settings()
retriever: Optional[MemoirRetriever] = None
plain_rag_retriever: Optional[PlainVectorRAGRetriever] = None
generator: Optional[LiteraryGenerator] = None
current_provider: Optional[str] = None  # 跟踪当前使用的 provider
current_model: Optional[str] = None  # 跟踪当前使用的模型名（与各供应商共用）


def init_components(provider: str = "hunyuan", model: Optional[str] = None):
    """初始化组件"""
    global retriever, plain_rag_retriever, generator, current_provider, current_model

    # 如果 provider / model 相同且组件已初始化，则跳过
    if (
        provider == current_provider
        and model == current_model
        and retriever is not None
        and plain_rag_retriever is not None
        and generator is not None
    ):
        return f"✅ 已使用 {provider}{f'/{model}' if model else ''} 模型"

    try:
        llm_adapter = create_llm_adapter(provider=provider, model=model)
        retriever = MemoirRetriever(llm_adapter=llm_adapter)
        plain_rag_retriever = PlainVectorRAGRetriever()
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        current_provider = provider
        current_model = model
        return f"✅ 初始化成功！使用 {provider}{f'/{model}' if model else ''} 模型"
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"


def _make_llm(provider, model):
    """创建 LLM 适配器的快捷方式"""
    return create_llm_adapter(provider=provider, model=model)


def _is_plain_vector_rag_mode(retrieval_mode: str) -> bool:
    return retrieval_mode == PLAIN_VECTOR_RAG_MODE


def _get_retriever_for_mode(retrieval_mode: str):
    global plain_rag_retriever
    if _is_plain_vector_rag_mode(retrieval_mode):
        if plain_rag_retriever is None:
            plain_rag_retriever = PlainVectorRAGRetriever()
        return plain_rag_retriever
    return retriever


def _format_retrieval_details(retrieval_result, retrieval_mode: str) -> str:
    """Format retrieval details for the Gradio retrieval panel."""
    if _is_plain_vector_rag_mode(retrieval_mode):
        meta = getattr(retrieval_result, "_plain_rag_meta", {}) or {}
        md = (
            "**检索后端**: 普通 RAG（向量 baseline）\n"
            f"**知识库路径**: `{meta.get('input_dir', settings.plain_rag_input_dir)}`\n"
            f"**Embedding**: `{meta.get('embedding_backend', settings.plain_rag_embedding_backend)}` / "
            f"`{meta.get('embedding_model', settings.plain_rag_embedding_model)}`\n"
            f"**索引 chunks**: {meta.get('chunk_count', 0)}  "
            f"**返回 chunks**: {len(retrieval_result.text_units)}\n"
        )
        top_scores = meta.get("top_scores") or []
        if top_scores:
            md += "\n**Top chunks**:\n"
            for item in top_scores[:3]:
                title = item.get("title") or item.get("source_file", "")
                md += (
                    f"- #{item.get('rank')} `{item.get('source_file')}` "
                    f"chunk={item.get('chunk_index')} score={float(item.get('score', 0.0)):.4f}"
                    f"{f' · {title}' if title else ''}\n"
                )
        if retrieval_result.text_units:
            md += "\n**片段预览**:\n"
            for i, text in enumerate(retrieval_result.text_units[:3], 1):
                preview = str(text).replace("\n", " ")[:180]
                md += f"{i}. {preview}...\n"
        return md

    if retrieval_mode == "vector":
        backend_label = "GraphRAG Local Search（实体向量 + 关系/文本/社区扩展）"
    elif retrieval_mode == "hybrid":
        backend_label = "GraphRAG Hybrid（Local Search + 关键词补充）"
    else:
        backend_label = "GraphRAG 关键词检索"

    md = (
        f"**检索后端**: {backend_label}\n"
        f"**找到实体**: {len(retrieval_result.entities)} 个  "
        f"**找到关系**: {len(retrieval_result.relationships)} 个  "
        f"**社区报告**: {len(retrieval_result.communities)} 个  "
        f"**相关文本**: {len(retrieval_result.text_units)} 段"
    )
    if retrieval_result.entities:
        md += "\n\n**主要实体**:\n"
        for entity in retrieval_result.entities[:5]:
            md += f"- {entity.get('name', '未知')}: {entity.get('description', '')[:100]}...\n"
    return md


def _format_retrieval_quality(eval_result):
    """格式化检索质量评估结果为 Markdown"""
    md = f"**nDCG@3**: {eval_result['ndcg_at_3']:.4f}  "
    md += f"**nDCG@5**: {eval_result['ndcg_at_5']:.4f}  "
    md += f"**nDCG@10**: {eval_result['ndcg_at_10']:.4f}\n\n"
    md += f"**MRR**: {eval_result['mrr']:.4f}\n"
    if eval_result.get("relevance_vector"):
        vector = ", ".join(f"{v:.0f}" for v in eval_result["relevance_vector"])
        md += f"\n**LLM相关性向量**: [{vector}]（0-3；MRR按首个 >=2 的文档计算）\n"
    if eval_result["per_doc_scores"]:
        md += "\n| # | 文档摘要 | 相关性 | 理由 |\n|---|----------|--------|------|\n"
        for item in eval_result["per_doc_scores"][:10]:
            stars = int(item["score"])
            score_bar = "\u2605" * stars + "\u2606" * (3 - stars)
            md += f"| {item['doc_id']} | {item['snippet'][:30]} | {score_bar} ({item['score']:.0f}/3) | {item['reason'][:30]} |\n"
    return md


def _format_accuracy(fact_result, gen_time):
    """格式化事实准确性结果为 Markdown"""
    status_icon = "\u2705" if fact_result.is_factual else "\u26a0\ufe0f"
    md = f"### {status_icon} FActScore\n\n"
    if fact_result.total_facts > 0:
        md += f"**FActScore**: {fact_result.factscore:.1%} ({fact_result.supported_facts}/{fact_result.total_facts} 事实被支持)\n\n"
    md += f"**一致性判定**: {'事实一致' if fact_result.is_factual else '存在潜在问题'}\n\n"
    # 置信度、实体覆盖率、证据支持度为规则计算，不展示
    md += f"**总结**: {fact_result.summary}\n"
    return md


def _format_dimension(dim_score):
    """格式化单个评估维度为 Markdown"""
    md = f"**评分**: {dim_score.score:.1f} / 10\n\n"
    md += f"**评价**: {dim_score.explanation}\n"
    return md


def _format_compliance(dim_score):
    """格式化合规性结果为 Markdown"""
    icon = "\u2705" if dim_score.score >= 8 else "\u26a0\ufe0f"
    md = f"### {icon} 合规检查\n\n"
    md += f"**合规评分**: {dim_score.score:.1f} / 10\n\n"
    md += f"**结果**: {dim_score.explanation}\n"
    return md


def _format_safe_check(safe_result: SAFECheckResult) -> str:
    """格式化 SAFE 独立知识验证结果为 Markdown"""
    icon = "\u2705" if safe_result.safe_score >= 0.7 else "\u26a0\ufe0f"
    md = f"### {icon} SAFE 独立验证\n\n"
    md += f"**SAFE Score**: {safe_result.safe_score:.1%} ({safe_result.supported_facts}/{safe_result.total_facts} 事实被支持)\n\n"
    if safe_result.unsupported_facts:
        md += f"**不支持**: {safe_result.unsupported_facts} 条\n"
    if safe_result.irrelevant_facts:
        md += f"**无法判断**: {safe_result.irrelevant_facts} 条\n"
    md += "\n"

    # KB 对比
    if safe_result.kb_comparison:
        cmp = safe_result.kb_comparison
        md += "---\n#### 知识库对比\n\n"
        md += f"| 指标 | KB FActScore | SAFE Score |\n"
        md += f"|------|-------------|------------|\n"
        md += f"| 得分 | {cmp['kb_factscore']:.1%} | {cmp['safe_score']:.1%} |\n\n"
        md += f"**评估**: {cmp['assessment']}\n\n"

    # 不支持的事实详情
    unsupported = [d for d in safe_result.fact_details if d.get("verdict") == "NOT_SUPPORTED"]
    if unsupported:
        md += "---\n#### 不支持的事实\n\n"
        for d in unsupported[:5]:
            md += f"- {d.get('fact', '')[:80]}  \n  *{d.get('explanation', '')}*\n"

    return md


METRIC_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "temporal_coherence": {
        "name": "时间一致性",
        "definition": "生成文本中出现的年份必须全部在原文年份白名单内，防止 LLM 捏造年份。",
        "formula": "合规年份数 / 生成年份总数；无生成年份时得满分",
    },
    "topic_coherence": {
        "name": "主题一致性",
        "definition": "生成内容与回忆录主题的词汇/语义重叠程度（规则度量）。",
        "formula": "关键词与回忆录重叠度等指标的组合规则评分",
    },
    "length_score": {
        "name": "长度得分",
        "definition": "生成长度是否在允许区间及最佳区间附近（随任务与 length_hint 动态校准）。",
        "formula": "区间外惩罚 + 区间内距最优区间距离的分段打分",
    },
    "transition_usage": {
        "name": "衔接词使用",
        "definition": "叙事过渡自然度：时间过渡、场景切换、叙事连贯性（LLM rubric 评分）。",
        "formula": "LLM-as-a-judge 单次推理，按 rubric 打分（0.0-1.0）",
    },
    "descriptive_richness": {
        "name": "描写丰富度",
        "definition": "感官描写、情感表达、比喻修辞、细节具体度的综合丰富程度（LLM rubric 评分）。",
        "formula": "LLM-as-a-judge 单次推理，按 rubric 打分（0.0-1.0）",
    },
    "rag_utilization": {
        "name": "RAG 利用率",
        "definition": "生成文本实际使用了多少 RAG 检索到的新实体（原文未提及的）。",
        "formula": "按使用的新实体个数分段：0→0.0；1→0.5；2→0.7；3→0.8；4+→1.0",
    },
    "hallucination": {
        "name": "幻觉风险控制",
        "definition": "从生成文本抽出的专有实体中缺乏回忆录与 RAG 支撑的比例越低越好。",
        "formula": "1 - （无支撑实体数 / 抽取实体总数）",
    },
    "year_diversity": {
        "name": "年代跨度（篇级）",
        "definition": "合并长文中检出年代种类的多样性启发式 penalize。",
        "formula": "规则：检出年份种类越少越稳定；过多年代略扣分",
    },
    "merged_length": {
        "name": "合并篇幅（篇级）",
        "definition": "合并全文总长是否落在合理长篇区间附近。",
        "formula": "基于 min/max/optimal 区间的 length_score 变体",
    },
    "inter_chapter_repetition": {
        "name": "跨章重复度",
        "definition": "相邻章节 n-gram 重叠程度（越低越好，分数已转为质量分）。",
        "formula": "基于跨章 jaccard/overlap 的规则指标",
    },
    "style_consistency": {
        "name": "风格一致性",
        "definition": "各章句式、标点的统计分布相似度近似。",
        "formula": "篇章间标点/句式特征向量相似度规则分",
    },
    "summary_sentence_ratio": {
        "name": "总结句比率",
        "definition": "章末总结/套话式句子占比启发式检测。",
        "formula": "匹配总结句式数量 / 分句总数",
    },
}


def _metric_abbr_tip(metric_key: str) -> str:
    meta = METRIC_DEFINITIONS.get(metric_key, {})
    name = meta.get("name", metric_key)
    definition = meta.get("definition", "")
    formula = meta.get("formula", "")
    tip = f"{name}: {definition}｜计算: {formula}"
    return html.escape(tip.replace("\n", " "), quote=True)


def _format_metric_line(metric_key: str, mr: Any, value_fmt: str = "{:.3f}") -> str:
    meta = METRIC_DEFINITIONS.get(metric_key, {})
    display = meta.get("name", metric_key)
    mv = getattr(mr, "max_value", 1.0) or 1.0
    val = getattr(mr, "value", 0.0)
    expl = getattr(mr, "explanation", "") or ""
    tip = _metric_abbr_tip(metric_key)
    line = (
        f"- **{display}** "
        f'<abbr title="{tip}">\u2139\uFE0F</abbr> '
        f"`{value_fmt.format(val)}` / `{mv:.3f}`  "
        f"（归一约 **{100.0 * val / mv:.1f}%**）"
    )
    if expl.strip():
        line += f"\n  \n*{expl.strip()}*"
    return line


def _entity_names_sorted_by_length(rr: Any) -> List[str]:
    names: List[str] = []
    for e in rr.entities or []:
        n = e.get("name") or e.get("title") or ""
        if n:
            names.append(n)
    seen: Set[str] = set()
    uniq: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return sorted(uniq, key=len, reverse=True)


def _bold_entities_in_text(text: str, names: List[str]) -> str:
    if not text or not names:
        return text
    out = text
    for name in names:
        if len(name) < 2:
            continue
        if name not in out:
            continue
        out = out.replace(name, f"**{name}**")
        out = out.replace(f"****{name}****", f"**{name}**")
    out = out.replace("****", "**")
    return out


def _format_chapter_entities_block(
    idx: int,
    chapter: Any,
    novel_used: Optional[Set[str]] = None,
) -> List[str]:
    lines: List[str] = []
    rr = chapter.retrieval_result
    ctx = rr.context
    lines.append(f"### 第 {idx + 1} 章检索")
    lines.append(
        f"- **解析时间**: {ctx.year or '—'}｜**地点**: {ctx.location or '—'}\n"
        f"- **实体数**: {len(rr.entities or [])}｜**关系数**: {len(rr.relationships or [])}"
    )
    plain_meta = getattr(rr, "_plain_rag_meta", None)
    if plain_meta:
        lines.append(
            f"- **普通向量 RAG chunks**: {len(rr.text_units or [])} / "
            f"{plain_meta.get('chunk_count', 0)}"
        )
        for j, text in enumerate((rr.text_units or [])[:3], 1):
            preview = str(text).replace("\n", " ")[:160]
            lines.append(f"  {j}. {preview}...")

    brief = getattr(rr, "_novel_content_brief", None)
    nu = novel_used or set()

    if brief:
        lines.append("")
        lines.append(f"- **对齐实体**（回忆录已提及，约 {len(brief.aligned_entities)} 个）")
        for ent in brief.aligned_entities[:20]:
            name = ent.get("name") or ent.get("title") or "?"
            et = ent.get("type") or ""
            lines.append(f"  - `{name}`" + (f"（{et}）" if et else ""))
        if len(brief.aligned_entities) > 20:
            lines.append(f"  - … 共 {len(brief.aligned_entities)} 个，其余略")

        lines.append("")
        lines.append(f"- **新增实体**（RAG 提供、回忆录未提，约 {len(brief.novel_entities)} 个）")
        for ent in brief.novel_entities[:25]:
            name = ent.get("name") or ent.get("title") or "?"
            et = ent.get("type") or ""
            used = name in nu
            line = f"  - "
            line += (f"**`{name}`**" if used else f"`{name}`")
            line += (f"（{et}）" if et else "") + (" ✓ 已写入生成文" if used else "")
            lines.append(line)
        if len(brief.novel_entities) > 25:
            lines.append(f"  - … 共 {len(brief.novel_entities)} 个，其余略")
    else:
        lines.append("")
        lines.append("- **检索实体（摘要）**")
        for ent in (rr.entities or [])[:12]:
            name = ent.get("name") or ent.get("title") or "?"
            et = ent.get("type") or ""
            lines.append(f"  - `{name}`" + (f"（{et}）" if et else ""))

    rels = rr.relationships or []
    if rels:
        lines.append("")
        lines.append(f"- **关系**（{len(rels)}）")
        for j, rel in enumerate(rels[:15], 1):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            desc = rel.get("description", "")
            line = f"  {j}. `{src}` \u2192 `{tgt}`"
            if desc:
                desc_more = "\u2026" if len(desc) > 80 else ""
                line += f" — _{desc[:80]}{desc_more}_"
            lines.append(line)
        if len(rels) > 15:
            lines.append(f"  - … 其余 {len(rels) - 15} 条略")

    return lines


def _format_long_form_facts_and_rules(ev: Any) -> str:
    parts: List[str] = []
    parts.append("## \u6458\u8981\n\n")
    parts.append(ev.summary_text)
    parts.append("\n\n---\n## \u6BB5\u843D\u7EA7 \u00B7 \u89C4\u5219\u4E0E\u51C6\u786E\u6027\u6307\u6807\n")

    ACC_KEYS = (
        "temporal_coherence",
        "rag_utilization",
        "hallucination",
    )

    for r in ev.segments:
        parts.append(f"\n### \u6BB5 {r.segment_index + 1}\n")

        fc = getattr(r, "fact_check", None)
        skip = getattr(r, "fact_check_skipped_reason", None)
        if fc is not None:
            icon = "\u2705" if fc.is_factual else "\u26a0\uFE0F"
            verdict_cn = "\u4e8b\u5b9e\u4e00\u81f4\u503e\u5411" if fc.is_factual else "\u6709\u5f85\u590d\u6838"
            parts.append(f"#### {icon} FActScore\uff08\u6BB5\u843D\uff09\n\n")
            parts.append(
                f"- **\u5F97\u5206**: {fc.factscore:.1%}\n"
                f"- **\u5224\u5B9A**: {verdict_cn}\n"
                f"- **\u6458\u8981**: {fc.summary}\n"
            )
        elif skip:
            parts.append(f"_\u672C\u6BB5\u672A\u8DD1\u901A\u4E8B\u5B9E\u6838\u67E5: `{skip}`_\n")
        else:
            parts.append("_\u672C\u6BB5\u4E8B\u5B9E\u6838\u67E5\u672A\u542F\u7528\u6216\u8DF3\u8FC7_\n")

        metrics = getattr(r, "metrics", {}) or {}
        for k in ACC_KEYS:
            if k in metrics:
                parts.append(_format_metric_line(k, metrics[k]))
                parts.append("\n")

        other_keys = sorted(
            set(metrics.keys()) - set(ACC_KEYS)
            - {"topic_coherence", "length_score", "transition_usage", "descriptive_richness"}
        )
        if other_keys:
            parts.append("\n<details><summary>\u5176\u5B83\u6BB5\u7EA7\u89C4\u5219\u6307\u6807\uff08\u5C55\u5F00\uff09</summary>\n\n")
            for k in other_keys:
                parts.append(_format_metric_line(k, metrics[k]))
                parts.append("\n")
            parts.append("\n</details>\n")

    parts.append("\n---\n## \u6587\u6863\u7EA7\u89C4\u5219\u6307\u6807\n")
    for k, mr in getattr(ev, "document_metrics", {}).items():
        parts.append(_format_metric_line(k, mr))
        parts.append("\n")

    return "".join(parts)


def _format_long_form_novel_content(ev: Any) -> str:
    lines: List[str] = []
    metric_order = ("rag_utilization", "hallucination")

    lines.append("## \u65B0\u5185\u5BB9\u7ED3\u6784\u5316\u6307\u6807\n")
    for r in ev.segments:
        lines.append(f"\n### \u6BB5 {r.segment_index + 1}\n")
        metrics = getattr(r, "metrics", {}) or {}
        for k in metric_order:
            if k in metrics:
                lines.append(_format_metric_line(k, metrics[k]))
                lines.append("\n")

        nci = getattr(r, "novel_content_info", None)
        if not nci:
            lines.append("_\u672C\u6BB5\u65E0\u53EF\u7528\u7684 novel_content_\n")
            continue

        lines.append("\n#### \u65B0\u5B9E\u4F53\u4F7F\u7528\u60C5\u51B5\n")
        avail = nci.get("novel_entities_available") or []
        used = nci.get("novel_entities_used") or []
        lines.append(
            f"- \u53EF\u7528\u65B0\u5B9E\u4F53 ({len(avail)}): `{', '.join(avail[:15])}`" + ("\u2026" if len(avail) > 15 else "")
        )
        lines.append(f"\n- **\u5DF2\u5199\u5165**: `{', '.join(used)}`\n")

        nf = nci.get("new_facts_in_output") or []
        gf = nci.get("grounded_facts") or []
        ug = nci.get("ungrounded_facts") or []

        lines.append("\n\n#### \u65B0\u4E8B\u5B9E / \u6EAF\u6EAF\n")
        lines.append(f"- \u751F\u6210\u4E2D\u51FA\u73B0\u7684\u4E13\u6709\u4FE1\u606F\u6761\u76EE: {len(nf)}\n")
        if nf:
            for i, txt in enumerate(nf[:12], 1):
                lines.append(f"  {i}. {txt}")
            if len(nf) > 12:
                lines.append(f"\n  \u2026 \u5176\u4ED6 {len(nf) - 12} \u6761")
        lines.append(f"\n- \u2713 **\u6709 RAG \u652F\u6491**: {len(gf)}")
        if gf:
            for i, txt in enumerate(gf[:10], 1):
                lines.append(f"\n  {i}. {txt}")
        lines.append(f"\n- \u26a0 **\u7F3A\u4E4F RAG \u652F\u6491 (\u7591\u4F3C\u5E7D\u7075)** : {len(ug)}")
        if ug:
            for i, txt in enumerate(ug[:10], 1):
                lines.append(f"\n  {i}. {txt}")

        summary = nci.get("summary")
        if summary:
            lines.append(f"\n\n**\u7B80\u62A5**: {summary}\n")

    return "".join(lines)


def _format_long_form_relevance_literary(ev: Any, enable_llm_judge_dim: bool) -> Tuple[str, str]:
    rel_lines: List[str] = []
    lit_lines: List[str] = []

    cc = getattr(ev, "cross_chapter_metrics", {}) or {}
    if cc:
        rel_lines.append("## \u8DE8\u7BC7\u7AE0\u89C4\u5219\u6307\u6807\n")
        for k, mr in cc.items():
            rel_lines.append(_format_metric_line(k, mr))
            rel_lines.append("\n")

    REL_KEYS = ("topic_coherence",)
    LIT_KEYS = (
        "length_score",
        "transition_usage",
        "descriptive_richness",
    )

    rel_lines.append("\n## \u5404\u6BB5\u76F8\u5173\u6027 (\u89C4\u5219)\n")
    lit_lines.append("## \u5404\u6BB5\u6587\u5B66\u6027 (\u89C4\u5219)\n")

    for r in ev.segments:
        metrics = getattr(r, "metrics", {}) or {}
        agg = aggregate_scores(metrics)
        head = f"\n### \u6BB5 {r.segment_index + 1} \uff08\u6BB5\u7EA7\u89C4\u5219\u805A\u5408 `\u2248 {agg:.2f}`\uff09\n"
        rel_lines.append(head)
        lit_lines.append(head)
        for k in REL_KEYS:
            if k in metrics:
                rel_lines.append(_format_metric_line(k, metrics[k]))
                rel_lines.append("\n")
        for k in LIT_KEYS:
            if k in metrics:
                lit_lines.append(_format_metric_line(k, metrics[k]))
                lit_lines.append("\n")

    if enable_llm_judge_dim:
        rel_lines.append(
            "\n> _\u5206\u7AE0\u6A21\u5F0F\u4E0E\u5355\u6BB5 LLM Judge \u7684「\u76F8\u5173\u6027」\u5E73\u884C\u5B58\u5728_\n"
        )
        lit_lines.append(
            "\n> _\u5206\u7AE0\u6A21\u5F0F\u4E3A\u89C4\u5219\u6307\u6807+\u957F\u6587评估\uff1B\u5355\u6BB5 LLM 「\u6587\u5B66\u6027」\u4E92\u8865_\n"
        )

    return "".join(rel_lines), "".join(lit_lines)


def _format_long_form_compliance_note() -> str:
    return (
        "### \u5408\u89C4\u6027\u8BF4\u660E\n\n"
        "\u5206\u7AE0\u6A21\u5F0F\u4E0B\u672A\u5BF9\u6BCF\u6BB5\u5355\u72EC\u8C03\u7528 LLM \u5408\u89C4 Judge\u3002\n\n"
        "LLM Judge \u4EC5\u5728**\u975E**\u5206\u7AE0\u5355\u6BB5\u6D41\u6C34\u7EBF\u4E2D\u751F\u6548\uFF1B\u4E0E\u6B64\u533A\u6570\u636E**\u517C\u5BB9\u5E76\u5B58**.\n"
    )


def _format_quality_gate_markdown(ev: Any) -> str:
    gate = getattr(ev, "quality_gate", None)
    if gate is None:
        return "_\u672A\u542F\u7528\u6216\u65E0\u8D28\u91CF\u95E8\u63A7\u7ED3\u679C_\n"
    tip = html.escape(
        "\u5982\u6BB5\u5206\u6570/\u957F\u6BD4/\u5B9E\u4F53\u8986\u76D6/\u6EAF\u6EAF/\u8DE8\u7AE0\u91CD\u590D\u7B49\u9608\u503C\u68C0\u67E5\u3002", quote=True
    )
    gv = "\u2713 \u901A\u8FC7" if gate.passed else "\u2717 \u672A\u901A\u8FC7"
    lines: List[str] = [
        f"## <abbr title=\"{tip}\">\u2139\uFE0F</abbr> \u603B\u89C8\n\n",
        f"- **\u5224\u5B9A**: {gv}\n",
        f"- **\u7EFC\u5408\u5206**: `{gate.overall_score:.2f}`\n",
    ]
    lines.append("\n### \u7AE0\u8282\u68C0\u67E5\n")
    for cr in gate.chapter_results or []:
        st = "\u2713" if cr.passed else "\u2717"
        lines.append(f"\n#### \u7B2C {cr.chapter_index + 1} \u7AE0 {st}\n")
        for issue in getattr(cr, "issues", []) or []:
            lines.append(
                f"- **[{issue.severity}] `{issue.dimension}`**: {issue.message}\n"
                f"  - *\u5EFA\u8BAE*: {issue.suggestion}\n"
            )
    if getattr(gate, "cross_chapter_issues", None):
        lines.append("\n### \u8DE8\u7AE0\u95EE\u9898\n")
        for ci in gate.cross_chapter_issues:
            chs = ", ".join(str(x + 1) for x in (ci.chapters_involved or []))
            lines.append(
                f"- **[{ci.severity}] `{ci.dimension}`**\uff08\u6D89\u53CA\u7AE0\u8282: {chs}\uff09\n"
                f"  - {ci.message}\n"
                f"  - *\u5EFA\u8BAE*: {ci.suggestion}\n"
            )
    rem = getattr(gate, "remediation", None)
    if rem and getattr(rem, "chapters_to_regenerate", None):
        regen = ", ".join(str(c + 1) for c in rem.chapters_to_regenerate)
        lines.append(f"\n### \u4FEE\u590D\u5EFA\u8BAE \u00B7 \u4F18\u5148\u91CD\u751F\u6210\n**\u7B2C {regen} \u7AE0**\n")
        for ch_idx, reasons in (rem.reasons or {}).items():
            lines.append(f"\n- \u7B2C **{int(ch_idx) + 1}** \u7AE0\u539F\u56E0:\n")
            for rs in reasons or []:
                lines.append(f"  - {rs}\n")

    return "".join(lines)


def _log_long_form_chapter_generation(lf: Any) -> None:
    """记录分章初次生成后各章字数与重复警告（便于分析门控失败原因）。"""
    for i, ch in enumerate(getattr(lf, "chapters", []) or []):
        gen = getattr(ch, "generation", None)
        content = getattr(gen, "content", "") if gen else ""
        hint = getattr(ch, "length_hint", "") or ""
        rep = getattr(ch, "repetition_warning", "") or "无"
        logger.info(
            "[LongForm] 第%d章生成完毕: %d字, hint=%s, rep_warn=%s",
            i + 1,
            len(content),
            hint,
            rep,
        )


def _log_long_form_quality_gate(evaluated: Any, round_label: str = "初始") -> None:
    """结构化记录质量门控与各章 issues。"""
    gate = getattr(evaluated, "quality_gate", None)
    if gate is None:
        logger.info("[LongForm] 质量门控(%s): 无结果", round_label)
        return
    logger.info(
        "[LongForm] 质量门控(%s): passed=%s overall=%.2f",
        round_label,
        gate.passed,
        getattr(gate, "overall_score", 0.0) or 0.0,
    )
    for cr in getattr(gate, "chapter_results", []) or []:
        issues_repr = [
            (iss.dimension, iss.severity, iss.message)
            for iss in (getattr(cr, "issues", None) or [])
        ]
        logger.info(
            "  第%d章: passed=%s issues=%s",
            getattr(cr, "chapter_index", -1) + 1,
            cr.passed,
            issues_repr,
        )
    for ci in getattr(gate, "cross_chapter_issues", []) or []:
        logger.info(
            "  跨章: %s severity=%s chapters=%s",
            ci.message,
            ci.severity,
            getattr(ci, "chapters_involved", None),
        )


def _segment_by_chapter_index(evaluated: Any, ch_idx: int) -> Any:
    """按 segment_index（0-based 与章节索引一致）查找段评估记录。"""
    for seg in getattr(evaluated, "segments", []) or []:
        if int(getattr(seg, "segment_index", -1)) == int(ch_idx):
            return seg
    return None


def _entity_avail_used_for_chapter(lf: Any, evaluated: Any, ch_idx: int) -> Tuple[List[str], List[str]]:
    """
    返回 (novel_entities_available, novel_entities_used)。
    优先用评估里的 novel_content_info；若无则从当前章 retrieval 的 brief 回填 available。
    """
    seg = _segment_by_chapter_index(evaluated, ch_idx) if evaluated is not None else None
    avail: List[str] = []
    used: List[str] = []
    if seg is not None:
        nci = getattr(seg, "novel_content_info", None) or {}
        avail = list(nci.get("novel_entities_available") or [])
        used = list(nci.get("novel_entities_used") or [])
    if not avail and lf is not None and 0 <= ch_idx < len(getattr(lf, "chapters", []) or []):
        rr = lf.chapters[ch_idx].retrieval_result
        brief = getattr(rr, "_novel_content_brief", None)
        if brief is not None:
            names = getattr(brief, "novel_entity_names", None) or []
            avail = [n for n in names if n]
    return avail, used


def _log_long_form_segment_rag_entities(evaluated: Any, round_label: str = "") -> None:
    """每章记录 RAG 利用率与 novel 实体可用/已用/未用，便于对照门控。"""
    if not evaluated:
        return
    suffix = f" ({round_label})" if round_label else ""
    for seg in getattr(evaluated, "segments", []) or []:
        idx = int(getattr(seg, "segment_index", 0))
        metrics = getattr(seg, "metrics", {}) or {}
        rag_m = metrics.get("rag_utilization")
        rag_v: Optional[float] = None
        if rag_m is not None and hasattr(rag_m, "value"):
            rag_v = float(rag_m.value)
        nci = getattr(seg, "novel_content_info", None) or {}
        avail = list(nci.get("novel_entities_available") or [])
        used = list(nci.get("novel_entities_used") or [])
        used_set = set(used)
        unused = [e for e in avail if e not in used_set]
        logger.info(
            "[LongForm][RAG实体]%s 第%d章: rag_util=%s | used_n=%d avail_n=%d | used=%s | unused=%s | avail(前15)=%s",
            suffix,
            idx + 1,
            f"{rag_v:.3f}" if rag_v is not None else "n/a",
            len(used),
            len(avail),
            used,
            unused[:25],
            avail[:15],
        )


def _rag_entity_retry_instruction(avail: List[str], used: List[str]) -> str:
    """门控重试时追加：点名未使用的新实体，抬高写入概率。"""
    if not avail:
        return ""
    used_set = set(used)
    unused = [e for e in avail if e not in used_set]
    picks = unused[:8] if unused else avail[:8]
    if not picks:
        return ""
    joined = "、".join(picks)
    return (
        "【RAG新实体】正文中须自然融入至少1个下列名称（须在叙事场景/对话/叙述中出现，禁止单独开列清单）："
        + joined
    )


# 分章模式下回忆录切段偏好（不再占用主界面滑块；与用户「每章生成字数」解耦）
_MEMOIR_SEG_TARGET_MIN = 300
_MEMOIR_SEG_TARGET_MAX = 800


def _clamp_generation_char_bounds(a: int, b: int) -> Tuple[int, int]:
    """用户设定的「生成成文」字数区间上下限（用于 length_hint）。"""
    a, b = int(a), int(b)
    lo, hi = (a, b) if a <= b else (b, a)
    lo = max(80, lo)
    hi = max(lo + 50, hi)
    hi = min(hi, 50_000)
    return lo, hi


def _build_long_form_markdown_output(lf: Any, evaluated: Any) -> str:
    blocks: List[str] = []
    novel_by_seg: Dict[int, Set[str]] = {}
    if evaluated is not None:
        for r in getattr(evaluated, "segments", []) or []:
            nci = getattr(r, "novel_content_info", None) or {}
            used = set(nci.get("novel_entities_used") or [])
            novel_by_seg[int(r.segment_index)] = used

    for i, ch in enumerate(lf.chapters):
        names = _entity_names_sorted_by_length(ch.retrieval_result)
        body = getattr(ch.generation, "content", "") or ""
        body = _bold_entities_in_text(body, names)
        snippet = ch.segment_text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "\u2026"
        nu = novel_by_seg.get(int(ch.segment_index), set())
        used_hint = ""
        if nu:
            used_hint = f"\n\n_\u672C\u6BB5\u5199\u5165\u7684\u65B0 RAG \u5B9E\u4F53_: **{', '.join(sorted(nu))}**"

        src_n = len((ch.segment_text or "").strip())
        gen_n = len(getattr(ch.generation, "content", "") or "")
        blocks.append(
            f"## \u7B2C {i + 1} \u7AE0\n\n"
            f"> **\u539F\u6587\u8282\u9009**: {snippet}\n\n"
            f"- **\u672C\u6BB5\u539F\u6587\u5B57\u6570**: {src_n}\n"
            f"- **\u5B9E\u9645\u751F\u6210\u5B57\u6570**: **{gen_n}**{used_hint}\n\n"
            f"{body.strip()}\n\n---\n"
        )
    return "\n".join(blocks)


async def process_memoir_async(
    memoir_text: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "vector",
    use_rule_decompose: bool = False,
    chapter_mode: bool = False,
    chapter_gen_min_chars: int = 400,
    chapter_gen_max_chars: int = 800,
) -> Tuple[str, str, str, str]:
    """
    异步处理回忆录
    
    Returns:
        (生成的历史背景, 提取的信息, 检索结果, 事实性检查结果)
    """
    global retriever, generator, current_provider, current_model
    
    if not memoir_text.strip():
        return "请输入回忆录文本", "", "", ""

    if not provider:
        return "请先选择 LLM 模型（供应商）", "", "", ""
    
    if (
        retriever is None
        or generator is None
        or current_provider != provider
        or current_model != model
    ):
        init_result = init_components(provider, model=model)
        if "失败" in init_result:
            return init_result, "", "", ""
    active_retriever = _get_retriever_for_mode(retrieval_mode)
    
    try:
        start_time = time.time()
        
        print(f"[DEBUG] 开始处理回忆录，长度: {len(memoir_text)} 字符")

        if chapter_mode:
            seg_lo, seg_hi = _MEMOIR_SEG_TARGET_MIN, _MEMOIR_SEG_TARGET_MAX
            cg_lo, cg_hi = _clamp_generation_char_bounds(chapter_gen_min_chars, chapter_gen_max_chars)
            segs = segment_memoir(
                memoir_text,
                target_min_chars=seg_lo,
                target_max_chars=seg_hi,
            )
            n_ch = max(1, len(segs))
            budget_timeout = estimate_long_form_generation_timeout(n_ch)
            lf = await asyncio.wait_for(
                run_long_form_generation(
                    memoir_text,
                    active_retriever,
                    generator,
                    style=style,
                    temperature=temperature,
                    retrieval_mode=retrieval_mode,
                    use_llm_parsing=False,
                    target_min_chars=seg_lo,
                    target_max_chars=seg_hi,
                    chapter_gen_min_chars=cg_lo,
                    chapter_gen_max_chars=cg_hi,
                ),
                timeout=budget_timeout,
            )
            gen_time = time.time() - start_time
            lines = [f"**分章模式** 共 {len(lf.chapters)} 章（输入约 {len(memoir_text)} 字）"]
            for i, ch in enumerate(lf.chapters):
                ctx = ch.retrieval_result.context
                lines.append(
                    f"- 第{i + 1}章: 年 {ctx.year or '—'} 地 {ctx.location or '—'}"
                )
            extracted_info = "\n".join(lines)
            if _is_plain_vector_rag_mode(retrieval_mode):
                chunk_counts = [len(ch.retrieval_result.text_units) for ch in lf.chapters]
                retrieval_info = (
                    "**分章检索后端**: 普通 RAG（向量 baseline）；"
                    f"各章 chunks 数 {chunk_counts}"
                )
            else:
                rent = [len(ch.retrieval_result.entities) for ch in lf.chapters]
                retrieval_info = (
                    f"**分章检索**: 各章实体数 {rent}；"
                    f"关系总计 {sum(len(ch.retrieval_result.relationships) for ch in lf.chapters)}"
                )
            fact_check_info = f"**生成总耗时**: {gen_time:.2f} 秒\n"
            if enable_fact_check:
                try:
                    llm_adapter = create_llm_adapter(
                        provider=provider, model=model
                    )
                    eval_kwargs = build_long_form_eval_options(
                        llm_adapter=llm_adapter,
                        use_llm_eval=False,
                        enable_fact_check=True,
                        max_atomic_facts_per_segment=12,
                        fact_check_timeout_per_segment=45.0,
                        use_rule_decompose=use_rule_decompose,
                    )
                    ev = await asyncio.wait_for(
                        evaluate_long_form(lf, **eval_kwargs),
                        timeout=estimate_long_form_evaluation_timeout(len(lf.chapters)),
                    )
                    fact_check_info += "### 长文评估汇总\n\n" + ev.summary_text
                    fact_check_info += "\n\n<details><summary>JSON 摘要</summary>\n\n```json\n"
                    fact_check_info += long_form_eval_to_json(ev)[:8000]
                    fact_check_info += "\n```\n</details>\n"
                except Exception as e:
                    fact_check_info += f"\n长文评估未完成: {e}"
            total_time = time.time() - start_time
            fact_check_info += f"\n**总耗时**: {total_time:.2f} 秒"
            return lf.merged_content, extracted_info, retrieval_info, fact_check_info
        
        # 检索相关内容（包含实体提取）
        retrieve_start = time.time()
        print("[DEBUG] 开始检索相关内容...")
        retrieval_result = await asyncio.wait_for(
            active_retriever.retrieve(
                memoir_text,
                top_k=10,
                use_llm_parsing=False,
                mode=retrieval_mode,
            ),
            timeout=45.0,
        )
        retrieve_time = time.time() - retrieve_start
        print(f"[DEBUG] 检索完成，耗时: {retrieve_time:.2f} 秒")
        
        context = retrieval_result.context
        extracted_info = f"""**提取的时间**: {context.year or '未识别'}
**提取的地点**: {context.location or '未识别'}
**关键词**: {', '.join(context.keywords) if context.keywords else '无'}
**生成的查询**: {retrieval_result.query}
**提取+检索耗时**: {retrieve_time:.2f} 秒"""
        
        retrieval_info = _format_retrieval_details(retrieval_result, retrieval_mode)
        
        # 生成文本
        gen_start = time.time()
        print("[DEBUG] 开始生成文本...")
        t_lo, t_hi = _clamp_generation_char_bounds(total_gen_min_chars, total_gen_max_chars)
        single_cfg = single_segment_generation_config_from_range(t_lo, t_hi)

        gen_result = await asyncio.wait_for(
            generator.generate(
                memoir_text=memoir_text,
                retrieval_result=retrieval_result,
                temperature=temperature,
                max_tokens=single_cfg["max_tokens"],
                style=style,
                length_hint=single_cfg["length_hint"],
            ),
            timeout=90.0,
        )
        gen_time = time.time() - gen_start
        print(f"[DEBUG] 生成完成，耗时: {gen_time:.2f} 秒")
        
        # 先生成基本的事实性检查信息
        fact_check_info = f"**生成耗时**: {gen_time:.2f} 秒\n"
        
        # 如果不需要事实性检查，直接返回
        if not enable_fact_check:
            total_time = time.time() - start_time
            fact_check_info += f"\n**总耗时**: {total_time:.2f} 秒"
            print(f"[DEBUG] 处理完成，总耗时: {total_time:.2f} 秒")
            return gen_result.content, extracted_info, retrieval_info, fact_check_info
        
        # 事实性检查（带超时处理）
        print("[DEBUG] 开始事实性检查...")
        
        async def run_fact_check():
            """运行事实性检查"""
            check_start = time.time()
            try:
                llm_adapter = create_llm_adapter(provider=provider, model=model)
                # 使用FActScoreChecker替代原来的FactChecker
                fact_checker = FActScoreChecker(llm_adapter=llm_adapter)
                
                # 带超时的事实性检查
                async def check_with_timeout():
                    return await fact_checker.check(
                        memoir_text=memoir_text,
                        generated_text=gen_result.content,
                        retrieval_result=retrieval_result,
                        use_llm=True,
                        use_rule_decompose=use_rule_decompose,
                    )
                
                # 设置90秒超时（宽限超时时间）
                fact_result = await asyncio.wait_for(check_with_timeout(), timeout=90.0)
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查完成，耗时: {check_time:.2f} 秒")
                
                status_icon = "✅" if fact_result.is_factual else "⚠️"
                return f"""**生成耗时**: {gen_time:.2f} 秒

### {status_icon} 事实性检查结果

**一致性判定**: {'事实一致' if fact_result.is_factual else '存在潜在问题'}
**置信度**: {fact_result.confidence:.2%}
**实体覆盖率**: {fact_result.entity_coverage:.2%}
**证据支持度**: {fact_result.evidence_support:.2%}
**检查耗时**: {check_time:.2f} 秒

**总结**: {fact_result.summary}

"""
                
            except asyncio.TimeoutError:
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查超时，耗时: {check_time:.2f} 秒")
                return f"**生成耗时**: {gen_time:.2f} 秒\n**检查耗时**: {check_time:.2f} 秒\n事实性检查超时，请稍后重试"
            except Exception as e:
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查失败: {str(e)}")
                return f"**生成耗时**: {gen_time:.2f} 秒\n**检查耗时**: {check_time:.2f} 秒\n事实性检查失败: {str(e)}"
        
        # 优化：不再使用后台任务，而是直接同步执行事实性检查
        # 这样可以确保结果能够正确返回给Gradio
        print("[DEBUG] 开始事实性检查...")
        
        # 运行事实性检查（带超时）
        try:
            fact_check_info = await run_fact_check()
            print("[DEBUG] 事实性检查完成")
        except Exception as e:
            print(f"[DEBUG] 事实性检查失败: {e}")
            fact_check_info = fact_check_info + f"\n**事实性检查失败**: {str(e)}"
        
        total_time = time.time() - start_time
        fact_check_info += f"\n**总耗时**: {total_time:.2f} 秒"
        print(f"[DEBUG] 处理完成，总耗时: {total_time:.2f} 秒")
        
        return gen_result.content, extracted_info, retrieval_info, fact_check_info
        
    except Exception as e:
        print(f"[DEBUG] 处理失败: {str(e)}")
        return f"处理失败: {str(e)}", "", "", ""


def process_memoir(
    memoir_text: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "vector",
    use_rule_decompose: bool = False,
    chapter_mode: bool = False,
    chapter_gen_min_chars: int = 400,
    chapter_gen_max_chars: int = 800,
) -> Tuple[str, str, str, str]:
    """处理回忆录（同步包装）"""
    return asyncio.run(
        process_memoir_async(
            memoir_text,
            provider,
            model,
            style,
            total_gen_min_chars,
            total_gen_max_chars,
            temperature,
            enable_fact_check,
            retrieval_mode,
            use_rule_decompose,
            chapter_mode,
            chapter_gen_min_chars,
            chapter_gen_max_chars,
        )
    )


def _loading_html(title: str, subtitle: str = "") -> str:
    """生成生成中占位 UI（依赖全局 _CSS 中的 .loading-card 动画）。

    必须使用多行块级 HTML：Gradio 前端用 marked 解析 Markdown，单行紧挨的 <div>
    不满足 GFM 块级 HTML 规则，会导致标签无法按 DOM 插入，动画样式全部失效。
    """
    safe_title = html.escape(title)
    safe_sub = html.escape(subtitle) if subtitle else ""
    sub_block = f'<br />\n<span class="loading-sub">{safe_sub}</span>' if safe_sub else ""
    # 每行独立、开标签后换行，便于 marked 识别为 raw HTML block（勿再拼成单行）
    return (
        '<div class="graphrag-loading loading-card">\n'
        '<div class="loading-dots"><span></span><span></span><span></span></div>\n'
        '<div class="loading-bar"></div>\n'
        f'<div class="loading-text"><strong>{safe_title}</strong>{sub_block}</div>\n'
        "</div>\n"
    )


def process_memoir_stream(
    memoir_text: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "vector",
    use_rule_decompose: bool = False,
    enable_safe_check: bool = False,
    enable_retrieval_quality: bool = True,
    enable_llm_judge: bool = True,
    chapter_mode: bool = False,
    batch_size: int = 5,
    chapter_gen_min_chars: int = 400,
    chapter_gen_max_chars: int = 800,
    chapter_use_llm_parsing: bool = False,
    chapter_max_gate_retries: int = 2,
):
    """
    Gradio 流式生成。
    yields: (
        output_markdown, extracted_info, retrieval_quality, accuracy_kb,
        safe_check, novel_content, relevance, literary, compliance, quality_gate
    )
    """
    global retriever, generator, current_provider, current_model

    if not memoir_text.strip():
        yield ("请输入回忆录文本",) + ("",) * 9
        return
    if not provider:
        yield ("请先选择 LLM 模型（供应商）",) + ("",) * 9
        return

    if (
        retriever is None
        or generator is None
        or current_provider != provider
        or current_model != model
    ):
        init_result = init_components(provider, model=model)
        if "失败" in init_result:
            yield (init_result,) + ("",) * 9
            return
    active_retriever = _get_retriever_for_mode(retrieval_mode)

    t_lo, t_hi = _clamp_generation_char_bounds(total_gen_min_chars, total_gen_max_chars)
    single_cfg = single_segment_generation_config_from_range(t_lo, t_hi)

    # 各栏目状态
    extracted_md = ""
    retrieval_q_md = ""
    accuracy_md = ""
    safe_md = ""
    novel_md = ""
    relevance_md = ""
    literary_md = ""
    compliance_md = ""
    gate_md = ""

    loop = asyncio.new_event_loop()
    try:
        if chapter_mode:
            seg_lo, seg_hi = _MEMOIR_SEG_TARGET_MIN, _MEMOIR_SEG_TARGET_MAX
            cg_lo, cg_hi = _clamp_generation_char_bounds(chapter_gen_min_chars, chapter_gen_max_chars)
            segs_preview = segment_memoir(
                memoir_text.strip(),
                target_min_chars=seg_lo,
                target_max_chars=seg_hi,
            )
            n_ch = max(1, len(segs_preview))

            yield (
                _loading_html(
                    "分章模式 · 正在检索与生成…",
                    f"预计 {len(segs_preview)} 段｜每章目标生成字数 {cg_lo}–{cg_hi}｜"
                    f"回忆录切段内部默认偏好 {seg_lo}–{seg_hi} 字",
                ),
            ) + ("",) * 9

            async def _long_form():
                return await run_long_form_generation(
                    memoir_text,
                    active_retriever,
                    generator,
                    style=style,
                    temperature=temperature,
                    retrieval_mode=retrieval_mode,
                    use_llm_parsing=chapter_use_llm_parsing,
                    target_min_chars=seg_lo,
                    target_max_chars=seg_hi,
                    chapter_gen_min_chars=cg_lo,
                    chapter_gen_max_chars=cg_hi,
                )

            budget_timeout = estimate_long_form_generation_timeout(n_ch)
            lf = loop.run_until_complete(asyncio.wait_for(_long_form(), timeout=budget_timeout))
            _log_long_form_chapter_generation(lf)

            max_gate_retries = int(max(0, min(10, int(chapter_max_gate_retries))))

            retrieval_q_md = ""
            if enable_retrieval_quality and lf.chapters:
                try:
                    rr0 = lf.chapters[0].retrieval_result
                    q0 = lf.chapters[0].segment_text
                    eval_rq = loop.run_until_complete(
                        asyncio.wait_for(
                            evaluate_retrieval_quality(
                                query_text=q0,
                                text_units=rr0.text_units,
                                llm_adapter=_make_llm(provider, model),
                            ),
                            timeout=45.0,
                        )
                    )
                    retrieval_q_md = _format_retrieval_quality(eval_rq)
                    retrieval_q_md = (
                        "_分章模式：以下以第 1 章回忆录片段为 query 的检索质量（其它章见左侧实体与关系）_\n\n"
                        + retrieval_q_md
                    )
                except Exception as e:
                    retrieval_q_md = f"_分章检索质量评估失败_: {e}"
            elif not enable_retrieval_quality:
                retrieval_q_md = "_未启用_"
            else:
                retrieval_q_md = "_无章节可评估_"

            ent_blocks: List[str] = []
            for i, ch in enumerate(lf.chapters):
                ent_blocks.extend(_format_chapter_entities_block(i, ch, None))
                ent_blocks.append("")
            extracted_md = "\n".join(ent_blocks).strip()

            content_md = _build_long_form_markdown_output(lf, None)
            safe_interim = (
                "_分章模式本流程未调用 SAFE（与单段路径并行；需 SAFE 时请用非分章模式）_"
                if enable_safe_check
                else "_未启用_"
            )
            yield (
                content_md,
                extracted_md,
                retrieval_q_md,
                "_正在运行长文评估…_",
                safe_interim,
                "_正在汇总新内容指标…_",
                "_汇总中…_",
                "_汇总中…_",
                "_分章合规说明待完成_",
                "_质量门控计算中…_",
            )

            evaluated = None
            gate_retry_round = 0

            try:
                llm_adapter = create_llm_adapter(
                    provider=provider,
                    model=model,
                )
                eval_kwargs = build_long_form_eval_options(
                    llm_adapter=llm_adapter,
                    use_llm_eval=False,
                    enable_fact_check=enable_fact_check,
                    max_atomic_facts_per_segment=12,
                    fact_check_timeout_per_segment=45.0,
                    use_rule_decompose=use_rule_decompose,
                    batch_size=int(batch_size),
                )
                evaluated = loop.run_until_complete(
                    asyncio.wait_for(
                        evaluate_long_form(
                            lf,
                            **eval_kwargs,
                            quality_thresholds=QualityThresholds.for_expansion_task(),
                            enable_quality_gate=True,
                        ),
                        timeout=estimate_long_form_evaluation_timeout(len(lf.chapters)),
                    )
                )
                _log_long_form_segment_rag_entities(evaluated, "初始评估")
                _log_long_form_quality_gate(evaluated, "初始评估")

                # 仅在质量门控未通过且存在可执行修复计划时定向重生成（不在初版生成阶段因跨章重复等单独重试）
                while (
                    evaluated is not None
                    and evaluated.quality_gate is not None
                    and not evaluated.quality_gate.passed
                    and evaluated.quality_gate.remediation is not None
                    and gate_retry_round < max_gate_retries
                ):
                    gate_retry_round += 1
                    remed = evaluated.quality_gate.remediation
                    to_regen = remed.chapters_to_regenerate
                    regen_str = ", ".join(str(c + 1) for c in to_regen)
                    logger.info(
                        "[LongForm] 门控重试 %d/%d: 重新生成第 %s 章",
                        gate_retry_round,
                        max_gate_retries,
                        regen_str,
                    )
                    yield (
                        _loading_html(
                            "质量门控未通过，正在重新生成章节…",
                            f"第 {regen_str} 章｜重试 {gate_retry_round}/{max_gate_retries}",
                        ),
                    ) + ("",) * 9

                    merged_adj: Dict[int, str] = dict(remed.prompt_adjustments or {})
                    for ch_idx in to_regen:
                        avail_e, used_e = _entity_avail_used_for_chapter(lf, evaluated, ch_idx)
                        hint_txt = _rag_entity_retry_instruction(avail_e, used_e)
                        if hint_txt:
                            prev_adj = merged_adj.get(ch_idx, "")
                            merged_adj[ch_idx] = f"{prev_adj}；{hint_txt}" if prev_adj else hint_txt
                        unused_n = len([e for e in avail_e if e not in set(used_e)])
                        logger.info(
                            "[LongForm][重试指令] 第%d章: avail_n=%d used_n=%d unused_n=%s entity_hint=%s",
                            ch_idx + 1,
                            len(avail_e),
                            len(used_e),
                            unused_n,
                            bool(hint_txt),
                        )

                    gen_to = estimate_long_form_generation_timeout(max(1, len(to_regen)))
                    lf = loop.run_until_complete(
                        asyncio.wait_for(
                            regenerate_chapters(
                                lf,
                                active_retriever,
                                generator,
                                chapters_to_regenerate=to_regen,
                                prompt_adjustments=merged_adj,
                                style=style,
                                temperature=temperature,
                                retrieval_mode=retrieval_mode,
                                use_llm_parsing=chapter_use_llm_parsing,
                            ),
                            timeout=gen_to,
                        )
                    )
                    _log_long_form_chapter_generation(lf)

                    evaluated = loop.run_until_complete(
                        asyncio.wait_for(
                            evaluate_long_form(
                                lf,
                                **eval_kwargs,
                                quality_thresholds=QualityThresholds.for_expansion_task(),
                                enable_quality_gate=True,
                            ),
                            timeout=estimate_long_form_evaluation_timeout(len(lf.chapters)),
                        )
                    )
                    _log_long_form_segment_rag_entities(
                        evaluated, f"定向重试第{gate_retry_round}轮后"
                    )
                    _log_long_form_quality_gate(
                        evaluated, f"定向重试第{gate_retry_round}轮后"
                    )

                ent_blocks = []
                for i, ch in enumerate(lf.chapters):
                    seg_ix = int(ch.segment_index)
                    nci = next(
                        (
                            s.novel_content_info
                            for s in evaluated.segments
                            if int(s.segment_index) == seg_ix
                        ),
                        None,
                    )
                    used_set = set((nci or {}).get("novel_entities_used") or [])
                    ent_blocks.extend(_format_chapter_entities_block(i, ch, used_set))
                    ent_blocks.append("")
                extracted_md = "\n".join(ent_blocks).strip()

                accuracy_md = _format_long_form_facts_and_rules(evaluated)
                novel_md = _format_long_form_novel_content(evaluated)
                relevance_md, literary_md = _format_long_form_relevance_literary(
                    evaluated, enable_llm_judge
                )
                compliance_md = _format_long_form_compliance_note()
                gate_md = _format_quality_gate_markdown(evaluated)
                if gate_retry_round > 0:
                    gate_md += (
                        f"\n\n---\n_经过 **{gate_retry_round}** 轮定向章节重试。_\n"
                    )

            except Exception as e:
                accuracy_md = f"_长文评估失败_: {e}\n"

            safe_md = (
                "_分章模式本流程未调用 SAFE（与单段路径并行）_"
                if enable_safe_check
                else "_未启用_"
            )

            if evaluated is not None:
                content_md = _build_long_form_markdown_output(lf, evaluated)

            yield (
                content_md,
                extracted_md,
                retrieval_q_md,
                accuracy_md,
                safe_md,
                novel_md if evaluated else "_评估未完成，无法展示_\n",
                relevance_md if evaluated else "",
                literary_md if evaluated else "",
                compliance_md,
                gate_md if evaluated else "",
            )
            return

        yield (
            _loading_html(
                "正在检索相关历史背景…",
                "请稍候，检索完成后将开始生成。",
            ),
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        )

        # ── 1. 检索 ──
        retrieval_result = loop.run_until_complete(asyncio.wait_for(
            active_retriever.retrieve(
                memoir_text, top_k=10, use_llm_parsing=False, mode=retrieval_mode,
            ),
            timeout=45.0,
        ))

        context = retrieval_result.context
        extracted_md = (
            f"**提取的时间**: {context.year or '未识别'}\n"
            f"**提取的地点**: {context.location or '未识别'}\n"
            f"**关键词**: {', '.join(context.keywords) if context.keywords else '无'}\n"
            f"**生成的查询**: {retrieval_result.query}\n\n"
            + _format_retrieval_details(retrieval_result, retrieval_mode)
        )
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, "", relevance_md, literary_md, compliance_md, "")

        # ── 2. 检索质量评估 (LLM-as-a-Judge) ──
        if enable_retrieval_quality:
            try:
                eval_result = loop.run_until_complete(asyncio.wait_for(
                    evaluate_retrieval_quality(
                        query_text=memoir_text,
                        text_units=retrieval_result.text_units,
                        llm_adapter=_make_llm(provider, model),
                    ),
                    timeout=30.0,
                ))
                retrieval_q_md = _format_retrieval_quality(eval_result)
            except Exception as e:
                retrieval_q_md = f"检索质量评估失败: {e}"
        else:
            retrieval_q_md = "_未启用_"
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, "", relevance_md, literary_md, compliance_md, "")

        # ── 3. 流式生成 ──
        async def _stream():
            async for delta in generator.generate_stream(
                memoir_text=memoir_text,
                retrieval_result=retrieval_result,
                style=style,
                length_hint=single_cfg["length_hint"],
                temperature=temperature,
                max_tokens=single_cfg["max_tokens"],
            ):
                yield delta

        content = ""
        agen = _stream()
        gen_start = time.monotonic()
        while True:
            try:
                remaining = 90.0 - (time.monotonic() - gen_start)
                if remaining <= 0:
                    raise asyncio.TimeoutError()
                delta = loop.run_until_complete(asyncio.wait_for(agen.__anext__(), timeout=remaining))
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                content += "\n\n（\u26a0\ufe0f 生成超时：已达到 90s 上限，返回已生成的部分内容）"
                yield (
                    content,
                    extracted_md,
                    retrieval_q_md,
                    accuracy_md,
                    safe_md,
                    "",
                    relevance_md,
                    literary_md,
                    compliance_md,
                    "",
                )
                break
            content += delta
            yield (
                content,
                extracted_md,
                retrieval_q_md,
                accuracy_md,
                safe_md,
                "",
                relevance_md,
                literary_md,
                compliance_md,
                "",
            )

        # ── 4. 综合评估（事实准确性 + SAFE + LLM-as-a-Judge） ──
        any_eval = enable_fact_check or enable_safe_check or enable_llm_judge
        if any_eval and content:
            try:
                async def _run_eval():
                    llm = _make_llm(provider, model)
                    evaluator = Evaluator(
                        llm_adapter=llm,
                        google_api_key=settings.google_api_key,
                        google_cse_id=getattr(settings, 'google_cse_id', None),
                    )
                    return await evaluator.evaluate(
                        memoir_text=memoir_text,
                        generated_text=content,
                        retrieval_result=retrieval_result,
                        use_llm=True,
                        enable_fact_check=enable_fact_check,
                        enable_safe_check=enable_safe_check,
                        enable_llm_judge=enable_llm_judge,
                        batch_size=int(batch_size),
                    )

                eval_res = loop.run_until_complete(asyncio.wait_for(_run_eval(), timeout=180.0))

                if eval_res.fact_check:
                    accuracy_md = _format_accuracy(eval_res.fact_check, 0)
                if eval_res.safe_check:
                    safe_md = _format_safe_check(eval_res.safe_check)
                if "relevance" in eval_res.scores:
                    relevance_md = _format_dimension(eval_res.scores["relevance"])
                if "literary" in eval_res.scores:
                    literary_md = _format_dimension(eval_res.scores["literary"])
                if "compliance" in eval_res.scores:
                    compliance_md = _format_compliance(eval_res.scores["compliance"])

            except Exception as e:
                accuracy_md = f"评估失败: {e}"

        if not enable_fact_check and not accuracy_md:
            accuracy_md = "_未启用_"
        if not enable_safe_check and not safe_md:
            safe_md = "_未启用_"
        if not enable_llm_judge:
            if not relevance_md:
                relevance_md = "_未启用_"
            if not literary_md:
                literary_md = "_未启用_"
            if not compliance_md:
                compliance_md = "_未启用_"

        novel_note = (
            "_单段模式的长文「新内容评估」指标见分章流水线；此处留空（与分章路径并存）_"
        )
        gate_note = "_单段模式无跨章质量门控；分章模式单独展示（两套逻辑并存）_"

        yield (
            content,
            extracted_md,
            retrieval_q_md,
            accuracy_md,
            safe_md,
            novel_note,
            relevance_md,
            literary_md,
            compliance_md,
            gate_note,
        )
    finally:
        try:
            loop.close()
        except Exception:
            pass


async def compare_providers_async(
    memoir_text: str,
    selected_providers: List[str],
    temperature: float,
) -> str:
    """
    使用多个LLM对比生成
    """
    if not memoir_text.strip():
        return "请输入回忆录文本"

    if not selected_providers:
        return "请至少选择一个LLM提供商"

    try:
        # 创建多LLM路由器
        router = LLMRouter()

        # 创建生成器
        gen = LiteraryGenerator(llm_router=router)

        # 创建检索器并检索
        ret = MemoirRetriever()
        retrieval_result = ret.retrieve_sync(memoir_text, top_k=10)

        # 并行生成
        multi_result = await gen.generate_parallel(
            memoir_text=memoir_text,
            retrieval_result=retrieval_result,
            providers=selected_providers,
            temperature=temperature,
        )

        # 格式化结果
        output = []
        for provider_name, result in multi_result.results.items():
            output.append(f"## {provider_name.upper()} ({result.model})\n")
            output.append(result.content)
            output.append("\n---\n")

        for provider_name, error in multi_result.errors.items():
            output.append(f"## {provider_name.upper()} - 错误\n")
            output.append(f"\u274c {error}\n---\n")

        return "\n".join(output)

    except Exception as e:
        return f"对比生成失败: {str(e)}"


def compare_providers(
    memoir_text: str,
    selected_providers: List[str],
    temperature: float,
) -> str:
    """对比生成（同步包装）"""
    return asyncio.run(compare_providers_async(memoir_text, selected_providers, temperature))


# ────────────────────────────────────────────────────────────
# 批量测试 (Benchmark) — 在固定测试集上系统性地跑生成 + 评估
# ────────────────────────────────────────────────────────────

BENCHMARK_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "memoirs" / "segments"

BENCHMARK_COLUMNS = [
    "来源文件",
    "ID",
    "章节",
    "历史标签",
    "检索模式",
    "原文",
    "扩写文本",
    "nDCG@5",
    "MRR",
    "Precision",
    "Recall",
    "F1",
    "Hit@3",
    "Hit@5",
    "Hit@10",
    "检索耗时(ms)",
    "FActScore",
    "SAFE",
    "时间一致性",
    "幻觉风险控制",
    "相关性",
    "主题一致性",
    "文学性",
    "长度得分",
    "衔接词使用",
    "描写丰富度",
    "RAG利用率",
    "信息增量",
    "扩展溯源率",
    "合规性",
    "综合分",
    "段级综合分",
    "耗时(s)",
    "状态",
]


def _benchmark_retrieval_specs(
    retrieval_mode: str,
    compare_plain_vector_rag: bool,
) -> List[Tuple[str, str]]:
    """Return retrieval modes to run for benchmark rows."""
    def graph_label(mode: str) -> str:
        if mode == "vector":
            return "GraphRAG Local Search"
        if mode == "hybrid":
            return "GraphRAG Hybrid"
        if mode == "keyword":
            return "GraphRAG 关键词"
        return f"GraphRAG:{mode}"

    if not compare_plain_vector_rag:
        label = (
            "普通向量RAG"
            if _is_plain_vector_rag_mode(retrieval_mode)
            else graph_label(retrieval_mode)
        )
        return [(retrieval_mode, label)]

    graph_mode = retrieval_mode
    if _is_plain_vector_rag_mode(graph_mode):
        graph_mode = "hybrid"

    return [
        (graph_mode, graph_label(graph_mode)),
        (PLAIN_VECTOR_RAG_MODE, "普通向量RAG"),
    ]

# 用于求平均的数值指标键 → 显示名 / 是否百分比
BENCHMARK_METRIC_KEYS = [
    ("ndcg_at_5", "nDCG@5", False),
    ("mrr", "MRR", False),
    ("precision", "Precision", False),
    ("recall", "Recall", False),
    ("f1", "F1", False),
    ("hit_at_3", "Hit@3", False),
    ("hit_at_5", "Hit@5", False),
    ("hit_at_10", "Hit@10", False),
    ("retrieval_latency_ms", "检索耗时(ms)", False),
    ("factscore", "FActScore", True),
    ("safe_score", "SAFE", True),
    ("temporal_coherence", "时间一致性", False),
    ("hallucination", "幻觉风险控制", False),
    ("relevance", "相关性", False),
    ("topic_coherence", "主题一致性", False),
    ("literary", "文学性", False),
    ("length_score", "长度得分", False),
    ("transition_usage", "衔接词使用", False),
    ("descriptive_richness", "描写丰富度", False),
    ("rag_utilization", "RAG利用率", False),
    ("information_gain", "信息增量", False),
    ("expansion_grounding", "扩展溯源率", False),
    ("compliance", "合规性", False),
    ("overall_score", "综合分", False),
    ("aggregated_score", "段级综合分", False),
    ("elapsed_seconds", "平均耗时(s)", False),
]


_benchmark_stop_event = threading.Event()
_benchmark_running_event = threading.Event()
_benchmark_resume_lock = threading.Lock()
_benchmark_resume_state: Optional[Dict[str, Any]] = None


def _clear_benchmark_resume_state() -> None:
    global _benchmark_resume_state
    with _benchmark_resume_lock:
        _benchmark_resume_state = None


def _get_benchmark_resume_state() -> Optional[Dict[str, Any]]:
    with _benchmark_resume_lock:
        return copy.deepcopy(_benchmark_resume_state)


def _save_benchmark_resume_state(state: Dict[str, Any]) -> None:
    global _benchmark_resume_state
    with _benchmark_resume_lock:
        _benchmark_resume_state = copy.deepcopy(state)


def _benchmark_params_snapshot(
    dataset_dir: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool,
    retrieval_mode: str,
    use_rule_decompose: bool,
    enable_safe_check: bool,
    enable_retrieval_quality: bool,
    enable_llm_judge: bool,
    batch_size: int,
    compare_plain_vector_rag: bool,
) -> Dict[str, Any]:
    return {
        "dataset_dir": dataset_dir,
        "provider": provider,
        "model": model,
        "style": style,
        "total_gen_min_chars": int(total_gen_min_chars),
        "total_gen_max_chars": int(total_gen_max_chars),
        "temperature": float(temperature),
        "enable_fact_check": bool(enable_fact_check),
        "retrieval_mode": retrieval_mode,
        "use_rule_decompose": bool(use_rule_decompose),
        "enable_safe_check": bool(enable_safe_check),
        "enable_retrieval_quality": bool(enable_retrieval_quality),
        "enable_llm_judge": bool(enable_llm_judge),
        "batch_size": int(batch_size),
        "compare_plain_vector_rag": bool(compare_plain_vector_rag),
    }


def _restore_benchmark_params(params: Dict[str, Any]) -> Tuple[
    str,
    str,
    Optional[str],
    str,
    int,
    int,
    float,
    bool,
    str,
    bool,
    bool,
    bool,
    bool,
    int,
    bool,
]:
    return (
        params["dataset_dir"],
        params["provider"],
        params.get("model"),
        params["style"],
        int(params["total_gen_min_chars"]),
        int(params["total_gen_max_chars"]),
        float(params["temperature"]),
        bool(params["enable_fact_check"]),
        params["retrieval_mode"],
        bool(params["use_rule_decompose"]),
        bool(params["enable_safe_check"]),
        bool(params["enable_retrieval_quality"]),
        bool(params["enable_llm_judge"]),
        int(params["batch_size"]),
        bool(params["compare_plain_vector_rag"]),
    )


def request_benchmark_stop() -> str:
    """由「停止」按钮调用：设置标志位，请求批量任务在当前段落结束后中止。"""
    _benchmark_stop_event.set()
    return "🛑 已请求停止：将在当前段落处理完成后中止，并导出已完成部分的结果。"


def _benchmark_control_button_update():
    """Return the UI state for the combined stop/resume benchmark button."""
    if _benchmark_running_event.is_set():
        if _benchmark_stop_event.is_set():
            return gr.update(value="⏳ 停止中…", variant="secondary", interactive=False)
        return gr.update(value="⏹️ 停止", variant="stop", interactive=True)
    if _get_benchmark_resume_state():
        return gr.update(value="▶️ 继续批量测试", variant="secondary", interactive=True)
    return gr.update(value="⏹️ 停止 / ▶️ 继续", variant="secondary", interactive=True)


def _with_benchmark_control_button(stream):
    """Append the combined stop/resume button update to benchmark UI stream outputs."""
    for output in stream:
        yield (*output, _benchmark_control_button_update())


def _load_benchmark_dataset(dataset_dir: str) -> Tuple[List[dict], List[str], List[str]]:
    """
    加载文件夹下的所有 *.json 测试集。每条片段附加 `source_file` 字段。

    Returns: (segments, loaded_files, skipped_files)
    """
    p = Path(dataset_dir).expanduser()
    if not p.is_absolute():
        p = (Path(__file__).resolve().parent.parent / p).resolve()
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"目录不存在或不是文件夹: {p}")

    segments: List[dict] = []
    loaded: List[str] = []
    skipped: List[str] = []
    for fp in sorted(p.glob("*.json")):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list) or not all(
                isinstance(x, dict) and "original_text" in x for x in data
            ):
                skipped.append(f"{fp.name} (schema 不匹配)")
                continue
            for item in data:
                item = dict(item)
                item["source_file"] = fp.name
                segments.append(item)
            loaded.append(fp.name)
        except Exception as e:
            skipped.append(f"{fp.name} ({type(e).__name__}: {e})")
    return segments, loaded, skipped


def _compute_averages(detail_records: List[dict]) -> Dict[str, Optional[float]]:
    """对所有数值评估指标 + elapsed_seconds 求平均（忽略 None）。"""
    out: Dict[str, Optional[float]] = {}
    for key, _, _ in BENCHMARK_METRIC_KEYS:
        if key == "elapsed_seconds":
            vals = [r.get("elapsed_seconds") for r in detail_records if r.get("elapsed_seconds") is not None]
        else:
            vals = [r["scores"].get(key) for r in detail_records if r["scores"].get(key) is not None]
        out[key] = (sum(vals) / len(vals)) if vals else None
    return out


def _format_averages_md(averages: Dict[str, Optional[float]], n: int) -> str:
    """渲染平均值为 Markdown 表（用于 UI 与导出）。"""
    if n == 0:
        return "_暂无数据_"
    lines = ["| 指标 | 平均值 | 有效样本 |", "| --- | --- | --- |"]
    for key, label, pct in BENCHMARK_METRIC_KEYS:
        v = averages.get(key)
        if v is None:
            cell = "—"
        elif pct:
            cell = f"{v:.1%}"
        elif key == "elapsed_seconds":
            cell = f"{v:.2f}s"
        else:
            cell = f"{v:.3f}"
        lines.append(f"| {label} | {cell} | {n} |")
    return "\n".join(lines)


def _format_average_value(
    averages: Dict[str, Optional[float]],
    key: str,
    pct: bool,
) -> str:
    v = averages.get(key)
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    if key == "elapsed_seconds":
        return f"{v:.2f}s"
    if key == "retrieval_latency_ms":
        return f"{v:.1f}"
    return f"{v:.3f}"


def _format_grouped_averages_md(detail_records: List[dict]) -> str:
    """Render per-retrieval-mode benchmark averages as one side-by-side table."""
    if not detail_records:
        return "_暂无数据_"

    modes = []
    seen = set()
    for record in detail_records:
        mode_label = record.get("retrieval_label") or record.get("retrieval_mode") or "unknown"
        if mode_label not in seen:
            seen.add(mode_label)
            modes.append(mode_label)

    grouped = []
    for mode_label in modes:
        records = [
            r for r in detail_records
            if (r.get("retrieval_label") or r.get("retrieval_mode") or "unknown") == mode_label
        ]
        grouped.append((mode_label, _compute_averages(records), len(records)))

    header = ["指标"]
    for mode_label, _, n in grouped:
        header.append(f"{mode_label} (n={n})")
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for key, label, pct in BENCHMARK_METRIC_KEYS:
        row = [label]
        for _, averages, _ in grouped:
            row.append(_format_average_value(averages, key, pct))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _truncate(text: Optional[str], n: int = 100) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_score(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def _retrieval_binary_metrics(eval_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Derive binary retrieval metrics from LLM relevance scores.

    The batch memoir datasets do not provide manual qrels. We therefore use the
    same LLM relevance vector as nDCG/MRR and treat score >= 2 as relevant.
    Recall is measured over relevant documents within the returned top-10 list.
    """
    rel_vector = list(eval_result.get("relevance_vector") or [])
    if not rel_vector:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "hit_at_10": None,
        }

    relevant = [1 if float(v) >= 2.0 else 0 for v in rel_vector]
    returned = len(relevant)
    total_relevant = sum(relevant)
    precision = total_relevant / returned if returned else None
    recall = 1.0 if total_relevant > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else 0.0
    )

    def hit_at(k: int) -> float:
        return 1.0 if any(relevant[:k]) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "hit_at_10": hit_at(10),
    }


async def _calculate_benchmark_rule_metrics(
    original_text: str,
    generated_text: str,
    retrieval_result: Any,
    min_chars: int,
    max_chars: int,
) -> Dict[str, Optional[float]]:
    """Compute the same rule/new-content metrics used by long-form evaluation."""
    novel_brief = getattr(retrieval_result, "_novel_content_brief", None)
    retrieval_entities: List[str] = []
    if novel_brief is not None:
        retrieval_entities = list(getattr(novel_brief, "novel_entity_names", []) or [])[:10]
    elif retrieval_result is not None:
        for e in (getattr(retrieval_result, "entities", None) or [])[:15]:
            name = e.get("name") or e.get("title") or ""
            if name:
                retrieval_entities.append(name)

    ctx = getattr(retrieval_result, "context", None)
    metrics = await calculate_all_metrics(
        original_text,
        generated_text,
        reference_entities=retrieval_entities,
        reference_year=getattr(ctx, "year", None),
        keywords=getattr(ctx, "keywords", None),
        literary_length_min=max(50, min_chars),
        literary_length_max=max(max_chars, min_chars + 1),
        literary_optimal_min=max(50, min_chars),
        literary_optimal_max=max(max_chars, min_chars + 1),
        novel_content_brief=novel_brief,
        task_type="expansion",
        min_required_entities=2,
    )

    scores: Dict[str, Optional[float]] = {
        key: metrics[key].normalized if key in metrics else None
        for key in (
            "temporal_coherence",
            "topic_coherence",
            "length_score",
            "transition_usage",
            "descriptive_richness",
            "rag_utilization",
            "hallucination",
        )
    }
    scores["aggregated_score"] = aggregate_scores(metrics) if metrics else None

    novel_info = getattr(novel_brief, "has_novel_content", False) if novel_brief is not None else False
    if novel_brief is not None and novel_info:
        from src.evaluation import analyze_novel_content

        analysis = await analyze_novel_content(
            original_text,
            generated_text,
            novel_brief,
            None,
        )
        scores["information_gain"] = analysis.information_gain
        scores["expansion_grounding"] = analysis.expansion_grounding
    else:
        scores["information_gain"] = None
        scores["expansion_grounding"] = None

    return scores


def _benchmark_rows_snapshot(rows: List[List[str]]) -> List[List[str]]:
    """Return a detached snapshot for Gradio streaming updates."""
    return [list(row) for row in rows]


def batch_benchmark_stream(
    dataset_dir: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool,
    retrieval_mode: str,
    use_rule_decompose: bool,
    enable_safe_check: bool,
    enable_retrieval_quality: bool,
    enable_llm_judge: bool,
    batch_size: int,
    compare_plain_vector_rag: bool = True,
):
    yield from _with_benchmark_control_button(
        _batch_benchmark_stream(
            dataset_dir=dataset_dir,
            provider=provider,
            model=model,
            style=style,
            total_gen_min_chars=total_gen_min_chars,
            total_gen_max_chars=total_gen_max_chars,
            temperature=temperature,
            enable_fact_check=enable_fact_check,
            retrieval_mode=retrieval_mode,
            use_rule_decompose=use_rule_decompose,
            enable_safe_check=enable_safe_check,
            enable_retrieval_quality=enable_retrieval_quality,
            enable_llm_judge=enable_llm_judge,
            batch_size=batch_size,
            compare_plain_vector_rag=compare_plain_vector_rag,
            resume=False,
        )
    )


def resume_benchmark_stream():
    state = _get_benchmark_resume_state()
    if not state:
        yield (
            "⚠️ 当前没有可继续的批量测试。请先运行批量测试并在中途停止。",
            [],
            "_暂无可继续的平均值_",
            None,
            None,
            None,
        )
        return

    yield from _batch_benchmark_stream(
        *_restore_benchmark_params(state["params"]),
        resume=True,
    )


def benchmark_stop_or_resume_stream():
    if _benchmark_running_event.is_set():
        _benchmark_stop_event.set()
        yield (
            "🛑 已请求停止：将在当前段落处理完成后中止，并导出已完成部分的结果。",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            _benchmark_control_button_update(),
        )
        return

    if not _get_benchmark_resume_state():
        yield (
            "⚠️ 当前没有可继续的批量测试。请先运行批量测试并在中途停止。",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            _benchmark_control_button_update(),
        )
        return

    yield from _with_benchmark_control_button(resume_benchmark_stream())


def _batch_benchmark_stream(
    dataset_dir: str,
    provider: str,
    model: Optional[str],
    style: str,
    total_gen_min_chars: int,
    total_gen_max_chars: int,
    temperature: float,
    enable_fact_check: bool,
    retrieval_mode: str,
    use_rule_decompose: bool,
    enable_safe_check: bool,
    enable_retrieval_quality: bool,
    enable_llm_judge: bool,
    batch_size: int,
    compare_plain_vector_rag: bool = True,
    resume: bool = False,
):
    """
    在指定文件夹下的所有 *.json 测试集上系统性地跑扩写 + 评估。

    - 扩写文本始终生成（无视 UI 评估开关）
    - 评估模块按 UI 当前勾选状态执行
    - 所有参数沿用主标签页 UI 设置
    - 跑完后对所有数值指标求平均，回写到 UI / JSON / Markdown / CSV

    yields: (status_md, dataframe_rows, averages_md, json_file, md_file, csv_file)
    """
    global retriever, generator, current_provider, current_model

    # 重置停止标志，避免上次残留
    _benchmark_stop_event.clear()
    _benchmark_running_event.set()
    empty_avg = "_尚未开始_"
    params = _benchmark_params_snapshot(
        dataset_dir=dataset_dir,
        provider=provider,
        model=model,
        style=style,
        total_gen_min_chars=total_gen_min_chars,
        total_gen_max_chars=total_gen_max_chars,
        temperature=temperature,
        enable_fact_check=enable_fact_check,
        retrieval_mode=retrieval_mode,
        use_rule_decompose=use_rule_decompose,
        enable_safe_check=enable_safe_check,
        enable_retrieval_quality=enable_retrieval_quality,
        enable_llm_judge=enable_llm_judge,
        batch_size=batch_size,
        compare_plain_vector_rag=compare_plain_vector_rag,
    )

    if not provider:
        yield ("⚠️ 请先在「生成历史背景」标签页选择 LLM 模型（供应商）。", [], empty_avg, None, None, None)
        _benchmark_running_event.clear()
        return

    resume_state = _get_benchmark_resume_state() if resume else None
    if resume and not resume_state:
        yield (
            "⚠️ 当前没有可继续的批量测试。请先运行批量测试并在中途停止。",
            [],
            "_暂无可继续的平均值_",
            None,
            None,
            None,
        )
        _benchmark_running_event.clear()
        return
    if not resume:
        _clear_benchmark_resume_state()

    # 复用 / 必要时初始化检索器与生成器
    if (
        retriever is None
        or generator is None
        or current_provider != provider
        or current_model != model
    ):
        init_msg = init_components(provider, model=model)
        if "失败" in init_msg:
            yield (init_msg, [], empty_avg, None, None, None)
            _benchmark_running_event.clear()
            return

    if resume_state:
        dataset = resume_state["dataset"]
        loaded_files = resume_state["loaded_files"]
        skipped_files = resume_state["skipped_files"]
        retrieval_specs = resume_state["retrieval_specs"]
        segment_total = int(resume_state["segment_total"])
        total_jobs = int(resume_state["total_jobs"])
        total = total_jobs
        bg_lo = int(resume_state["bg_lo"])
        bg_hi = int(resume_state["bg_hi"])
        single_cfg = resume_state["single_cfg"]
        rows = resume_state["rows"]
        detail_records = resume_state["detail_records"]
        start_index = int(resume_state.get("next_index", len(detail_records)))
        elapsed_before = float(resume_state.get("elapsed_seconds", 0.0))
    else:
        try:
            dataset, loaded_files, skipped_files = _load_benchmark_dataset(
                dataset_dir or str(BENCHMARK_DEFAULT_DIR)
            )
        except Exception as e:
            yield (f"❌ 读取测试集目录失败: {e}", [], empty_avg, None, None, None)
            _benchmark_running_event.clear()
            return

        total = len(dataset)
        if total == 0:
            skip_note = f"\n跳过的文件: {', '.join(skipped_files)}" if skipped_files else ""
            yield (f"⚠️ 目录下未找到有效测试集 *.json。{skip_note}", [], empty_avg, None, None, None)
            _benchmark_running_event.clear()
            return

        bg_lo, bg_hi = _clamp_generation_char_bounds(total_gen_min_chars, total_gen_max_chars)
        single_cfg = single_segment_generation_config_from_range(bg_lo, bg_hi)
        retrieval_specs = _benchmark_retrieval_specs(
            retrieval_mode,
            bool(compare_plain_vector_rag),
        )
        segment_total = total
        benchmark_jobs: List[dict] = []
        for item in dataset:
            for run_mode, run_label in retrieval_specs:
                job = dict(item)
                job["_benchmark_retrieval_mode"] = run_mode
                job["_benchmark_retrieval_label"] = run_label
                benchmark_jobs.append(job)
        dataset = benchmark_jobs
        total_jobs = len(dataset)
        total = total_jobs
        rows = []
        detail_records = []
        start_index = 0
        elapsed_before = 0.0

    def current_averages_md() -> str:
        return _format_grouped_averages_md(detail_records) if detail_records else empty_avg

    bench_start = time.monotonic() - elapsed_before

    files_note = f" 来源: {len(loaded_files)} 个文件 ({', '.join(loaded_files)})"
    if skipped_files:
        files_note += f" · 跳过: {', '.join(skipped_files)}"
    start_label = "继续" if resume_state else "开始"
    yield (
        f"⏳ {start_label}批量测试：共 {segment_total} 条段落、{len(retrieval_specs)} 个检索模式，合计 {total_jobs} 个任务。"
        f"已完成 {len(detail_records)} 个，将从第 {start_index + 1} 个任务继续。{files_note}\n\n"
        f"当前配置：{provider}{f'/{model}' if model else ''} · "
        f"风格 `{style}` · 整篇目标生成 `{bg_lo}`–`{bg_hi}` 字 · "
        f"温度 `{temperature}` · 检索 `{', '.join(label for _, label in retrieval_specs)}`",
        _benchmark_rows_snapshot(rows),
        current_averages_md(),
        None,
        None,
        None,
    )

    loop = asyncio.new_event_loop()
    stopped_early = False
    try:
        for idx, item in enumerate(dataset[start_index:], start=start_index + 1):
            # 段落级停止检查：只在新段落开始前响应停止请求
            if _benchmark_stop_event.is_set():
                stopped_early = True
                yield (
                    f"🛑 已停止：在第 {idx} / {total} 条段落开始前中止。已完成 {len(rows)} 条，正在导出…",
                    _benchmark_rows_snapshot(rows), current_averages_md(), None, None, None,
                )
                break

            source_file = item.get("source_file", "")
            seg_id = item.get("id", f"#{idx}")
            chapter = item.get("chapter", "")
            tags = item.get("historical_tag", []) or []
            tags_str = "、".join(tags)
            original_text = item.get("original_text", "")
            run_mode = item.get("_benchmark_retrieval_mode", retrieval_mode)
            run_label = item.get("_benchmark_retrieval_label", run_mode)
            active_retriever = _get_retriever_for_mode(run_mode)

            seg_start = time.monotonic()
            base_status = (
                f"### 进度: {idx} / {total}\n\n"
                f"- 来源: `{source_file}`\n"
                f"- 当前: **{seg_id}** — {chapter}\n"
                f"- 检索模式: `{run_label}`\n"
                f"- 历史标签: {tags_str}\n"
                f"- 已耗时: {time.monotonic() - bench_start:.1f}s\n"
            )
            row = [
                source_file, seg_id, chapter, tags_str, run_label,
                _truncate(original_text, 120), "",
                "—", "—", "—", "—", "—", "—", "—", "—", "—",
                "—", "—", "—", "—", "—", "—", "—", "—", "—",
                "—", "—", "—", "—", "—", "—", "—", "—",
                "进行中",
            ]
            rows.append(row)
            yield (
                base_status + "\n_正在检索…_",
                _benchmark_rows_snapshot(rows),
                current_averages_md(),
                None,
                None,
                None,
            )

            generated_text = ""
            ndcg5 = mrr = precision = recall = f1 = hit_at_3 = hit_at_5 = hit_at_10 = None
            retrieval_latency_ms = None
            factscore = safe_score = temporal_coherence = hallucination = None
            relevance = topic_coherence = literary = length_score = None
            transition_usage = descriptive_richness = None
            rag_utilization = information_gain = expansion_grounding = None
            compliance = overall_score = aggregated_score = None
            error_msg = ""
            retrieval_result = None

            try:
                # ── 1. 检索 ──
                retrieval_start = time.monotonic()
                retrieval_result = loop.run_until_complete(asyncio.wait_for(
                    active_retriever.retrieve(
                        original_text, top_k=10, use_llm_parsing=False, mode=run_mode,
                    ),
                    timeout=60.0,
                ))
                retrieval_latency_ms = (time.monotonic() - retrieval_start) * 1000
                row[15] = f"{retrieval_latency_ms:.1f}"

                # ── 2. 检索质量（按需，先于生成，避免生成失败吞掉检索指标） ──
                if enable_retrieval_quality:
                    yield (
                        base_status + "\n_正在评估检索质量…_",
                        _benchmark_rows_snapshot(rows),
                        current_averages_md(),
                        None,
                        None,
                        None,
                    )
                    try:
                        eval_result = loop.run_until_complete(asyncio.wait_for(
                            evaluate_retrieval_quality(
                                query_text=original_text,
                                text_units=retrieval_result.text_units,
                                llm_adapter=_make_llm(provider, model),
                            ),
                            timeout=60.0,
                        ))
                        if eval_result.get("judge_error"):
                            row[7] = "失败"
                            row[8] = "失败"
                            error_msg = (
                                (error_msg + " | " if error_msg else "")
                                + f"检索评估: {eval_result['judge_error']}"
                            )
                        else:
                            ndcg5 = eval_result.get("ndcg_at_5")
                            mrr = eval_result.get("mrr")
                            rbm = _retrieval_binary_metrics(eval_result)
                            precision = rbm.get("precision")
                            recall = rbm.get("recall")
                            f1 = rbm.get("f1")
                            hit_at_3 = rbm.get("hit_at_3")
                            hit_at_5 = rbm.get("hit_at_5")
                            hit_at_10 = rbm.get("hit_at_10")
                            row[7] = _fmt_score(ndcg5)
                            row[8] = _fmt_score(mrr)
                            row[9] = _fmt_score(precision)
                            row[10] = _fmt_score(recall)
                            row[11] = _fmt_score(f1)
                            row[12] = _fmt_score(hit_at_3)
                            row[13] = _fmt_score(hit_at_5)
                            row[14] = _fmt_score(hit_at_10)
                    except Exception as e:
                        row[7] = "失败"
                        row[8] = "失败"
                        error_msg = (error_msg + " | " if error_msg else "") + f"检索评估: {e}"

                # ── 3. 扩写（必跑） ──
                yield (
                    base_status + "\n_正在生成扩写…_",
                    _benchmark_rows_snapshot(rows),
                    current_averages_md(),
                    None,
                    None,
                    None,
                )
                gen_result = loop.run_until_complete(asyncio.wait_for(
                    generator.generate(
                        memoir_text=original_text,
                        retrieval_result=retrieval_result,
                        temperature=temperature,
                        max_tokens=single_cfg["max_tokens"],
                        style=style,
                        length_hint=single_cfg["length_hint"],
                    ),
                    timeout=120.0,
                ))
                generated_text = gen_result.content or ""
                row[6] = _truncate(generated_text, 120)
                yield (
                    base_status + "\n_扩写完成，进入评估…_",
                    _benchmark_rows_snapshot(rows),
                    current_averages_md(),
                    None,
                    None,
                    None,
                )

                # ── 4. 长文同款规则指标 + 新内容指标 ──
                try:
                    rule_metric_scores = loop.run_until_complete(asyncio.wait_for(
                        _calculate_benchmark_rule_metrics(
                            original_text=original_text,
                            generated_text=generated_text,
                            retrieval_result=retrieval_result,
                            min_chars=bg_lo,
                            max_chars=bg_hi,
                        ),
                        timeout=180.0,
                    ))
                    temporal_coherence = rule_metric_scores.get("temporal_coherence")
                    hallucination = rule_metric_scores.get("hallucination")
                    topic_coherence = rule_metric_scores.get("topic_coherence")
                    length_score = rule_metric_scores.get("length_score")
                    transition_usage = rule_metric_scores.get("transition_usage")
                    descriptive_richness = rule_metric_scores.get("descriptive_richness")
                    rag_utilization = rule_metric_scores.get("rag_utilization")
                    information_gain = rule_metric_scores.get("information_gain")
                    expansion_grounding = rule_metric_scores.get("expansion_grounding")
                    aggregated_score = rule_metric_scores.get("aggregated_score")
                    row[18] = _fmt_score(temporal_coherence)
                    row[19] = _fmt_score(hallucination)
                    row[21] = _fmt_score(topic_coherence)
                    row[23] = _fmt_score(length_score)
                    row[24] = _fmt_score(transition_usage)
                    row[25] = _fmt_score(descriptive_richness)
                    row[26] = _fmt_score(rag_utilization)
                    row[27] = _fmt_score(information_gain)
                    row[28] = _fmt_score(expansion_grounding)
                    row[31] = _fmt_score(aggregated_score)
                except Exception as e:
                    error_msg = (error_msg + " | " if error_msg else "") + f"规则指标: {e}"

                # ── 5. 准确性 + LLM-as-Judge（按需） ──
                if enable_fact_check or enable_safe_check or enable_llm_judge:
                    try:
                        async def _eval():
                            llm = _make_llm(provider, model)
                            evaluator = Evaluator(
                                llm_adapter=llm,
                                google_api_key=settings.google_api_key,
                                google_cse_id=getattr(settings, "google_cse_id", None),
                            )
                            return await evaluator.evaluate(
                                memoir_text=original_text,
                                generated_text=generated_text,
                                retrieval_result=retrieval_result,
                                use_llm=True,
                                enable_fact_check=enable_fact_check,
                                enable_safe_check=enable_safe_check,
                                enable_llm_judge=enable_llm_judge,
                                batch_size=int(batch_size),
                            )

                        eval_res = loop.run_until_complete(asyncio.wait_for(_eval(), timeout=240.0))

                        if eval_res.fact_check:
                            factscore = eval_res.fact_check.factscore
                            row[16] = _fmt_score(factscore, pct=True)
                        if eval_res.safe_check:
                            safe_score = eval_res.safe_check.safe_score
                            row[17] = _fmt_score(safe_score, pct=True)
                        if "relevance" in eval_res.scores:
                            relevance = eval_res.scores["relevance"].score
                            row[20] = f"{relevance:.1f}"
                        if "literary" in eval_res.scores:
                            literary = eval_res.scores["literary"].score
                            row[22] = f"{literary:.1f}"
                        if "compliance" in eval_res.scores:
                            compliance = eval_res.scores["compliance"].score
                            row[29] = f"{compliance:.1f}"
                        overall_score = eval_res.overall_score
                        row[30] = _fmt_score(overall_score)
                    except Exception as e:
                        error_msg = (error_msg + " | " if error_msg else "") + f"评估: {e}"

                seg_elapsed = time.monotonic() - seg_start
                row[32] = f"{seg_elapsed:.1f}"
                row[33] = "✅ 完成" if not error_msg else f"⚠️ {_truncate(error_msg, 80)}"

            except Exception as e:
                seg_elapsed = time.monotonic() - seg_start
                row[32] = f"{seg_elapsed:.1f}"
                row[33] = f"❌ {_truncate(str(e), 80)}"
                error_msg = (error_msg + " | " if error_msg else "") + str(e)

            detail_records.append({
                "source_file": source_file,
                "id": seg_id,
                "chapter": chapter,
                "historical_tag": tags,
                "retrieval_mode": run_mode,
                "retrieval_label": run_label,
                "original_text": original_text,
                "generated_text": generated_text,
                "scores": {
                    "ndcg_at_5": ndcg5,
                    "mrr": mrr,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "hit_at_3": hit_at_3,
                    "hit_at_5": hit_at_5,
                    "hit_at_10": hit_at_10,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "factscore": factscore,
                    "safe_score": safe_score,
                    "temporal_coherence": temporal_coherence,
                    "hallucination": hallucination,
                    "relevance": relevance,
                    "topic_coherence": topic_coherence,
                    "literary": literary,
                    "length_score": length_score,
                    "transition_usage": transition_usage,
                    "descriptive_richness": descriptive_richness,
                    "rag_utilization": rag_utilization,
                    "information_gain": information_gain,
                    "expansion_grounding": expansion_grounding,
                    "compliance": compliance,
                    "overall_score": overall_score,
                    "aggregated_score": aggregated_score,
                },
                "elapsed_seconds": round(seg_elapsed, 2),
                "error": error_msg or None,
            })
            yield (
                base_status + f"\n_实时平均值已更新：{len(detail_records)} / {total} 个任务已纳入统计。_",
                _benchmark_rows_snapshot(rows),
                current_averages_md(),
                None,
                None,
                None,
            )

        # ── 处理结束（正常完成或中途停止）：导出 JSON / Markdown / CSV ──
        done_count = len(rows)
        yield (
            f"💾 已处理 {done_count} / {total} 条段落"
            f"{'（已停止）' if stopped_early else ''}，"
            f"共耗时 {time.monotonic() - bench_start:.1f}s，正在导出…",
            _benchmark_rows_snapshot(rows), current_averages_md(), None, None, None,
        )

        out_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"benchmark_{ts}.json"
        md_path = out_dir / f"benchmark_{ts}.md"
        csv_path = out_dir / f"benchmark_{ts}.csv"

        # ── 计算平均值（用于 UI / JSON / Markdown / CSV）──
        averages = _compute_averages(detail_records)
        averages_by_mode = {
            label: _compute_averages([r for r in detail_records if r.get("retrieval_label") == label])
            for _, label in retrieval_specs
        }
        averages_md = _format_grouped_averages_md(detail_records)

        run_settings = {
            "provider": provider,
            "model": model,
            "style": style,
            "total_gen_min_chars": bg_lo,
            "total_gen_max_chars": bg_hi,
            "temperature": temperature,
            "retrieval_mode": retrieval_mode,
            "retrieval_modes": [
                {"mode": mode, "label": label}
                for mode, label in retrieval_specs
            ],
            "compare_plain_vector_rag": bool(compare_plain_vector_rag),
            "evals": {
                "retrieval_quality": enable_retrieval_quality,
                "fact_check": enable_fact_check,
                "safe_check": enable_safe_check,
                "llm_judge": enable_llm_judge,
            },
            "batch_size": int(batch_size),
            "use_rule_decompose": use_rule_decompose,
            "dataset_dir": dataset_dir,
            "loaded_files": loaded_files,
            "skipped_files": skipped_files,
            "timestamp": ts,
            "total_segments": segment_total,
            "total_jobs": total_jobs,
            "completed_segments": len(detail_records),
            "stopped_early": stopped_early,
            "total_elapsed_seconds": round(time.monotonic() - bench_start, 2),
        }

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "settings": run_settings,
                    "summary": {
                        "averages": averages,
                        "averages_by_mode": averages_by_mode,
                        "n": len(detail_records),
                    },
                    "results": detail_records,
                },
                f, ensure_ascii=False, indent=2,
            )

        # CSV
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "source_file", "id", "chapter", "historical_tag", "retrieval_mode",
                "original_text", "generated_text",
                "ndcg_at_5", "mrr", "precision", "recall", "f1",
                "hit_at_3", "hit_at_5", "hit_at_10", "retrieval_latency_ms",
                "factscore", "safe_score", "temporal_coherence", "hallucination",
                "relevance", "topic_coherence", "literary", "length_score",
                "transition_usage", "descriptive_richness", "rag_utilization",
                "information_gain", "expansion_grounding", "compliance",
                "overall_score", "aggregated_score",
                "elapsed_seconds", "error",
            ])
            for r in detail_records:
                s = r["scores"]
                w.writerow([
                    r.get("source_file", ""), r["id"], r["chapter"], "|".join(r["historical_tag"]),
                    r.get("retrieval_label") or r.get("retrieval_mode", ""),
                    r["original_text"], r["generated_text"],
                    s["ndcg_at_5"], s["mrr"], s["precision"], s["recall"], s["f1"],
                    s["hit_at_3"], s["hit_at_5"], s["hit_at_10"], s["retrieval_latency_ms"],
                    s["factscore"], s["safe_score"], s["temporal_coherence"], s["hallucination"],
                    s["relevance"], s["topic_coherence"], s["literary"], s["length_score"],
                    s["transition_usage"], s["descriptive_richness"], s["rag_utilization"],
                    s["information_gain"], s["expansion_grounding"], s["compliance"],
                    s["overall_score"], s["aggregated_score"],
                    r["elapsed_seconds"], r["error"] or "",
                ])
            # 平均值行
            w.writerow([])
            w.writerow([
                "AVERAGE", f"n={len(detail_records)}", "", "", "", "", "",
                averages.get("ndcg_at_5"), averages.get("mrr"),
                averages.get("precision"), averages.get("recall"), averages.get("f1"),
                averages.get("hit_at_3"), averages.get("hit_at_5"), averages.get("hit_at_10"),
                averages.get("retrieval_latency_ms"),
                averages.get("factscore"), averages.get("safe_score"),
                averages.get("temporal_coherence"), averages.get("hallucination"),
                averages.get("relevance"), averages.get("topic_coherence"),
                averages.get("literary"), averages.get("length_score"),
                averages.get("transition_usage"), averages.get("descriptive_richness"),
                averages.get("rag_utilization"), averages.get("information_gain"),
                averages.get("expansion_grounding"), averages.get("compliance"),
                averages.get("overall_score"), averages.get("aggregated_score"),
                averages.get("elapsed_seconds"), "",
            ])

        # Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 批量测试结果 ({ts})\n\n")
            f.write(f"- LLM: `{provider}`{f' / `{model}`' if model else ''}\n")
            f.write(f"- 风格: `{style}` | 目标生成字数: `{bg_lo}`-`{bg_hi}` | 温度: `{temperature}` | 检索: `{', '.join(label for _, label in retrieval_specs)}`\n")
            f.write(
                f"- 评估: 检索质量={enable_retrieval_quality}, "
                f"FActScore={enable_fact_check}, SAFE={enable_safe_check}, LLM-Judge={enable_llm_judge}\n"
            )
            f.write(f"- 数据来源: `{dataset_dir}`（{len(loaded_files)} 个文件: {', '.join(loaded_files)}）\n")
            if skipped_files:
                f.write(f"- 跳过: {', '.join(skipped_files)}\n")
            f.write(f"- 段落数: {segment_total} | 任务数: {total_jobs} | 总耗时: {run_settings['total_elapsed_seconds']}s\n\n")
            f.write("## 📊 指标平均\n\n")
            f.write(averages_md + "\n\n---\n\n")
            for r in detail_records:
                s = r["scores"]
                f.write(f"## {r['id']} — {r['chapter']}\n\n")
                f.write(f"**来源**: `{r.get('source_file', '')}`  |  ")
                f.write(f"**检索**: `{r.get('retrieval_label') or r.get('retrieval_mode', '')}`  |  ")
                f.write(f"**历史标签**: {'、'.join(r['historical_tag'])}\n\n")
                f.write(f"### 原文\n\n{r['original_text']}\n\n")
                f.write(f"### 扩写\n\n{r['generated_text'] or '_未生成_'}\n\n")
                f.write("### 评分\n\n")
                f.write(f"- nDCG@5: {_fmt_score(s['ndcg_at_5'])}\n")
                f.write(f"- MRR: {_fmt_score(s['mrr'])}\n")
                f.write(f"- Precision: {_fmt_score(s['precision'])}\n")
                f.write(f"- Recall: {_fmt_score(s['recall'])}\n")
                f.write(f"- F1: {_fmt_score(s['f1'])}\n")
                f.write(f"- Hit@3: {_fmt_score(s['hit_at_3'])}\n")
                f.write(f"- Hit@5: {_fmt_score(s['hit_at_5'])}\n")
                f.write(f"- Hit@10: {_fmt_score(s['hit_at_10'])}\n")
                f.write(f"- 检索耗时(ms): {_fmt_score(s['retrieval_latency_ms'])}\n")
                f.write(f"- FActScore: {_fmt_score(s['factscore'], pct=True)}\n")
                f.write(f"- SAFE: {_fmt_score(s['safe_score'], pct=True)}\n")
                f.write(f"- 时间一致性: {_fmt_score(s['temporal_coherence'])}\n")
                f.write(f"- 幻觉风险控制: {_fmt_score(s['hallucination'])}\n")
                f.write(f"- 相关性: {_fmt_score(s['relevance'])}\n")
                f.write(f"- 主题一致性: {_fmt_score(s['topic_coherence'])}\n")
                f.write(f"- 文学性: {_fmt_score(s['literary'])}\n")
                f.write(f"- 长度得分: {_fmt_score(s['length_score'])}\n")
                f.write(f"- 衔接词使用: {_fmt_score(s['transition_usage'])}\n")
                f.write(f"- 描写丰富度: {_fmt_score(s['descriptive_richness'])}\n")
                f.write(f"- RAG利用率: {_fmt_score(s['rag_utilization'])}\n")
                f.write(f"- 信息增量: {_fmt_score(s['information_gain'])}\n")
                f.write(f"- 扩展溯源率: {_fmt_score(s['expansion_grounding'])}\n")
                f.write(f"- 合规性: {_fmt_score(s['compliance'])}\n")
                f.write(f"- 综合分: {_fmt_score(s['overall_score'])}\n")
                f.write(f"- 段级综合分: {_fmt_score(s['aggregated_score'])}\n")
                f.write(f"- 耗时: {r['elapsed_seconds']}s\n")
                if r["error"]:
                    f.write(f"\n**错误**: {r['error']}\n")
                f.write("\n---\n\n")

        final_icon = "🛑" if stopped_early else "✅"
        final_label = "已停止" if stopped_early else "完成"
        if stopped_early and len(detail_records) < total_jobs:
            _save_benchmark_resume_state({
                "params": params,
                "dataset": dataset,
                "loaded_files": loaded_files,
                "skipped_files": skipped_files,
                "retrieval_specs": retrieval_specs,
                "segment_total": segment_total,
                "total_jobs": total_jobs,
                "bg_lo": bg_lo,
                "bg_hi": bg_hi,
                "single_cfg": single_cfg,
                "rows": rows,
                "detail_records": detail_records,
                "next_index": len(detail_records),
                "elapsed_seconds": time.monotonic() - bench_start,
            })
        else:
            _clear_benchmark_resume_state()
        _benchmark_running_event.clear()
        yield (
            f"{final_icon} {final_label}！已处理 {len(detail_records)} / {total_jobs} 个任务，"
            f"导出完毕（总耗时 {run_settings['total_elapsed_seconds']}s）。"
            f"{'点击「继续批量测试」可从下一条继续。' if stopped_early and len(detail_records) < total_jobs else ''}",
            _benchmark_rows_snapshot(rows),
            averages_md,
            str(json_path),
            str(md_path),
            str(csv_path),
        )
    finally:
        _benchmark_running_event.clear()
        try:
            loop.close()
        except Exception:
            pass


def build_index(llm_provider: str) -> str:
    """构建知识图谱索引"""
    try:
        builder = GraphBuilder(llm_provider=llm_provider)
        result = builder.build_index_sync()

        if result.success:
            stats = builder.get_index_stats()
            return f"""\u2705 索引构建成功！

**统计信息**:
- 输入文件: {stats['input_files']} 个
- 实体数量: {stats['entities']} 个
- 关系数量: {stats['relationships']} 个
- 社区数量: {stats['communities']} 个

输出目录: {result.output_dir}"""
        else:
            return f"\u274c 索引构建失败: {result.message}"

    except Exception as e:
        return f"\u274c 索引构建异常: {str(e)}"


def get_index_status() -> str:
    """获取索引状态"""
    try:
        builder = GraphBuilder()
        stats = builder.get_index_stats()

        if stats["indexed"]:
            return f"""\u2705 索引已就绪

**统计信息**:
- 输入文件: {stats['input_files']} 个
- 实体数量: {stats['entities']} 个
- 关系数量: {stats['relationships']} 个
- 社区数量: {stats['communities']} 个"""
        else:
            return f"""\u26a0\ufe0f 索引未构建

输入目录中有 {stats['input_files']} 个文件等待处理。
请点击"构建索引"按钮开始构建。"""

    except Exception as e:
        return f"\u274c 获取状态失败: {str(e)}"


def create_ui():
    """创建Gradio界面"""

    # 获取可用的LLM提供商
    available = get_available_providers()
    available_providers = [k for k, v in available.items() if v]
    if not available_providers:
        available_providers = ["deepseek"]  # 无可配密钥时的占位

    configured_provider = (settings.default_llm_provider or "").strip().lower()
    _default_provider = (
        configured_provider
        if configured_provider in available_providers
        else (available_providers[0] if available_providers else None)
    )
    _initial_model_choices = get_provider_models(_default_provider) if _default_provider else []
    configured_model = (settings.default_llm_model or "").strip()
    _initial_model_value = (
        configured_model
        if configured_model in _initial_model_choices
        else (_initial_model_choices[0] if _initial_model_choices else None)
    )

    with gr.Blocks(
        title="记忆图谱 - 历史背景注入系统",
    ) as app:

        gr.Markdown(
            """
            # \U0001f3ad 记忆图谱
            ### 基于RAG与知识图谱的个人回忆录历史背景自动注入系统

            输入您的回忆录片段，系统将自动检索相关历史背景，并生成具有文学性的描述文本。
            """,
            elem_classes=["main-title"]
        )

        with gr.Tabs():
            # 主功能标签页
            with gr.TabItem("\U0001f4dd 生成历史背景"):
                with gr.Row():
                    with gr.Column(scale=1):
                        memoir_input = gr.Textbox(
                            label="回忆录片段",
                            placeholder="输入您的回忆录文本，例如：\n1988年夏天，我从大学毕业，怀揣着梦想来到了深圳...",
                            lines=8,
                        )

                        demo_presets = [
                            (
                                "深圳打工\uff5c南下特区的第一份工",
                                "1998年夏天，我背着行李从内地来到深圳。火车站外的热浪裹着汽油味，霓虹把夜色照得发白。"
                                "我在华强北附近找了份电子厂的工作，白天流水线的机器声像潮水一样涌来，手指被焊锡烫出小泡。"
                                "晚上回到城中村的出租屋，楼道狭窄、风扇吱呀转，隔壁的普通话夹着各地口音。"
                                "我一边攒钱，一边学着适应这座快得让人喘不过气的城市，心里却始终留着一点不服输的劲。",
                            ),
                            (
                                "知青生活\uff5c下乡岁月与集体劳动",
                                "1972年冬天，我作为知青下乡到了北方的一个生产队。天还没亮就集合出工，镰刀与铁锹在霜里发冷。"
                                "春耕时脚陷在泥里拔不出来，秋收时肩上挑着沉甸甸的麻袋，手掌磨出厚茧。"
                                "夜里回到土屋，煤油灯火苗忽明忽暗，我们靠着一张旧桌子写信、抄歌、偷偷读几页书。"
                                "返城的消息时有时无，年轻的心在盼望与失落之间来回摇摆。可也是在这段日子里，我学会了忍耐，学会了与土地、与人群相处。",
                            ),
                            (
                                "下海经商\uff5c告别铁饭碗去闯荡",
                                "1992年春天，南巡讲话的消息传来，街头巷尾都在谈\u201c机会\u201d。我犹豫了许久，还是辞去单位的\u201c铁饭碗\u201d，跟着朋友南下闯一闯。"
                                "最开始是在批发市场摆摊，清晨抢货、午后跑客户、晚上对账算成本，兜里零钱叮当作响却不敢乱花。"
                                "行情好的时候一夜卖空，行情差时货压在仓里，心里像压着石头。"
                                "我学着看人脸色谈合作，学着在风险里做决定。那几年，机会与不确定并存，我也在改革开放的浪潮里一点点找到自己的位置。",
                            ),
                            (
                                "广州外贸\uff5c金融危机下的坚守",
                                "2009年春天，我背着一个旧行李箱，从湖南老家来到广州。那时的广州，到处都是机会的味道。"
                                "我在天河区租了一间小房子，月租800块，房间不大，但离市中心很近。每天早上，我都会挤地铁去白云区的一家外贸公司上班。"
                                "公司主要做纺织品出口，客户遍布欧美。那年生意特别难做，电视上天天说国外经济不好，很多同行公司都裁员甚至关门了。"
                                "记得那年4月份，老板让我去参加一个大型商品交易会，我第一次见到那么多外国商人，虽然他们比往年少了很多。"
                                "周末的时候，我喜欢去老城区逛逛，经常去一条叫上下九的步行街，吃一碗云吞面。"
                                "转过年来，2010年了，广州要办亚运会，到处都在建设，整个城市都是一片热火朝天的景象。",
                            ),
                        ]

                        with gr.Row():
                            provider_select = gr.Dropdown(
                                choices=available_providers,
                                value=_default_provider,
                                label="LLM 供应商",
                            )
                            model_select = gr.Dropdown(
                                choices=_initial_model_choices,
                                value=_initial_model_value,
                                label="模型",
                                visible=True,
                            )
                            style_select = gr.Dropdown(
                                choices=list(PromptTemplates.list_styles().keys()),
                                value="standard",
                                label="写作风格",
                            )

                        with gr.Row(visible=True) as total_gen_row:
                            total_gen_min_slider = gr.Slider(
                                minimum=200,
                                maximum=6000,
                                value=400,
                                step=50,
                                label="整篇 · 目标生成字数下限",
                                info="仅**单段模式**（未勾选分章）：与上限一起写入模型长度提示。",
                            )
                            total_gen_max_slider = gr.Slider(
                                minimum=300,
                                maximum=8000,
                                value=800,
                                step=50,
                                label="整篇 · 目标生成字数上限",
                                info="分章模式下此两项隐藏，请改用下方「每章」区间。",
                            )

                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="创意度 (Temperature)",
                        )

                        retrieval_mode_select = gr.Radio(
                            choices=[
                                ("GraphRAG 关键词检索", "keyword"),
                                ("GraphRAG Local Search（实体向量+图谱）", "vector"),
                                ("GraphRAG Hybrid（Local+关键词）", "hybrid"),
                                ("普通 RAG（向量 baseline）", PLAIN_VECTOR_RAG_MODE),
                            ],
                            value="vector",
                            label="检索模式",
                            info="选择 GraphRAG 或普通向量 RAG 检索策略",
                        )

                        with gr.Group():
                            gr.Markdown("**\U0001f50d 评估模块**（可独立开关）")
                            retrieval_quality_checkbox = gr.Checkbox(
                                value=True,
                                label="\U0001f3af 检索质量",
                                info="LLM 评判检索到的实体/关系与回忆录的契合度",
                            )
                            fact_check_checkbox = gr.Checkbox(
                                value=True,
                                label="\u2705 事实准确性 (FActScore)",
                                info="基于知识库逐条核查原子事实",
                            )
                            safe_check_checkbox = gr.Checkbox(
                                value=True,
                                label="\U0001f310 事实准确性 (SAFE独立知识验证)",
                                info="使用 LLM + SAFE 框架独立验证事实，不依赖知识库",
                            )
                            llm_judge_checkbox = gr.Checkbox(
                                value=True,
                                label="\u270d\ufe0f 相关性 / 文学性 / 合规性",
                                info="由 LLM 对生成内容进行多维度打分",
                            )
                            rule_decompose_checkbox = gr.Checkbox(
                                value=False,
                                label="\u26a1 使用规则拆分（事实检查加速）",
                                info="使用规则而非 LLM 进行原子事实拆分；仅在事实准确性/SAFE 启用时生效",
                            )
                            batch_size_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="\U0001f9ee 事实验证批次大小 (batch_size)",
                                info="每次 LLM 调用同时验证的原子事实数；越大调用越少但单次响应更长。仅在事实准确性/SAFE 启用时生效",
                            )

                        chapter_mode_checkbox = gr.Checkbox(
                            value=False,
                            label="分章/长文模式",
                            info="数千字长文：分段检索与生成后合并；默认仍为整篇单段",
                        )

                        with gr.Column(visible=False) as chapter_advanced_col:
                            gr.Markdown(
                                "**分章模式 · 每章长度**：下方区间对**每一章成品**生效（各章共用同一套提示）。"
                                "回忆录原文如何切段由系统内部固定偏好（约 300–800 字）完成。"
                            )
                            chapter_gen_min_slider = gr.Slider(
                                minimum=200,
                                maximum=4000,
                                value=400,
                                step=50,
                                label="每章 · 目标生成字数下限",
                            )
                            chapter_gen_max_slider = gr.Slider(
                                minimum=300,
                                maximum=6000,
                                value=800,
                                step=50,
                                label="每章 · 目标生成字数上限",
                            )
                            chapter_llm_parse_checkbox = gr.Checkbox(
                                value=False,
                                label="分章检索使用 LLM 解析",
                                info="对应 run_long_form_generation(use_llm_parsing)",
                            )
                            chapter_max_gate_retries_slider = gr.Slider(
                                minimum=0,
                                maximum=3,
                                value=2,
                                step=1,
                                label="质量门控未通过时的定向章节重试次数",
                                info="仅分章模式：失败章节按需重新生成并重评，设为 0 则关闭自动重试。",
                            )

                        generate_btn = gr.Button("\U0001f680 生成历史背景", variant="primary")
                        refresh_ollama_models_btn = gr.Button(
                            "\U0001f504 刷新 Ollama 模型列表",
                            variant="secondary",
                            visible=(_default_provider == "ollama"),
                        )

                        gr.Markdown("**Demo 场景预置（位于底部，选择即载入）**")
                        demo_select = gr.Dropdown(
                            choices=[label for label, _ in demo_presets],
                            value=None,
                            label="选择 Demo 场景（选择后自动载入到上方文本框）",
                        )

                        def _load_demo(selected_label: str) -> str:
                            for label, text in demo_presets:
                                if label == selected_label:
                                    return text
                            return ""

                        demo_select.change(
                            fn=_load_demo,
                            inputs=[demo_select],
                            outputs=[memoir_input],
                        )

                        def _chapter_mode_visibility(on: bool):
                            return gr.update(visible=on), gr.update(visible=not on)

                        chapter_mode_checkbox.change(
                            fn=_chapter_mode_visibility,
                            inputs=[chapter_mode_checkbox],
                            outputs=[chapter_advanced_col, total_gen_row],
                        )

                    with gr.Column(scale=1):
                        output_text = gr.Markdown(
                            value="",
                            elem_classes=["output-box"],
                            label="生成的历史背景（Markdown）",
                            sanitize_html=False,
                        )

                        with gr.Accordion("\U0001f4cb 提取与检索信息", open=False):
                            extracted_info = gr.Markdown(label="提取的信息")

                        with gr.Accordion("\U0001f3af 评估：检索质量", open=False):
                            retrieval_quality_output = gr.Markdown(label="检索质量")

                        with gr.Accordion("\u2705 评估：事实准确性", open=True):
                            gr.Markdown("#### \U0001f4da 基于知识库 (FActScore)")
                            accuracy_output = gr.Markdown(label="事实准确性 (KB)")
                            gr.Markdown("---\n#### \U0001f310 独立知识验证 (SAFE)")
                            safe_check_output = gr.Markdown(label="独立知识验证")

                        with gr.Accordion("\U0001f4dd 新内容评估", open=False):
                            novel_content_output = gr.Markdown(
                                label="RAG 利用率 · 新实体使用 · 新事实溯源（分章）",
                            )

                        with gr.Accordion("\U0001f517 评估：相关性", open=False):
                            relevance_output = gr.Markdown(label="相关性")

                        with gr.Accordion("\u270d\ufe0f 评估：文学性", open=False):
                            literary_output = gr.Markdown(label="文学性")

                        with gr.Accordion("\U0001f6e1\ufe0f 评估：合规性", open=False):
                            compliance_output = gr.Markdown(label="合规性")

                        with gr.Accordion("\U0001f6a6 质量门控", open=False):
                            quality_gate_output = gr.Markdown(
                                label="跨章阈值检查与修复建议（分章）",
                            )

                generate_btn.click(
                    fn=process_memoir_stream,
                    inputs=[
                        memoir_input,
                        provider_select,
                        model_select,
                        style_select,
                        total_gen_min_slider,
                        total_gen_max_slider,
                        temperature_slider,
                        fact_check_checkbox,
                        retrieval_mode_select,
                        rule_decompose_checkbox,
                        safe_check_checkbox,
                        retrieval_quality_checkbox,
                        llm_judge_checkbox,
                        chapter_mode_checkbox,
                        batch_size_slider,
                        chapter_gen_min_slider,
                        chapter_gen_max_slider,
                        chapter_llm_parse_checkbox,
                        chapter_max_gate_retries_slider,
                    ],
                    outputs=[
                        output_text,
                        extracted_info,
                        retrieval_quality_output,
                        accuracy_output,
                        safe_check_output,
                        novel_content_output,
                        relevance_output,
                        literary_output,
                        compliance_output,
                        quality_gate_output,
                    ],
                )

                def _update_model_choices(provider: Optional[str]):
                    """切换供应商时刷新模型列表，并控制「刷新 Ollama」按钮可见性。"""
                    if not provider:
                        return (
                            gr.update(choices=[], value=None, visible=True),
                            gr.update(visible=False),
                        )
                    models = get_provider_models(provider)
                    configured_model = (settings.default_llm_model or "").strip()
                    configured_provider = (settings.default_llm_provider or "").strip().lower()
                    value = (
                        configured_model
                        if provider == configured_provider and configured_model in models
                        else (models[0] if models else None)
                    )
                    show_ollama_refresh = provider == "ollama"
                    return (
                        gr.update(choices=models, value=value, visible=True),
                        gr.update(visible=show_ollama_refresh),
                    )

                provider_select.change(
                    fn=_update_model_choices,
                    inputs=[provider_select],
                    outputs=[model_select, refresh_ollama_models_btn],
                )

                app.load(
                    fn=_update_model_choices,
                    inputs=[provider_select],
                    outputs=[model_select, refresh_ollama_models_btn],
                )

                def _refresh_ollama_models(provider: str, current_value: Optional[str]):
                    if provider != "ollama":
                        return gr.update(), gr.update(visible=False)
                    models = get_provider_models("ollama")
                    value = (
                        current_value
                        if current_value and current_value in models
                        else (models[0] if models else None)
                    )
                    return (
                        gr.update(choices=models, value=value, visible=True),
                        gr.update(visible=True),
                    )

                refresh_ollama_models_btn.click(
                    fn=_refresh_ollama_models,
                    inputs=[provider_select, model_select],
                    outputs=[model_select, refresh_ollama_models_btn],
                )

            # 多模型对比标签页
            with gr.TabItem("\U0001f504 多模型对比"):
                gr.Markdown("使用多个LLM同时生成，对比不同模型的输出效果。")

                with gr.Row():
                    with gr.Column(scale=1):
                        compare_input = gr.Textbox(
                            label="回忆录片段",
                            placeholder="输入回忆录文本...",
                            lines=6,
                        )

                        providers_checkbox = gr.CheckboxGroup(
                            choices=available_providers,
                            value=available_providers[:2] if len(available_providers) >= 2 else available_providers,
                            label="选择要对比的模型",
                        )

                        compare_temp = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="创意度",
                        )

                        compare_btn = gr.Button("\U0001f504 开始对比", variant="primary")

                    with gr.Column(scale=1):
                        compare_output = gr.Markdown(label="对比结果")

                compare_btn.click(
                    fn=compare_providers,
                    inputs=[compare_input, providers_checkbox, compare_temp],
                    outputs=[compare_output],
                )

            # 批量测试标签页
            with gr.TabItem("\U0001f9ea 批量测试"):
                gr.Markdown(
                    f"""
                    ## 批量测试 (Benchmark)

                    在指定文件夹下的所有 `*.json` 测试集上系统性地跑扩写生成。

                    - **数据来源**：默认 `{BENCHMARK_DEFAULT_DIR.relative_to(Path(__file__).resolve().parent.parent)}`，
                      可在下方输入框修改；所有 schema 合法的 `*.json` 会被合并成一份测试集。
                    - **参数复用**：LLM / 模型 / 风格 / **整篇目标生成字数** / 温度 / 检索模式 / 评估开关 / batch_size 等，
                      全部沿用「\U0001f4dd 生成历史背景」标签页当前的设置。
                    - **扩写文本**：每次都会生成（与评估开关无关）。
                    - **评估模块**：按主标签页当前勾选状态执行；未勾选的不跑。
                    - **进度**：逐条流式更新；跑完后**对所有数值指标求平均**并展示在 UI，同时导出
                      JSON / Markdown / CSV 三种格式供下载（导出文件均含平均值汇总）。
                    """
                )

                bench_dataset_dir = gr.Textbox(
                    label="测试集目录",
                    value=str(BENCHMARK_DEFAULT_DIR),
                    placeholder="data/memoirs/segments/",
                    info="可填绝对路径或相对项目根目录的相对路径；目录下所有 *.json 会被合并使用。",
                )

                bench_compare_plain_rag_checkbox = gr.Checkbox(
                    value=True,
                    label="对比普通 RAG（向量 baseline）",
                    info="开启后，每条样本会同时跑当前 GraphRAG 检索模式和普通向量 RAG，各生成一行结果。",
                )

                with gr.Row():
                    bench_btn = gr.Button("▶️ 开始批量测试", variant="primary")
                    bench_stop_resume_btn = gr.Button("⏹️ 停止 / ▶️ 继续", variant="secondary")

                bench_status = gr.Markdown(
                    value="点击「开始批量测试」启动；运行中可点同一个控制按钮停止，停止完成后再点同一个按钮继续。",
                )

                bench_averages = gr.Markdown(
                    value="_尚未开始_",
                    label="📊 指标平均值",
                )

                bench_table = gr.Dataframe(
                    headers=BENCHMARK_COLUMNS,
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="结果（逐条更新）",
                )

                with gr.Row():
                    bench_json_file = gr.File(label="\U0001f4c4 JSON")
                    bench_md_file = gr.File(label="\U0001f4dd Markdown")
                    bench_csv_file = gr.File(label="\U0001f4ca CSV")

                bench_btn.click(
                    fn=batch_benchmark_stream,
                    inputs=[
                        bench_dataset_dir,
                        provider_select,
                        model_select,
                        style_select,
                        total_gen_min_slider,
                        total_gen_max_slider,
                        temperature_slider,
                        fact_check_checkbox,
                        retrieval_mode_select,
                        rule_decompose_checkbox,
                        safe_check_checkbox,
                        retrieval_quality_checkbox,
                        llm_judge_checkbox,
                        batch_size_slider,
                        bench_compare_plain_rag_checkbox,
                    ],
                    outputs=[
                        bench_status,
                        bench_table,
                        bench_averages,
                        bench_json_file,
                        bench_md_file,
                        bench_csv_file,
                        bench_stop_resume_btn,
                    ],
                )

                # 同一个控制按钮：运行中请求停止；停止导出后再点则从下一条继续
                bench_stop_resume_btn.click(
                    fn=benchmark_stop_or_resume_stream,
                    inputs=[],
                    outputs=[
                        bench_status,
                        bench_table,
                        bench_averages,
                        bench_json_file,
                        bench_md_file,
                        bench_csv_file,
                        bench_stop_resume_btn,
                    ],
                    queue=False,
                )

            # 索引管理标签页
            with gr.TabItem("\u2699\ufe0f 索引管理"):
                gr.Markdown("""
                ## \U0001f4ca 知识图谱索引管理

                索引构建会使用 LLM 从历史事件数据中提取实体和关系，构建知识图谱。

                **支持的 LLM 模型：**
                | 提供商 | 环境变量 | 默认模型 |
                |-------|----------|---------|
                | Gemini | `GOOGLE_API_KEY` | gemini-2.5-flash |
                | DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat / deepseek-v4-flash / deepseek-v4-pro |
                | Qwen | `QWEN_API_KEY` | qwen-plus |
                | 智谱GLM | `GLM_API_KEY` | glm-4.7-flash |
                | OpenAI | `OPENAI_API_KEY` | gpt-4o / gpt-5（UI 可选） |
                | 混元 | `HUNYUAN_API_KEY` | hunyuan-lite 等（OpenAI 兼容） |

                **注意**：首次构建索引需要几分钟时间，会产生一定的 API 费用。
                """)

                with gr.Row():
                    with gr.Column():
                        index_status = gr.Markdown(value=get_index_status())

                        refresh_btn = gr.Button("\U0001f504 刷新状态")
                        refresh_btn.click(fn=get_index_status, outputs=[index_status])

                    with gr.Column():
                        index_provider_choices = list(available_providers)
                        index_default_provider = (
                            _default_provider
                            if _default_provider in index_provider_choices
                            else (index_provider_choices[0] if index_provider_choices else None)
                        )

                        index_provider = gr.Dropdown(
                            choices=index_provider_choices,
                            value=index_default_provider,
                            label="构建索引使用的 LLM",
                            info="选择用于构建知识图谱的语言模型（仅显示已在 .env 中配置密钥的供应商，含本地 Ollama）",
                        )

                        build_btn = gr.Button("\U0001f528 构建索引", variant="primary")
                        build_output = gr.Markdown()

                        build_btn.click(
                            fn=build_index,
                            inputs=[index_provider],
                            outputs=[build_output],
                        )

            # 使用说明标签页
            with gr.TabItem("\U0001f4d6 使用说明"):
                gr.Markdown("""
                ## 使用指南

                ### 1. 准备工作

                1. 在 `.env` 文件中配置您的 LLM API 密钥
                2. 将历史事件数据放入 `data/input/` 目录
                3. 构建知识图谱索引

                ### 2. 生成历史背景

                1. 在"生成历史背景"标签页输入回忆录片段
                2. 选择 LLM 供应商、模型和写作风格
                3. 调整创意度参数
                4. 点击"生成历史背景"按钮

                ### 3. 多模型对比

                1. 在"多模型对比"标签页输入回忆录片段
                2. 选择多个 LLM 模型
                3. 点击"开始对比"查看不同模型的输出

                ### 4. 写作风格说明

                - **standard**: 标准风格，平衡的文学性描述
                - **nostalgic**: 怀旧风格，温暖回忆的笔调
                - **narrative**: 叙事融合，与个人故事深度交织
                - **informative**: 简洁信息，重点突出的背景介绍
                - **conversational**: 对话风格，像讲故事一样亲切

                ### 5. 支持的 LLM 模型

                - Deepseek
                - Qwen (通义千问)
                - Hunyuan (腾讯混元)
                - Google Gemini
                - 智谱GLM
                - OpenAI GPT
                """)

        gr.Markdown(
            """
            ---
            **记忆图谱** - 让历史为您的回忆增添厚度 | Capstone Project 2026
            """
        )

    return app


_THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="gray")
_CSS = """
.main-title { text-align: center; margin-bottom: 20px; }
.output-box { min-height: 200px; }
div.prose.output-box .graphrag-loading.loading-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 220px;
  padding: 32px 20px 40px;
  gap: 20px;
  box-sizing: border-box;
  margin: 0 auto;
  max-width: 100%;
}
div.prose.output-box .graphrag-loading .loading-dots {
  display: flex;
  gap: 10px;
  align-items: center;
}
div.prose.output-box .graphrag-loading .loading-dots span {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--primary-500, #3b82f6);
  animation: graphragDotPulse 1.4s ease-in-out infinite;
}
div.prose.output-box .graphrag-loading .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
div.prose.output-box .graphrag-loading .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes graphragDotPulse {
  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}
div.prose.output-box .graphrag-loading .loading-bar {
  display: block;
  width: 80%;
  max-width: 320px;
  height: 6px;
  border-radius: 3px;
  overflow: hidden;
  background: var(--neutral-200, #e5e7eb);
}
div.prose.output-box .graphrag-loading .loading-bar::after {
  content: '';
  display: block;
  width: 38%;
  height: 100%;
  border-radius: 3px;
  background: linear-gradient(
    90deg,
    var(--primary-400, #60a5fa),
    var(--primary-600, #2563eb)
  );
  animation: graphragBarSlide 1.35s ease-in-out infinite;
}
@keyframes graphragBarSlide {
  0% { transform: translateX(-105%); }
  100% { transform: translateX(320%); }
}
div.prose.output-box .graphrag-loading .loading-text {
  font-size: 0.95rem;
  color: var(--body-text-color-subdued, #6b7280);
  text-align: center;
  line-height: 1.65;
  max-width: 36em;
}
div.prose.output-box .graphrag-loading .loading-text .loading-sub {
  display: inline-block;
  margin-top: 0.35em;
  font-size: 0.9em;
  font-weight: 500;
  color: var(--body-text-color, #374151);
}
"""

if __name__ == "__main__":
    app = create_ui()
    app.queue()
    app.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=False,
        theme=_THEME,
        css=_CSS,
    )
