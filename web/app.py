"""
记忆图谱 Web 应用
基于 Gradio 构建的用户界面
"""

import asyncio
import csv
import json
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import gradio as gr

from src.config import get_settings
from src.llm import (
    create_llm_adapter,
    get_available_providers,
    LLMRouter,
    build_ollama_model_choices,
)
from src.indexing import GraphBuilder, DataLoader
from src.retrieval import MemoirRetriever
from src.generation import (
    LiteraryGenerator,
    PromptTemplates,
    segment_memoir,
    run_long_form_generation,
    single_segment_generation_config,
    estimate_long_form_generation_timeout,
    estimate_long_form_evaluation_timeout,
    build_long_form_eval_options,
)
from src.evaluation import (
    Evaluator,
    EvaluationDimension,
    FActScoreChecker,
    SAFECheckResult,
    evaluate_retrieval_quality,
    evaluate_long_form,
    long_form_eval_to_json,
)


# 全局变量
settings = get_settings()
retriever: Optional[MemoirRetriever] = None
generator: Optional[LiteraryGenerator] = None
current_provider: Optional[str] = None  # 跟踪当前使用的 provider
current_model: Optional[str] = None  # 跟踪当前使用的模型（目前仅用于 ollama）

# 8 个输出的空元组，用于错误/提前返回
_EMPTY_8 = ("", "", "", "", "", "", "", "")


def init_components(provider: str = "gemini", model: Optional[str] = None):
    """初始化组件"""
    global retriever, generator, current_provider, current_model

    # 如果 provider 相同且组件已初始化，则跳过
    if (
        provider == current_provider
        and (provider != "ollama" or model == current_model)
        and retriever is not None
        and generator is not None
    ):
        return f"✅ 已使用 {provider}{f'/{model}' if model else ''} 模型"

    try:
        llm_adapter = create_llm_adapter(provider=provider, model=(model if provider == "ollama" else None))
        retriever = MemoirRetriever(llm_adapter=llm_adapter)
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        current_provider = provider
        current_model = model if provider == "ollama" else None
        return f"✅ 初始化成功！使用 {provider}{f'/{model}' if model else ''} 模型"
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"


def _make_llm(provider, model):
    """创建 LLM 适配器的快捷方式"""
    return create_llm_adapter(provider=provider, model=(model if provider == "ollama" else None))


def _format_retrieval_quality(eval_result):
    """格式化检索质量评估结果为 Markdown"""
    md = f"**nDCG@3**: {eval_result['ndcg_at_3']:.4f}  "
    md += f"**nDCG@5**: {eval_result['ndcg_at_5']:.4f}  "
    md += f"**nDCG@10**: {eval_result['ndcg_at_10']:.4f}\n\n"
    md += f"**MRR**: {eval_result['mrr']:.4f}\n"
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


async def process_memoir_async(
    memoir_text: str,
    provider: str,
    model: Optional[str],
    style: str,
    length_bucket: str,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "keyword",
    use_rule_decompose: bool = False,
    chapter_mode: bool = False,
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
        or (provider == "ollama" and current_model != model)
    ):
        init_result = init_components(provider, model=model)
        if "失败" in init_result:
            return init_result, "", "", ""
    
    try:
        start_time = time.time()
        
        print(f"[DEBUG] 开始处理回忆录，长度: {len(memoir_text)} 字符")

        if chapter_mode:
            segs = segment_memoir(memoir_text)
            n_ch = max(1, len(segs))
            budget_timeout = estimate_long_form_generation_timeout(n_ch)
            lf = await asyncio.wait_for(
                run_long_form_generation(
                    memoir_text,
                    retriever,
                    generator,
                    length_bucket=length_bucket,
                    style=style,
                    temperature=temperature,
                    retrieval_mode=retrieval_mode,
                    use_llm_parsing=False,
                ),
                timeout=budget_timeout,
            )
            gen_time = time.time() - start_time
            lines = [f"**分章模式** 共 {len(lf.chapters)} 章（输入约 {len(memoir_text)} 字）"]
            for i, ch in enumerate(lf.chapters):
                ctx = ch.retrieval_result.context
                lines.append(
                    f"- 第{i + 1}章: 查询 `{ch.retrieval_result.query}` | "
                    f"年 {ctx.year or '—'} 地 {ctx.location or '—'}"
                )
            extracted_info = "\n".join(lines)
            rent = [len(ch.retrieval_result.entities) for ch in lf.chapters]
            retrieval_info = (
                f"**分章检索**: 各章实体数 {rent}；"
                f"关系总计 {sum(len(ch.retrieval_result.relationships) for ch in lf.chapters)}"
            )
            fact_check_info = f"**生成总耗时**: {gen_time:.2f} 秒\n"
            if enable_fact_check:
                try:
                    llm_adapter = create_llm_adapter(
                        provider=provider, model=(model if provider == "ollama" else None)
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
            retriever.retrieve(
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
        
        retrieval_info = f"""**找到实体**: {len(retrieval_result.entities)} 个
**找到关系**: {len(retrieval_result.relationships)} 个
**社区报告**: {len(retrieval_result.communities)} 个
**相关文本**: {len(retrieval_result.text_units)} 段"""
        
        if retrieval_result.entities:
            retrieval_info += "\n\n**主要实体**:\n"
            for entity in retrieval_result.entities[:5]:
                retrieval_info += f"- {entity.get('name', '未知')}: {entity.get('description', '')[:100]}...\n"
        
        # 生成文本
        gen_start = time.time()
        print("[DEBUG] 开始生成文本...")
        single_cfg = single_segment_generation_config(length_bucket)

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
                llm_adapter = create_llm_adapter(provider=provider, model=(model if provider == "ollama" else None))
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
    length_bucket: str,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "keyword",
    use_rule_decompose: bool = False,
    chapter_mode: bool = False,
) -> Tuple[str, str, str, str]:
    """处理回忆录（同步包装）"""
    return asyncio.run(
        process_memoir_async(
            memoir_text,
            provider,
            model,
            style,
            length_bucket,
            temperature,
            enable_fact_check,
            retrieval_mode,
            use_rule_decompose,
            chapter_mode,
        )
    )


def process_memoir_stream(
    memoir_text: str,
    provider: str,
    model: Optional[str],
    style: str,
    length_bucket: str,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "keyword",
    use_rule_decompose: bool = False,
    enable_safe_check: bool = False,
    enable_retrieval_quality: bool = True,
    enable_llm_judge: bool = True,
    chapter_mode: bool = False,
    batch_size: int = 5,
):
    """
    Gradio 流式生成。
    yields: (output_text, extracted_info, retrieval_quality, accuracy, safe_check, relevance, literary, compliance)
    """
    global retriever, generator, current_provider, current_model

    if not memoir_text.strip():
        yield ("请输入回忆录文本",) + ("",) * 7
        return
    if not provider:
        yield ("请先选择 LLM 模型（供应商）",) + ("",) * 7
        return

    if (
        retriever is None
        or generator is None
        or current_provider != provider
        or (provider == "ollama" and current_model != model)
    ):
        init_result = init_components(provider, model=model)
        if "失败" in init_result:
            yield (init_result,) + ("",) * 7
            return

    single_cfg = single_segment_generation_config(length_bucket)

    # 各栏目状态
    extracted_md = ""
    retrieval_q_md = ""
    accuracy_md = ""
    safe_md = ""
    relevance_md = ""
    literary_md = ""
    compliance_md = ""

    loop = asyncio.new_event_loop()
    try:
        if chapter_mode:
            segs = segment_memoir(memoir_text)
            n_ch = max(1, len(segs))
            yield (f"[分章] 共 {len(segs)} 段，依次检索与生成…", "", "", "", "", "", "", "")

            async def _long_form():
                return await run_long_form_generation(
                    memoir_text,
                    retriever,
                    generator,
                    length_bucket=length_bucket,
                    style=style,
                    temperature=temperature,
                    retrieval_mode=retrieval_mode,
                    use_llm_parsing=False,
                )

            budget_timeout = estimate_long_form_generation_timeout(n_ch)
            lf = loop.run_until_complete(asyncio.wait_for(_long_form(), timeout=budget_timeout))
            lines_md = [f"**分章模式** 共 {len(lf.chapters)} 章"]
            for i, ch in enumerate(lf.chapters):
                ctx = ch.retrieval_result.context
                lines_md.append(
                    f"- 第{i + 1}章: 查询 `{ch.retrieval_result.query}` | "
                    f"年 {ctx.year or '—'} 地 {ctx.location or '—'}"
                )
            extracted_md = "\n".join(lines_md)
            rent = [len(ch.retrieval_result.entities) for ch in lf.chapters]
            retrieval_q_md = f"**分章检索**: 各章实体数 {rent}"
            content = lf.merged_content
            yield (content, extracted_md, retrieval_q_md, "", "", "", "", "")

            accuracy_md = ""
            if enable_fact_check:
                try:
                    llm_adapter = create_llm_adapter(
                        provider=provider, model=(model if provider == "ollama" else None)
                    )
                    eval_kwargs = build_long_form_eval_options(
                        llm_adapter=llm_adapter,
                        use_llm_eval=False,
                        enable_fact_check=True,
                        max_atomic_facts_per_segment=12,
                        fact_check_timeout_per_segment=45.0,
                        use_rule_decompose=use_rule_decompose,
                        batch_size=int(batch_size),
                    )
                    ev = loop.run_until_complete(
                        asyncio.wait_for(
                            evaluate_long_form(lf, **eval_kwargs),
                            timeout=estimate_long_form_evaluation_timeout(len(lf.chapters)),
                        )
                    )
                    accuracy_md = "### 长文评估汇总\n\n" + ev.summary_text
                    accuracy_md += "\n\n```json\n" + long_form_eval_to_json(ev)[:8000] + "\n```\n"
                except Exception as e:
                    accuracy_md = f"长文评估未完成: {e}"
            yield (content, extracted_md, retrieval_q_md, accuracy_md, "", "", "", "")
            return

        yield ("\u23f3 正在检索相关历史背景，请稍候\u2026", "", "", "", "", "", "", "")

        # ── 1. 检索 ──
        retrieval_result = loop.run_until_complete(asyncio.wait_for(
            retriever.retrieve(
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
            f"**找到实体**: {len(retrieval_result.entities)} 个  "
            f"**找到关系**: {len(retrieval_result.relationships)} 个  "
            f"**社区报告**: {len(retrieval_result.communities)} 个  "
            f"**相关文本**: {len(retrieval_result.text_units)} 段"
        )
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

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
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

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
                yield (content, extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)
                break
            content += delta
            yield (content, extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

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

        yield (content, extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)
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
    "原文",
    "扩写文本",
    "nDCG@5",
    "MRR",
    "FActScore",
    "SAFE",
    "相关性",
    "文学性",
    "合规性",
    "耗时(s)",
    "状态",
]

# 用于求平均的数值指标键 → 显示名 / 是否百分比
BENCHMARK_METRIC_KEYS = [
    ("ndcg_at_5", "nDCG@5", False),
    ("mrr", "MRR", False),
    ("factscore", "FActScore", True),
    ("safe_score", "SAFE", True),
    ("relevance", "相关性", False),
    ("literary", "文学性", False),
    ("compliance", "合规性", False),
    ("elapsed_seconds", "平均耗时(s)", False),
]


_benchmark_stop_event = threading.Event()


def request_benchmark_stop() -> str:
    """由「停止」按钮调用：设置标志位，请求批量任务在当前段落结束后中止。"""
    _benchmark_stop_event.set()
    return "🛑 已请求停止：将在当前段落处理完成后中止，并导出已完成部分的结果。"


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


def _truncate(text: Optional[str], n: int = 100) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _fmt_score(v, pct: bool = False) -> str:
    if v is None:
        return "—"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def batch_benchmark_stream(
    dataset_dir: str,
    provider: str,
    model: Optional[str],
    style: str,
    length_bucket: str,
    temperature: float,
    enable_fact_check: bool,
    retrieval_mode: str,
    use_rule_decompose: bool,
    enable_safe_check: bool,
    enable_retrieval_quality: bool,
    enable_llm_judge: bool,
    batch_size: int,
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
    empty_avg = "_尚未开始_"

    if not provider:
        yield ("⚠️ 请先在「生成历史背景」标签页选择 LLM 模型（供应商）。", [], empty_avg, None, None, None)
        return

    # 复用 / 必要时初始化检索器与生成器
    if (
        retriever is None
        or generator is None
        or current_provider != provider
        or (provider == "ollama" and current_model != model)
    ):
        init_msg = init_components(provider, model=model)
        if "失败" in init_msg:
            yield (init_msg, [], empty_avg, None, None, None)
            return

    try:
        dataset, loaded_files, skipped_files = _load_benchmark_dataset(
            dataset_dir or str(BENCHMARK_DEFAULT_DIR)
        )
    except Exception as e:
        yield (f"❌ 读取测试集目录失败: {e}", [], empty_avg, None, None, None)
        return

    total = len(dataset)
    if total == 0:
        skip_note = f"\n跳过的文件: {', '.join(skipped_files)}" if skipped_files else ""
        yield (f"⚠️ 目录下未找到有效测试集 *.json。{skip_note}", [], empty_avg, None, None, None)
        return

    single_cfg = single_segment_generation_config(length_bucket)

    rows: List[List[str]] = []
    detail_records: List[dict] = []
    bench_start = time.monotonic()

    files_note = f" 来源: {len(loaded_files)} 个文件 ({', '.join(loaded_files)})"
    if skipped_files:
        files_note += f" · 跳过: {', '.join(skipped_files)}"
    yield (
        f"⏳ 共 {total} 条段落待处理。{files_note}\n\n"
        f"当前配置：{provider}{f'/{model}' if model else ''} · "
        f"风格 `{style}` · 字数 `{length_bucket}` · 温度 `{temperature}` · 检索 `{retrieval_mode}`",
        rows,
        empty_avg,
        None,
        None,
        None,
    )

    loop = asyncio.new_event_loop()
    stopped_early = False
    try:
        for idx, item in enumerate(dataset, start=1):
            # 段落级停止检查：只在新段落开始前响应停止请求
            if _benchmark_stop_event.is_set():
                stopped_early = True
                yield (
                    f"🛑 已停止：在第 {idx} / {total} 条段落开始前中止。已完成 {len(rows)} 条，正在导出…",
                    rows, empty_avg, None, None, None,
                )
                break

            source_file = item.get("source_file", "")
            seg_id = item.get("id", f"#{idx}")
            chapter = item.get("chapter", "")
            tags = item.get("historical_tag", []) or []
            tags_str = "、".join(tags)
            original_text = item.get("original_text", "")

            seg_start = time.monotonic()
            base_status = (
                f"### 进度: {idx} / {total}\n\n"
                f"- 来源: `{source_file}`\n"
                f"- 当前: **{seg_id}** — {chapter}\n"
                f"- 历史标签: {tags_str}\n"
                f"- 已耗时: {time.monotonic() - bench_start:.1f}s\n"
            )
            yield (base_status + "\n_正在检索…_", rows, empty_avg, None, None, None)

            row = [
                source_file, seg_id, chapter, tags_str,
                _truncate(original_text, 120), "",
                "—", "—", "—", "—", "—", "—", "—",
                "—", "进行中",
            ]

            generated_text = ""
            ndcg5 = mrr = factscore = safe_score = relevance = literary = compliance = None
            error_msg = ""
            retrieval_result = None

            try:
                # ── 1. 检索 ──
                retrieval_result = loop.run_until_complete(asyncio.wait_for(
                    retriever.retrieve(
                        original_text, top_k=10, use_llm_parsing=False, mode=retrieval_mode,
                    ),
                    timeout=60.0,
                ))

                # ── 2. 扩写（必跑） ──
                yield (base_status + "\n_正在生成扩写…_", rows, empty_avg, None, None, None)
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
                row[5] = _truncate(generated_text, 120)
                yield (base_status + "\n_扩写完成，进入评估…_", rows, empty_avg, None, None, None)

                # ── 3. 检索质量（按需） ──
                if enable_retrieval_quality:
                    try:
                        eval_result = loop.run_until_complete(asyncio.wait_for(
                            evaluate_retrieval_quality(
                                query_text=original_text,
                                text_units=retrieval_result.text_units,
                                llm_adapter=_make_llm(provider, model),
                            ),
                            timeout=60.0,
                        ))
                        ndcg5 = eval_result.get("ndcg_at_5")
                        mrr = eval_result.get("mrr")
                        row[6] = _fmt_score(ndcg5)
                        row[7] = _fmt_score(mrr)
                    except Exception as e:
                        row[6] = "失败"
                        row[7] = "失败"
                        error_msg = (error_msg + " | " if error_msg else "") + f"检索评估: {e}"

                # ── 4. 准确性 + LLM-as-Judge（按需） ──
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
                            row[8] = _fmt_score(factscore, pct=True)
                        if eval_res.safe_check:
                            safe_score = eval_res.safe_check.safe_score
                            row[9] = _fmt_score(safe_score, pct=True)
                        if "relevance" in eval_res.scores:
                            relevance = eval_res.scores["relevance"].score
                            row[10] = f"{relevance:.1f}"
                        if "literary" in eval_res.scores:
                            literary = eval_res.scores["literary"].score
                            row[11] = f"{literary:.1f}"
                        if "compliance" in eval_res.scores:
                            compliance = eval_res.scores["compliance"].score
                            row[12] = f"{compliance:.1f}"
                    except Exception as e:
                        error_msg = (error_msg + " | " if error_msg else "") + f"评估: {e}"

                seg_elapsed = time.monotonic() - seg_start
                row[13] = f"{seg_elapsed:.1f}"
                row[14] = "✅ 完成" if not error_msg else f"⚠️ {_truncate(error_msg, 30)}"

            except Exception as e:
                seg_elapsed = time.monotonic() - seg_start
                row[13] = f"{seg_elapsed:.1f}"
                row[14] = f"❌ {_truncate(str(e), 30)}"
                error_msg = (error_msg + " | " if error_msg else "") + str(e)

            rows.append(row)
            detail_records.append({
                "source_file": source_file,
                "id": seg_id,
                "chapter": chapter,
                "historical_tag": tags,
                "original_text": original_text,
                "generated_text": generated_text,
                "scores": {
                    "ndcg_at_5": ndcg5,
                    "mrr": mrr,
                    "factscore": factscore,
                    "safe_score": safe_score,
                    "relevance": relevance,
                    "literary": literary,
                    "compliance": compliance,
                },
                "elapsed_seconds": round(seg_elapsed, 2),
                "error": error_msg or None,
            })
            yield (base_status, rows, empty_avg, None, None, None)

        # ── 处理结束（正常完成或中途停止）：导出 JSON / Markdown / CSV ──
        done_count = len(rows)
        yield (
            f"💾 已处理 {done_count} / {total} 条段落"
            f"{'（已停止）' if stopped_early else ''}，"
            f"共耗时 {time.monotonic() - bench_start:.1f}s，正在导出…",
            rows, empty_avg, None, None, None,
        )

        out_dir = Path(tempfile.mkdtemp(prefix="benchmark_"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = out_dir / f"benchmark_{ts}.json"
        md_path = out_dir / f"benchmark_{ts}.md"
        csv_path = out_dir / f"benchmark_{ts}.csv"

        # ── 计算平均值（用于 UI / JSON / Markdown / CSV）──
        averages = _compute_averages(detail_records)
        averages_md = _format_averages_md(averages, len(detail_records))

        run_settings = {
            "provider": provider,
            "model": model,
            "style": style,
            "length_bucket": length_bucket,
            "temperature": temperature,
            "retrieval_mode": retrieval_mode,
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
            "total_segments": total,
            "completed_segments": len(detail_records),
            "stopped_early": stopped_early,
            "total_elapsed_seconds": round(time.monotonic() - bench_start, 2),
        }

        # JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "settings": run_settings,
                    "summary": {"averages": averages, "n": len(detail_records)},
                    "results": detail_records,
                },
                f, ensure_ascii=False, indent=2,
            )

        # CSV
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "source_file", "id", "chapter", "historical_tag",
                "original_text", "generated_text",
                "ndcg_at_5", "mrr", "factscore", "safe_score",
                "relevance", "literary", "compliance",
                "elapsed_seconds", "error",
            ])
            for r in detail_records:
                s = r["scores"]
                w.writerow([
                    r.get("source_file", ""), r["id"], r["chapter"], "|".join(r["historical_tag"]),
                    r["original_text"], r["generated_text"],
                    s["ndcg_at_5"], s["mrr"], s["factscore"], s["safe_score"],
                    s["relevance"], s["literary"], s["compliance"],
                    r["elapsed_seconds"], r["error"] or "",
                ])
            # 平均值行
            w.writerow([])
            w.writerow([
                "AVERAGE", f"n={len(detail_records)}", "", "", "", "",
                averages.get("ndcg_at_5"), averages.get("mrr"),
                averages.get("factscore"), averages.get("safe_score"),
                averages.get("relevance"), averages.get("literary"),
                averages.get("compliance"),
                averages.get("elapsed_seconds"), "",
            ])

        # Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 批量测试结果 ({ts})\n\n")
            f.write(f"- LLM: `{provider}`{f' / `{model}`' if model else ''}\n")
            f.write(f"- 风格: `{style}` | 字数: `{length_bucket}` | 温度: `{temperature}` | 检索: `{retrieval_mode}`\n")
            f.write(
                f"- 评估: 检索质量={enable_retrieval_quality}, "
                f"FActScore={enable_fact_check}, SAFE={enable_safe_check}, LLM-Judge={enable_llm_judge}\n"
            )
            f.write(f"- 数据来源: `{dataset_dir}`（{len(loaded_files)} 个文件: {', '.join(loaded_files)}）\n")
            if skipped_files:
                f.write(f"- 跳过: {', '.join(skipped_files)}\n")
            f.write(f"- 段落数: {total} | 总耗时: {run_settings['total_elapsed_seconds']}s\n\n")
            f.write("## 📊 指标平均\n\n")
            f.write(averages_md + "\n\n---\n\n")
            for r in detail_records:
                s = r["scores"]
                f.write(f"## {r['id']} — {r['chapter']}\n\n")
                f.write(f"**来源**: `{r.get('source_file', '')}`  |  ")
                f.write(f"**历史标签**: {'、'.join(r['historical_tag'])}\n\n")
                f.write(f"### 原文\n\n{r['original_text']}\n\n")
                f.write(f"### 扩写\n\n{r['generated_text'] or '_未生成_'}\n\n")
                f.write("### 评分\n\n")
                f.write(f"- nDCG@5: {_fmt_score(s['ndcg_at_5'])}\n")
                f.write(f"- MRR: {_fmt_score(s['mrr'])}\n")
                f.write(f"- FActScore: {_fmt_score(s['factscore'], pct=True)}\n")
                f.write(f"- SAFE: {_fmt_score(s['safe_score'], pct=True)}\n")
                f.write(f"- 相关性: {_fmt_score(s['relevance'])}\n")
                f.write(f"- 文学性: {_fmt_score(s['literary'])}\n")
                f.write(f"- 合规性: {_fmt_score(s['compliance'])}\n")
                f.write(f"- 耗时: {r['elapsed_seconds']}s\n")
                if r["error"]:
                    f.write(f"\n**错误**: {r['error']}\n")
                f.write("\n---\n\n")

        final_icon = "🛑" if stopped_early else "✅"
        final_label = "已停止" if stopped_early else "完成"
        yield (
            f"{final_icon} {final_label}！已处理 {len(detail_records)} / {total} 条段落，"
            f"导出完毕（总耗时 {run_settings['total_elapsed_seconds']}s）。",
            rows,
            averages_md,
            str(json_path),
            str(md_path),
            str(csv_path),
        )
    finally:
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
        available_providers = ["deepseek"]  # 默认

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
                        ]

                        with gr.Row():
                            provider_select = gr.Dropdown(
                                choices=available_providers,
                                value=None,
                                label="LLM 模型",
                            )
                            ollama_model_select = gr.Dropdown(
                                choices=build_ollama_model_choices(settings.ollama_api_base),
                                value=None,
                                label="Ollama 模型",
                                visible=False,
                            )
                            style_select = gr.Dropdown(
                                choices=list(PromptTemplates.list_styles().keys()),
                                value="standard",
                                label="写作风格",
                            )

                        length_bucket_select = gr.Radio(
                            choices=[
                                ("200-400字", "200-400"),
                                ("400-800字", "400-800"),
                                ("800-1200字", "800-1200"),
                                ("1200字以上", "1200+"),
                            ],
                            value="400-800",
                            label="生成字数",
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
                                ("关键词检索 (快速)", "keyword"),
                                ("向量检索 (精准)", "vector"),
                                ("混合检索 (推荐)", "hybrid"),
                            ],
                            value="keyword",
                            label="检索模式",
                            info="选择知识图谱检索策略",
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

                        generate_btn = gr.Button("\U0001f680 生成历史背景", variant="primary")
                        refresh_ollama_models_btn = gr.Button("\U0001f504 刷新 Ollama 模型列表", variant="secondary")

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

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="生成的历史背景",
                            lines=10,
                            elem_classes=["output-box"],
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

                        with gr.Accordion("\U0001f517 评估：相关性", open=False):
                            relevance_output = gr.Markdown(label="相关性")

                        with gr.Accordion("\u270d\ufe0f 评估：文学性", open=False):
                            literary_output = gr.Markdown(label="文学性")

                        with gr.Accordion("\U0001f6e1\ufe0f 评估：合规性", open=False):
                            compliance_output = gr.Markdown(label="合规性")

                generate_btn.click(
                    fn=process_memoir_stream,
                    inputs=[
                        memoir_input,
                        provider_select,
                        ollama_model_select,
                        style_select,
                        length_bucket_select,
                        temperature_slider,
                        fact_check_checkbox,
                        retrieval_mode_select,
                        rule_decompose_checkbox,
                        safe_check_checkbox,
                        retrieval_quality_checkbox,
                        llm_judge_checkbox,
                        chapter_mode_checkbox,
                        batch_size_slider,
                    ],
                    outputs=[
                        output_text,
                        extracted_info,
                        retrieval_quality_output,
                        accuracy_output,
                        safe_check_output,
                        relevance_output,
                        literary_output,
                        compliance_output,
                    ],
                )

                def _update_ollama_model_ui(provider: str):
                    if provider == "ollama":
                        choices = build_ollama_model_choices(settings.ollama_api_base)
                        value = choices[0][1] if choices else None
                        return gr.update(visible=True, choices=choices, value=value)
                    return gr.update(visible=False, value=None)

                provider_select.change(
                    fn=_update_ollama_model_ui,
                    inputs=[provider_select],
                    outputs=[ollama_model_select],
                )

                app.load(
                    fn=_update_ollama_model_ui,
                    inputs=[provider_select],
                    outputs=[ollama_model_select],
                )

                def _refresh_ollama_models(provider: str, current_value: Optional[str]):
                    if provider != "ollama":
                        return gr.update()
                    choices = build_ollama_model_choices(settings.ollama_api_base)
                    value = current_value if any(v == current_value for _, v in choices) else (choices[0][1] if choices else None)
                    return gr.update(choices=choices, value=value, visible=True)

                refresh_ollama_models_btn.click(
                    fn=_refresh_ollama_models,
                    inputs=[provider_select, ollama_model_select],
                    outputs=[ollama_model_select],
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
                    - **参数复用**：LLM / 模型 / 风格 / 字数 / 温度 / 检索模式 / 评估开关 / batch_size 等，
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

                with gr.Row():
                    bench_btn = gr.Button("▶️ 开始批量测试", variant="primary")
                    bench_stop_btn = gr.Button("⏹️ 停止", variant="stop")

                bench_status = gr.Markdown(
                    value="点击「开始批量测试」启动；运行中可点「停止」让任务在当前段落跑完后中止。",
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
                        ollama_model_select,
                        style_select,
                        length_bucket_select,
                        temperature_slider,
                        fact_check_checkbox,
                        retrieval_mode_select,
                        rule_decompose_checkbox,
                        safe_check_checkbox,
                        retrieval_quality_checkbox,
                        llm_judge_checkbox,
                        batch_size_slider,
                    ],
                    outputs=[
                        bench_status,
                        bench_table,
                        bench_averages,
                        bench_json_file,
                        bench_md_file,
                        bench_csv_file,
                    ],
                )

                # 停止按钮：queue=False 让它绕过队列，立刻把停止标志位置上
                bench_stop_btn.click(
                    fn=request_benchmark_stop,
                    inputs=[],
                    outputs=[bench_status],
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
                | DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
                | Qwen | `QWEN_API_KEY` | qwen-plus |
                | 智谱GLM | `GLM_API_KEY` | glm-4.7-flash |
                | OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
                | 混元 | `HUNYUAN_API_KEY` | hunyuan-lite |

                **注意**：首次构建索引需要几分钟时间，会产生一定的 API 费用。
                """)

                with gr.Row():
                    with gr.Column():
                        index_status = gr.Markdown(value=get_index_status())

                        refresh_btn = gr.Button("\U0001f504 刷新状态")
                        refresh_btn.click(fn=get_index_status, outputs=[index_status])

                    with gr.Column():
                        # 完整的模型选择列表
                        all_providers = ["gemini", "deepseek", "qwen", "openai", "hunyuan", "glm"]
                        default_provider = available_providers[0] if available_providers else "gemini"

                        index_provider = gr.Dropdown(
                            choices=all_providers,
                            value=default_provider,
                            label="构建索引使用的 LLM",
                            info="选择用于构建知识图谱的语言模型",
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
                2. 选择 LLM 模型和写作风格
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
