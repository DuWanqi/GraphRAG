"""
记忆图谱 Web 应用
基于 Gradio 构建的用户界面
"""

import asyncio
import time
from typing import Optional, List, Tuple
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
    chapter_mode: bool = False,
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
            f"### 📋 提取的上下文信息\n\n"
            f"| 字段 | 值 |\n"
            f"|------|-----|\n"
            f"| **时间** | {context.year or '未识别'} |\n"
            f"| **地点** | {context.location or '未识别'} |\n"
            f"| **关键词** | {', '.join(context.keywords) if context.keywords else '无'} |\n"
            f"| **查询** | `{retrieval_result.query}` |\n\n"
            f"### 🔍 检索结果统计\n\n"
            f"| 类型 | 数量 |\n"
            f"|------|------|\n"
            f"| **实体** | {len(retrieval_result.entities)} 个 |\n"
            f"| **关系** | {len(retrieval_result.relationships)} 个 |\n"
            f"| **社区报告** | {len(retrieval_result.communities)} 个 |\n"
            f"| **相关文本** | {len(retrieval_result.text_units)} 段 |"
        )
        
        # 添加检索到的实体详情（展示所有）
        if retrieval_result.entities:
            extracted_md += "\n\n### 📌 检索到的实体\n\n"
            for i, entity in enumerate(retrieval_result.entities, 1):
                name = entity.get('name', '未知')
                desc = entity.get('description', '')[:100]
                entity_type = entity.get('type', '未知类型')
                extracted_md += f"**{i}. {name}** ({entity_type})\n\n{desc}...\n\n"
        
        # 添加检索到的关系详情（展示所有）
        if retrieval_result.relationships:
            extracted_md += "\n### 🔗 检索到的关系\n\n"
            for i, rel in enumerate(retrieval_result.relationships, 1):
                source = rel.get('source', '未知')
                target = rel.get('target', '未知')
                rel_type = rel.get('type', '关联')
                desc = rel.get('description', '')[:80]
                extracted_md += f"**{i}. {source} → {target}** ({rel_type})\n\n{desc}...\n\n"
        
        # 添加检索到的文本单元详情（展示所有）
        if retrieval_result.text_units:
            extracted_md += "\n### 📝 相关文本片段\n\n"
            for i, text_unit in enumerate(retrieval_result.text_units, 1):
                # text_unit 可能是字符串或字典
                if isinstance(text_unit, dict):
                    content = text_unit.get('content', '')[:200]
                else:
                    content = str(text_unit)[:200]
                extracted_md += f"**片段 {i}:** {content}...\n\n"
        
        print(f"[DEBUG] extracted_md length: {len(extracted_md)} chars")
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

        # ── 2. 检索质量评估 (LLM-as-a-Judge) ──
        # 注意：跳过检索质量评估以减少API调用，避免速率限制
        # 如果需要评估，可以单独运行评估脚本
        retrieval_q_md = "检索质量评估已跳过（避免API速率限制）"
        yield ("", extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

        # ── 3. 流式生成 ──
        async def _stream():
            try:
                async for delta in generator.generate_stream(
                    memoir_text=memoir_text,
                    retrieval_result=retrieval_result,
                    style=style,
                    length_hint=single_cfg["length_hint"],
                    temperature=temperature,
                    max_tokens=single_cfg["max_tokens"],
                ):
                    yield delta
            except Exception as e:
                import traceback
                error_msg = f"\n\n[生成错误: {type(e).__name__}: {e}]"
                print(f"[ERROR] 流式生成失败: {e}")
                print(traceback.format_exc())
                yield error_msg

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
            except Exception as e:
                content += f"\n\n[生成异常: {e}]"
                yield (content, extracted_md, retrieval_q_md, accuracy_md, relevance_md, literary_md, compliance_md)
                break
            content += delta
            yield (content, extracted_md, retrieval_q_md, accuracy_md, safe_md, relevance_md, literary_md, compliance_md)

        # ── 4. 综合评估（事实准确性 + 相关性 + 文学性 + 合规性 + SAFE独立验证） ──
        # 注意：跳过综合评估以减少API调用，避免速率限制
        if (enable_fact_check or enable_safe_check) and content:
            accuracy_md = "事实准确性评估已跳过（避免API速率限制）"
            safe_md = "SAFE验证已跳过（避免API速率限制）"
            relevance_md = "相关性评估已跳过（避免API速率限制）"
            literary_md = "文学性评估已跳过（避免API速率限制）"
            compliance_md = "合规性评估已跳过（避免API速率限制）"

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
    global retriever

    if not memoir_text.strip():
        return "请输入回忆录文本"

    if not selected_providers:
        return "请至少选择一个LLM提供商"

    try:
        # 创建多LLM路由器
        router = LLMRouter()

        # 创建生成器
        gen = LiteraryGenerator(llm_router=router)

        # 使用全局检索器，避免重复加载索引
        if retriever is None:
            retriever = MemoirRetriever()
        retrieval_result = retriever.retrieve_sync(memoir_text, top_k=10)

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

                        fact_check_checkbox = gr.Checkbox(
                            value=True,
                            label="\U0001f50d 启用评估",
                            info="检测生成内容的事实准确性、相关性、文学性和合规性",
                        )

                        rule_decompose_checkbox = gr.Checkbox(
                            value=False,
                            label="\u26a1 使用规则拆分",
                            info="使用规则而非LLM进行原子事实拆分，大幅提高速度",
                        )

                        safe_check_checkbox = gr.Checkbox(
                            value=False,
                            label="\U0001f310 独立知识验证 (SAFE)",
                            info="使用LLM自身知识独立验证事实，不依赖知识库，帮助评估知识库完整性",
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

                        with gr.Accordion("\U0001f3af 检索质量", open=False):
                            retrieval_quality_output = gr.Markdown(label="检索质量")

                        with gr.Accordion("\u2705 事实准确性 (KB)", open=True):
                            accuracy_output = gr.Markdown(label="事实准确性")

                        with gr.Accordion("\U0001f310 独立知识验证 (SAFE)", open=False):
                            safe_check_output = gr.Markdown(label="独立知识验证")

                        with gr.Accordion("\U0001f517 相关性 (LLM-as-a-Judge)", open=False):
                            relevance_output = gr.Markdown(label="相关性")

                        with gr.Accordion("\u270d\ufe0f 文学性 (LLM-as-a-Judge)", open=False):
                            literary_output = gr.Markdown(label="文学性")

                        with gr.Accordion("\U0001f6e1\ufe0f 合规性", open=False):
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
                        chapter_mode_checkbox,
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
