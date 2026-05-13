"""
生成详细的Pipeline测试报告
展示：输入 -> 检索 -> 生成 -> 评估的完整流程
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def main():
    # 导入模块
    from src.llm import create_llm_adapter
    from src.retrieval import MemoirRetriever
    from src.generation import LiteraryGenerator
    from src.evaluation.long_form_eval import evaluate_long_form
    import json


    print("=" * 80)
    print("详细 Pipeline 测试报告生成")
    print("=" * 80)

    # 1. 初始化组件
    print("\n[1/5] 初始化组件...")
    llm = create_llm_adapter(provider='openai', model='gpt-4o')
    retriever = MemoirRetriever(
        index_dir='data/graphrag_output',
        llm_adapter=llm
    )
    generator = LiteraryGenerator(llm)
    print("  ✓ 组件初始化完成")
    
    # 2. 准备测试数据
    print("\n[2/5] 准备测试数据...")
    memoir_text = """1980年，我考上了大学。那是恢复高考后的第三年，整个国家都在发生变化。
父亲激动得一夜没睡，母亲在一旁默默流泪。我知道，这张通知书不仅是我的，
也是全家人的梦想。

入学那天，我背着简单的行李来到校园。宿舍里住着来自五湖四海的同学，
大家都充满了对未来的憧憬。晚上，我们围坐在一起，聊着各自的家乡和梦想。"""
    
    print(f"  回忆录长度: {len(memoir_text)} 字")
    
    # 3. 执行生成
    print("\n[3/5] 执行生成...")
    result = await generator.generate_long_form(
        memoir_text=memoir_text,
        retriever=retriever,
        target_min_chars=100,
        target_max_chars=200,
        use_llm_parsing=True,
        retrieval_mode='keyword',
        style='standard',
        temperature=0.7,
    )
    print(f"  ✓ 生成完成，共 {len(result.chapters)} 章")
    
    # 4. 执行评估
    print("\n[4/5] 执行评估...")
    eval_result = await evaluate_long_form(
        result,
        llm_adapter=llm,
        use_llm_eval=False,
        enable_fact_check=True
    )
    print("  ✓ 评估完成")
    
    # 5. 生成报告
    print("\n[5/5] 生成报告...")
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("Pipeline 详细测试报告")
    report_lines.append("=" * 100)
    
    # 输入部分
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【输入】原始回忆录")
    report_lines.append("=" * 100)
    report_lines.append(f"\n长度: {len(memoir_text)} 字\n")
    report_lines.append("内容:")
    report_lines.append("-" * 100)
    report_lines.append(memoir_text)
    report_lines.append("-" * 100)
    
    # 每章详细信息
    for i, chapter in enumerate(result.chapters):
        report_lines.append("\n" + "=" * 100)
        report_lines.append(f"【第 {i+1} 章】")
        report_lines.append("=" * 100)
        
        # 原文片段
        report_lines.append("\n>>> 原文片段:")
        report_lines.append("-" * 100)
        report_lines.append(chapter.segment_text)
        report_lines.append("-" * 100)

        # 检索结果
        report_lines.append("\n>>> 检索结果:")
        report_lines.append("-" * 100)

        if hasattr(chapter, 'retrieval_result') and chapter.retrieval_result:
            rr = chapter.retrieval_result

            # 实体
            if hasattr(rr, 'entities') and rr.entities:
                report_lines.append(f"\n实体 (共 {len(rr.entities)} 个):")
                for j, ent in enumerate(rr.entities[:10], 1):
                    name = ent.get('name', ent.get('title', ''))
                    ent_type = ent.get('type', 'ENTITY')
                    desc = ent.get('description', '')
                    report_lines.append(f"\n  {j}. {name} ({ent_type})")
                    desc_short = desc[:150] + "..." if len(desc) > 150 else desc
                    report_lines.append(f"     描述: {desc_short}")
                if len(rr.entities) > 10:
                    report_lines.append(f"\n  ... 还有 {len(rr.entities) - 10} 个实体")

            # 关系
            if hasattr(rr, 'relationships') and rr.relationships:
                report_lines.append(f"\n关系 (共 {len(rr.relationships)} 个):")
                for j, rel in enumerate(rr.relationships[:5], 1):
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    desc = rel.get('description', '')
                    report_lines.append(f"\n  {j}. {source} -> {target}")
                    desc_short = desc[:150] + "..." if len(desc) > 150 else desc
                    report_lines.append(f"     描述: {desc_short}")
                if len(rr.relationships) > 5:
                    report_lines.append(f"\n  ... 还有 {len(rr.relationships) - 5} 个关系")
            
            # 新内容摘要
            if hasattr(rr, '_novel_content_brief') and rr._novel_content_brief:
                brief = rr._novel_content_brief
                report_lines.append(f"\n新内容摘要:")
                report_lines.append(f"  - 对齐实体: {len(brief.aligned_entities)} 个")
                report_lines.append(f"  - 新增实体: {len(brief.novel_entities)} 个")
                
                if brief.novel_entities:
                    report_lines.append(f"\n  新增实体列表:")
                    for j, ent in enumerate(brief.novel_entities[:10], 1):
                        report_lines.append(f"    {j}. {ent['name']} ({ent['type']})")
                        desc = ent['description'][:100] + "..." if len(ent['description']) > 100 else ent['description']
                        report_lines.append(f"       {desc}")
        
        report_lines.append("-" * 100)

        # 传入大模型的完整 prompt
        if hasattr(chapter, 'generation') and hasattr(chapter.generation, 'prompt') and chapter.generation.prompt:
            report_lines.append("\n>>> 传入大模型的完整 Prompt:")
            report_lines.append("-" * 100)
            report_lines.append("\n【System Prompt】")
            if hasattr(chapter.generation, 'system_prompt') and chapter.generation.system_prompt:
                report_lines.append(chapter.generation.system_prompt)
            else:
                report_lines.append("（未记录）")
            report_lines.append("\n【User Prompt】")
            report_lines.append(chapter.generation.prompt)
            report_lines.append("-" * 100)

        # 生成内容
        report_lines.append("\n>>> 生成内容:")
        report_lines.append("-" * 100)
        report_lines.append(chapter.generation.content)
        report_lines.append("-" * 100)
        report_lines.append(f"\n生成长度: {len(chapter.generation.content)} 字")
        
        # 评估结果
        if i < len(eval_result.segments):
            seg_eval = eval_result.segments[i]
            report_lines.append("\n>>> 评估结果:")
            report_lines.append("-" * 100)

            # 指标 - 分组显示
            report_lines.append("\n【核心指标】")
            core_metrics = ["entity_coverage", "temporal_coherence", "rag_entity_accuracy", "topic_coherence"]
            for metric_name in core_metrics:
                if metric_name in seg_eval.metrics:
                    metric_value = seg_eval.metrics[metric_name]
                    if hasattr(metric_value, 'value'):
                        report_lines.append(f"  - {metric_name}: {metric_value.value:.4f}")
                        # 显示详细说明
                        if hasattr(metric_value, 'explanation'):
                            # 处理多行 explanation
                            explanation_lines = metric_value.explanation.split('\n')
                            for line in explanation_lines:
                                report_lines.append(f"    {line}")
                    else:
                        report_lines.append(f"  - {metric_name}: {metric_value}")

            report_lines.append("\n【文学性指标】")
            literary_metrics = ["length_score", "paragraph_structure", "transition_usage", "descriptive_richness"]
            for metric_name in literary_metrics:
                if metric_name in seg_eval.metrics:
                    metric_value = seg_eval.metrics[metric_name]
                    if hasattr(metric_value, 'value'):
                        report_lines.append(f"  - {metric_name}: {metric_value.value:.4f}")
                        # 显示详细说明
                        if hasattr(metric_value, 'explanation'):
                            explanation_lines = metric_value.explanation.split('\n')
                            for line in explanation_lines:
                                report_lines.append(f"    {line}")
                    else:
                        report_lines.append(f"  - {metric_name}: {metric_value}")

            report_lines.append("\n【新内容指标】")
            novel_metrics = ["information_gain", "expansion_grounding", "rag_utilization", "hallucination"]
            for metric_name in novel_metrics:
                if metric_name in seg_eval.metrics:
                    metric_value = seg_eval.metrics[metric_name]
                    if hasattr(metric_value, 'value'):
                        report_lines.append(f"  - {metric_name}: {metric_value.value:.4f}")
                        # 显示详细说明
                        if hasattr(metric_value, 'explanation'):
                            explanation_lines = metric_value.explanation.split('\n')
                            for line in explanation_lines:
                                report_lines.append(f"    {line}")
                    else:
                        report_lines.append(f"  - {metric_name}: {metric_value}")

            # 新内容分析（保留旧格式以兼容）
            if seg_eval.novel_content_info:
                nci = seg_eval.novel_content_info
                report_lines.append(f"\n【新内容分析（详细）】")
                extracted_facts = nci.get('extracted_facts', [])
                report_lines.append(f"  - 提取到的新事实: {len(extracted_facts)} 个")
                if extracted_facts:
                    report_lines.append(f"    示例: {', '.join(extracted_facts[:10])}")

                grounded_facts = nci.get('grounded_facts', [])
                report_lines.append(f"  - 有依据的新事实: {len(grounded_facts)} 个")
                if grounded_facts:
                    report_lines.append(f"    示例: {', '.join(grounded_facts[:10])}")

                ungrounded_facts = nci.get('ungrounded_facts', [])
                report_lines.append(f"  - 无依据的新事实: {len(ungrounded_facts)} 个")
                if ungrounded_facts:
                    report_lines.append(f"    ⚠️  示例: {', '.join(ungrounded_facts[:10])}")

                report_lines.append(f"  - 信息增益: {nci.get('information_gain', 0):.1%}")
                report_lines.append(f"  - 扩展溯源率: {nci.get('expansion_grounding', 0):.1%}")

            report_lines.append("-" * 100)
    
    # 整体评估
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【整体评估】")
    report_lines.append("=" * 100)
    
    report_lines.append(f"\n综合分数: {eval_result.aggregated_score:.2f}/10")
    
    if eval_result.document_metrics:
        report_lines.append(f"\n文档级指标:")
        for metric_name, metric_value in eval_result.document_metrics.items():
            if hasattr(metric_value, 'value'):
                report_lines.append(f"  - {metric_name}: {metric_value.value:.2f}")
            else:
                report_lines.append(f"  - {metric_name}: {metric_value}")
    
    report_lines.append(f"\n质量门控: {'✓ 通过' if eval_result.quality_gate.passed else '✗ 未通过'}")

    # 收集失败的检查项
    failed_items = []
    for cr in eval_result.quality_gate.chapter_results:
        if not cr.passed:
            for issue in cr.issues:
                if issue.severity == "error":
                    failed_items.append(f"第{cr.chapter_index+1}章-{issue.dimension}")
    for ci in eval_result.quality_gate.cross_chapter_issues:
        if ci.severity == "error":
            failed_items.append(f"跨章-{ci.dimension}")

    if failed_items:
        report_lines.append(f"  失败检查: {', '.join(failed_items)}")
    
    report_lines.append("\n" + "=" * 100)
    report_lines.append("报告结束")
    report_lines.append("=" * 100)
    
    # 保存报告
    report_content = "\n".join(report_lines)
    output_path = project_root / "PIPELINE_DETAILED_REPORT.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n✓ 报告已保存到: {output_path}")
    print(f"  文件大小: {len(report_content)} 字符")
    
    # 同时输出到控制台
    print("\n" + report_content)


if __name__ == "__main__":
    asyncio.run(main())
