"""
完整 Pipeline 测试
测试从回忆录输入到分章生成输出的完整流程
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['GRAPHRAG_OUTPUT_DIR'] = 'data/graphrag_output'


async def test_full_pipeline():
    """测试完整的生成 pipeline"""
    
    print("=" * 80)
    print("完整 Pipeline 测试")
    print("=" * 80)
    
    # 1. 导入模块
    print("\n[1/6] 导入模块...")
    from src.llm import create_llm_adapter
    from src.retrieval import MemoirRetriever
    from src.generation import LiteraryGenerator
    from src.evaluation import evaluate_long_form
    
    # 2. 初始化组件
    print("\n[2/6] 初始化组件...")

    # Pipeline 配置
    pipeline_config = {
        "llm_provider": "openai",
        "llm_model": "gpt-4o",
        "index_dir": "data/graphrag_output",
        "retrieval_mode": "keyword",
        "style": "standard",
        "temperature": 0.7,
        "target_min_chars": 100,
        "target_max_chars": 200,
        "use_llm_parsing": True,
    }

    print("\n  Pipeline 配置:")
    for key, value in pipeline_config.items():
        print(f"    - {key}: {value}")

    # 创建 LLM 适配器
    llm_adapter = create_llm_adapter(
        provider=pipeline_config["llm_provider"],
        model=pipeline_config["llm_model"],
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    print(f"\n  ✓ LLM 适配器: {pipeline_config['llm_provider']} {pipeline_config['llm_model']}")

    # 创建检索器
    retriever = MemoirRetriever(
        index_dir=pipeline_config["index_dir"],
        llm_adapter=llm_adapter,
    )
    print(f"  ✓ 检索器: GraphRAG (索引目录: {pipeline_config['index_dir']})")

    # 创建生成器
    generator = LiteraryGenerator(llm_adapter=llm_adapter)
    print(f"  ✓ 生成器: LiteraryGenerator")
    
    # 3. 准备测试数据
    print("\n[3/6] 准备测试数据...")
    
    memoir_text = """
1980年，我考上了大学。那是恢复高考后的第三年，整个国家都在发生变化。
父亲激动得一夜没睡，母亲在一旁默默流泪。我知道，这张通知书不仅是我的，
也是全家人的梦想。

入学那天，我背着简单的行李来到校园。宿舍里住着来自五湖四海的同学，
大家都充满了对未来的憧憬。晚上，我们围坐在一起，聊着各自的家乡和梦想。
"""
    
    print(f"  回忆录长度: {len(memoir_text)} 字")
    print(f"  回忆录预览: {memoir_text[:100]}...")
    
    # 4. 执行分章生成
    print("\n[4/6] 执行分章生成...")
    print("  这可能需要几分钟，请耐心等待...")
    
    try:
        result = await generator.generate_long_form(
            memoir_text=memoir_text,
            retriever=retriever,
            target_min_chars=pipeline_config["target_min_chars"],
            target_max_chars=pipeline_config["target_max_chars"],
            use_llm_parsing=pipeline_config["use_llm_parsing"],
            retrieval_mode=pipeline_config["retrieval_mode"],
            style=pipeline_config["style"],
            temperature=pipeline_config["temperature"],
        )
        
        print(f"\n  ✓ 生成完成")
        print(f"  - 章节数: {len(result.chapters)}")
        print(f"  - 总字数: {len(result.merged_content)}")
        
    except Exception as e:
        print(f"\n  ✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 显示生成结果
    print("\n[5/6] 生成结果:")
    print("=" * 80)
    
    for i, chapter in enumerate(result.chapters, 1):
        print(f"\n【第 {i} 章】")
        print(f"原文片段: {chapter.segment_text[:80]}...")
        print(f"\n生成内容:")
        print(chapter.generation.content)
        print(f"\n字数: {len(chapter.generation.content)}")
        
        # 显示检索信息
        print(f"\n检索信息:")
        print(f"  - 实体数: {len(chapter.retrieval_result.entities)}")
        print(f"  - 关系数: {len(chapter.retrieval_result.relationships)}")

        # 显示所有实体名称
        if chapter.retrieval_result.entities:
            entity_names = [e.get('name', e.get('title', '')) for e in chapter.retrieval_result.entities]
            print(f"  - 实体列表: {', '.join(entity_names)}")

        # 显示新内容信息
        if hasattr(chapter.retrieval_result, '_novel_content_brief'):
            brief = chapter.retrieval_result._novel_content_brief
            print(f"\n新内容信息:")
            print(f"  - 对齐实体: {len(brief.aligned_entities)}")
            print(f"  - 可用新实体: {len(brief.novel_entities)}")
            if brief.novel_entities:
                names = [e.get('name', e.get('title', '')) for e in brief.novel_entities]
                print(f"  - 新实体列表: {', '.join(names)}")
        
        print("\n" + "-" * 80)
    
    # 6. 执行评估
    print("\n[6/6] 执行评估...")

    try:
        from src.evaluation.quality_gate import QualityThresholds

        eval_result = await evaluate_long_form(
            result,
            llm_adapter=llm_adapter,
            use_llm_eval=False,  # 不使用 LLM 评估（节省时间）
            enable_fact_check=False,  # Expansion 任务不需要严格的事实检查（依赖 expansion_grounding）
            quality_thresholds=QualityThresholds.for_expansion_task(),  # 使用 expansion 阈值
            enable_quality_gate=True,
        )

        print(f"\n  ✓ 评估完成")
        print(f"\n评估结果:")
        print(f"  - 综合分数: {eval_result.aggregated_score:.2f}/10")

        # 显示各章评分
        for seg in eval_result.segments:
            from src.evaluation import aggregate_scores
            score = aggregate_scores(seg.metrics)
            print(f"  - 第 {seg.segment_index + 1} 章: {score:.2f}/10")

            # 显示详细的metric计算过程
            print(f"    指标详情:")
            for metric_name, metric_value in seg.metrics.items():
                if hasattr(metric_value, 'value'):
                    print(f"      · {metric_name}: {metric_value.value:.3f}")
                    # 如果有计算细节，显示出来
                    if hasattr(metric_value, 'details') and metric_value.details:
                        for detail_key, detail_val in metric_value.details.items():
                            print(f"        - {detail_key}: {detail_val}")

            # 显示新内容指标
            if seg.novel_content_info:
                nci = seg.novel_content_info
                print(f"    新内容指标:")
                if 'information_gain' in nci:
                    print(f"      · 信息增益: {nci['information_gain']:.0%}")
                    # 显示计算细节
                    if 'novel_entities_used' in nci and 'novel_entities_available' in nci:
                        used_count = len(nci['novel_entities_used'])
                        available_count = len(nci['novel_entities_available'])
                        print(f"        计算: {used_count} / {available_count} = {nci['information_gain']:.3f}")
                if 'expansion_grounding' in nci:
                    print(f"      · 扩展溯源率: {nci['expansion_grounding']:.0%}")
                    # 显示计算细节
                    if 'grounded_facts' in nci and 'new_facts_in_output' in nci:
                        grounded_count = len(nci['grounded_facts'])
                        total_new_facts = len(nci['new_facts_in_output'])
                        if total_new_facts > 0:
                            print(f"        计算: {grounded_count} / {total_new_facts} = {nci['expansion_grounding']:.3f}")

        # 显示质量门控
        if eval_result.quality_gate:
            gate = eval_result.quality_gate
            print(f"\n质量门控: {'✓ 通过' if gate.passed else '✗ 未通过'}")

            # 显示每章的检查结果
            if gate.chapter_results:
                print(f"\n  章节检查:")
                for cr in gate.chapter_results:
                    status = "✓" if cr.passed else "✗"
                    print(f"    第{cr.chapter_index + 1}章 {status}")
                    if cr.issues:
                        for issue in cr.issues:
                            print(f"      [{issue.severity}] {issue.dimension}: {issue.message}")

            # 显示跨章问题
            if gate.cross_chapter_issues:
                print(f"\n  跨章问题:")
                for ci in gate.cross_chapter_issues:
                    print(f"    [{ci.severity}] {ci.dimension}: {ci.message}")

            if not gate.passed and gate.remediation:
                print(f"\n  建议重新生成: 第 {', '.join(str(c+1) for c in gate.remediation.chapters_to_regenerate)} 章")

    except Exception as e:
        print(f"\n  ✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        eval_result = None

    # 7. 生成详细报告
    print("\n[7/7] 生成详细报告...")

    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("Pipeline 详细测试报告")
    report_lines.append("=" * 100)

    # Pipeline 配置
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【Pipeline 配置】")
    report_lines.append("=" * 100)
    report_lines.append("\n配置参数:")
    for key, value in pipeline_config.items():
        report_lines.append(f"  - {key}: {value}")

    # 输入部分
    report_lines.append("\n" + "=" * 100)
    report_lines.append("【输入】原始回忆录")
    report_lines.append("=" * 100)
    report_lines.append(f"\n长度: {len(memoir_text)} 字\n")
    report_lines.append("内容:")
    report_lines.append("-" * 100)
    report_lines.append(memoir_text.strip())
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

        rr = chapter.retrieval_result

        # 获取评估结果（用于标注哪些实体被使用）
        seg_eval = eval_result.segments[i] if eval_result and i < len(eval_result.segments) else None
        novel_entities_used = []
        if seg_eval and hasattr(seg_eval, 'novel_content_info') and seg_eval.novel_content_info:
            nci = seg_eval.novel_content_info
            if hasattr(nci, 'novel_entities_used'):
                novel_entities_used = nci.novel_entities_used

        # 新内容分类摘要（先展示分类）
        if hasattr(rr, '_novel_content_brief') and rr._novel_content_brief:
            brief = rr._novel_content_brief
            report_lines.append(f"\n【检索内容分类】")
            report_lines.append(f"  ✓ 对齐实体 (原文已提到): {len(brief.aligned_entities)} 个")
            report_lines.append(f"  ★ 新增实体 (原文未提到): {len(brief.novel_entities)} 个")
            report_lines.append(f"  → 关系: {len(rr.relationships)} 个")

        # 对齐实体（原文已提到的）- 显示全部
        if hasattr(rr, '_novel_content_brief') and rr._novel_content_brief:
            brief = rr._novel_content_brief
            if brief.aligned_entities:
                report_lines.append(f"\n【对齐实体】(原文已提到，用于增强叙事氛围) - 共 {len(brief.aligned_entities)} 个")
                for j, ent in enumerate(brief.aligned_entities, 1):
                    name = ent.get('name', ent.get('title', ''))
                    ent_type = ent.get('type', '')
                    desc = ent.get('description', '')
                    report_lines.append(f"  {j}. {name} ({ent_type})")
                    if desc:
                        # 显示完整描述
                        report_lines.append(f"     {desc}")

        # 新增实体（原文未提到的，重点关注）- 显示全部
        if hasattr(rr, '_novel_content_brief') and rr._novel_content_brief:
            brief = rr._novel_content_brief
            if brief.novel_entities:
                report_lines.append(f"\n【新增实体】(原文未提到，可用于扩展) - 共 {len(brief.novel_entities)} 个")
                for j, ent in enumerate(brief.novel_entities, 1):
                    name = ent.get('name', ent.get('title', ''))
                    ent_type = ent.get('type', '')
                    desc = ent.get('description', '')

                    # 标注是否被使用
                    used_marker = " ✓ 已用于生成" if name in novel_entities_used else ""

                    report_lines.append(f"  {j}. {name} ({ent_type}){used_marker}")
                    if desc:
                        # 显示完整描述
                        report_lines.append(f"     {desc}")

        # 关系 - 显示全部
        if rr.relationships:
            report_lines.append(f"\n【关系】(共 {len(rr.relationships)} 个)")
            for j, rel in enumerate(rr.relationships, 1):
                source = rel.get('source', '')
                target = rel.get('target', '')
                desc = rel.get('description', '')
                report_lines.append(f"  {j}. {source} → {target}")
                if desc:
                    # 显示完整描述
                    report_lines.append(f"     {desc}")

        report_lines.append("-" * 100)

        # 生成内容
        report_lines.append("\n>>> 生成内容:")
        report_lines.append("-" * 100)
        report_lines.append(chapter.generation.content)
        report_lines.append("-" * 100)
        report_lines.append(f"\n生成长度: {len(chapter.generation.content)} 字")

        # 评估结果
        if eval_result and i < len(eval_result.segments):
            seg_eval = eval_result.segments[i]
            report_lines.append("\n>>> 评估结果:")
            report_lines.append("-" * 100)

            # 新内容使用情况（最重要的部分）
            if seg_eval.novel_content_info:
                nci = seg_eval.novel_content_info
                report_lines.append(f"\n【新内容使用情况】")

                # 1. 新实体使用率
                if 'novel_entities_used' in nci and 'novel_entities_available' in nci:
                    used = nci['novel_entities_used']
                    available = nci['novel_entities_available']
                    report_lines.append(f"  可用新实体: {len(available)} 个")
                    report_lines.append(f"  实际使用: {len(used)} 个")
                    if used:
                        report_lines.append(f"  已使用的新实体: {', '.join(used)}")
                    if set(available) - set(used):
                        unused = list(set(available) - set(used))
                        report_lines.append(f"  未使用的新实体: {', '.join(unused[:5])}")

                # 2. 新事实提取与溯源
                report_lines.append(f"\n【新事实提取与溯源】")

                if 'new_facts_in_output' in nci:
                    report_lines.append(f"  生成文本中的新事实: {len(nci['new_facts_in_output'])} 个")
                    if nci['new_facts_in_output']:
                        # 显示所有新事实
                        for idx, fact in enumerate(nci['new_facts_in_output'], 1):
                            report_lines.append(f"    {idx}. {fact}")

                if 'grounded_facts' in nci:
                    report_lines.append(f"\n  ✓ 有 RAG 支撑的新事实: {len(nci['grounded_facts'])} 个")
                    if nci['grounded_facts']:
                        # 显示所有有支撑的新事实
                        for idx, fact in enumerate(nci['grounded_facts'], 1):
                            report_lines.append(f"    {idx}. {fact}")

                if 'ungrounded_facts' in nci:
                    report_lines.append(f"\n  ✗ 无 RAG 支撑的新事实 (疑似幻觉): {len(nci['ungrounded_facts'])} 个")
                    if nci['ungrounded_facts']:
                        # 显示所有无支撑的新事实
                        for idx, fact in enumerate(nci['ungrounded_facts'], 1):
                            report_lines.append(f"    {idx}. ⚠️  {fact}")

                # 3. 评分
                report_lines.append(f"\n【评分】")
                if 'information_gain' in nci:
                    report_lines.append(f"  信息增益: {nci['information_gain']:.1%}")
                    # 显示计算过程
                    if 'novel_entities_used' in nci and 'novel_entities_available' in nci:
                        used_count = len(nci['novel_entities_used'])
                        available_count = len(nci['novel_entities_available'])
                        report_lines.append(f"    计算公式: 使用的新实体数 / 可用新实体数")
                        report_lines.append(f"    计算过程: {used_count} / {available_count} = {nci['information_gain']:.4f}")
                if 'expansion_grounding' in nci:
                    report_lines.append(f"  扩展溯源率: {nci['expansion_grounding']:.1%}")
                    # 显示计算过程
                    if 'grounded_facts' in nci and 'new_facts_in_output' in nci:
                        grounded_count = len(nci['grounded_facts'])
                        total_new_facts = len(nci['new_facts_in_output'])
                        report_lines.append(f"    计算公式: 有RAG支撑的新事实数 / 总新事实数")
                        if total_new_facts > 0:
                            report_lines.append(f"    计算过程: {grounded_count} / {total_new_facts} = {nci['expansion_grounding']:.4f}")
                        else:
                            report_lines.append(f"    计算过程: {grounded_count} / {total_new_facts} = N/A (无新事实)")

            # 其他指标
            report_lines.append(f"\n【其他指标】")
            for metric_name, metric_value in seg_eval.metrics.items():
                if hasattr(metric_value, 'value'):
                    report_lines.append(f"  - {metric_name}: {metric_value.value:.4f}")
                    # 如果有计算细节，显示出来
                    if hasattr(metric_value, 'details') and metric_value.details:
                        report_lines.append(f"    计算细节:")
                        for detail_key, detail_val in metric_value.details.items():
                            if isinstance(detail_val, float):
                                report_lines.append(f"      · {detail_key}: {detail_val:.4f}")
                            else:
                                report_lines.append(f"      · {detail_key}: {detail_val}")
                else:
                    report_lines.append(f"  - {metric_name}: {metric_value}")

            report_lines.append("-" * 100)

    # 整体评估
    if eval_result:
        report_lines.append("\n" + "=" * 100)
        report_lines.append("【整体评估】")
        report_lines.append("=" * 100)

        report_lines.append(f"\n综合分数: {eval_result.aggregated_score:.2f}/10")

        if eval_result.document_metrics:
            report_lines.append(f"\n文档级指标:")
            for metric_name, metric_value in eval_result.document_metrics.items():
                if hasattr(metric_value, 'value'):
                    report_lines.append(f"  - {metric_name}: {metric_value.value:.4f}")
                    # 如果有计算细节，显示出来
                    if hasattr(metric_value, 'details') and metric_value.details:
                        report_lines.append(f"    计算细节:")
                        for detail_key, detail_val in metric_value.details.items():
                            if isinstance(detail_val, float):
                                report_lines.append(f"      · {detail_key}: {detail_val:.4f}")
                            else:
                                report_lines.append(f"      · {detail_key}: {detail_val}")
                else:
                    report_lines.append(f"  - {metric_name}: {metric_value}")

        if eval_result.quality_gate:
            gate = eval_result.quality_gate
            report_lines.append(f"\n质量门控: {'✓ 通过' if gate.passed else '✗ 未通过'}")

            # 添加详细的失败信息
            if not gate.passed:
                # 章节检查结果
                if gate.chapter_results:
                    report_lines.append(f"\n【章节检查】")
                    for cr in gate.chapter_results:
                        status = "✓ 通过" if cr.passed else "✗ 未通过"
                        report_lines.append(f"  第{cr.chapter_index + 1}章: {status}")
                        if cr.issues:
                            for issue in cr.issues:
                                report_lines.append(f"    [{issue.severity}] {issue.dimension}: {issue.message}")
                                report_lines.append(f"      建议: {issue.suggestion}")

                # 跨章问题
                if gate.cross_chapter_issues:
                    report_lines.append(f"\n【跨章问题】")
                    for ci in gate.cross_chapter_issues:
                        report_lines.append(f"  [{ci.severity}] {ci.dimension}: {ci.message}")
                        report_lines.append(f"    涉及章节: 第{', '.join(str(c+1) for c in ci.chapters_involved)}章")
                        report_lines.append(f"    建议: {ci.suggestion}")

                # 修复建议
                if gate.remediation:
                    report_lines.append(f"\n【修复建议】")
                    report_lines.append(f"  需重新生成: 第 {', '.join(str(c+1) for c in gate.remediation.chapters_to_regenerate)} 章")
                    for ch_idx, reasons in gate.remediation.reasons.items():
                        report_lines.append(f"  第{ch_idx+1}章原因:")
                        for reason in reasons:
                            report_lines.append(f"    - {reason}")

    report_lines.append("\n" + "=" * 100)
    report_lines.append("报告结束")
    report_lines.append("=" * 100)

    # 保存报告
    report_content = "\n".join(report_lines)
    output_path = Path(__file__).parent / "PIPELINE_DETAILED_REPORT.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"\n✓ 详细报告已保存到: {output_path}")
    print(f"  文件大小: {len(report_content)} 字符")

    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
