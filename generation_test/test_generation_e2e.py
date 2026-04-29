"""
生成模块端到端测试

测试目标：
1. 验证生成模块的完整功能（分段、检索、生成、上下文管理）
2. 使用真实的RAG检索（需要已构建的知识图谱）
3. 只测试生成模块，不测试其他模块

前置条件：
- 需要已构建的知识图谱（在 output/ 目录下）
- 需要配置好的LLM（通过环境变量或配置文件）
"""

import asyncio
import os
from pathlib import Path

import pytest

from src.generation import (
    LiteraryGenerator,
    segment_memoir,
    allocate_segment_budgets,
    run_long_form_generation,
)
from src.retrieval import MemoirRetriever
from src.llm import create_llm_adapter


# 测试用的简短回忆录文本
TEST_MEMOIR = """
1978年，我考上了大学。那是恢复高考后的第二年，整个国家都充满了希望。
我记得收到录取通知书的那天，父亲激动得一夜没睡。

1980年，我在大学里遇到了我的妻子。她是图书馆的管理员，总是安静地整理着书籍。
我们在一次读书会上相识，从此开始了我们的故事。

1985年，我大学毕业后被分配到了一家国营工厂。虽然工资不高，但那时候大家都很满足。
工厂里的同事们关系很好，经常一起打篮球、下棋。
"""


@pytest.fixture
def output_dir():
    """获取输出目录"""
    output_path = Path("output")
    if not output_path.exists():
        pytest.skip("需要已构建的知识图谱（output/ 目录不存在）")
    return output_path


@pytest.fixture
def llm_adapter():
    """创建LLM适配器"""
    try:
        adapter = create_llm_adapter()
        return adapter
    except Exception as e:
        pytest.skip(f"无法创建LLM适配器: {e}")


@pytest.fixture
async def retriever(output_dir):
    """创建检索器"""
    try:
        retriever = MemoirRetriever(index_dir=str(output_dir))
        await retriever.initialize()
        return retriever
    except Exception as e:
        pytest.skip(f"无法初始化检索器: {e}")


class TestGenerationModule:
    """生成模块测试套件"""

    def test_memoir_segmentation(self):
        """测试1: 回忆录分段功能"""
        print("\n=== 测试1: 回忆录分段 ===")

        segments, report = segment_memoir(TEST_MEMOIR, target_chars_per_segment=200)

        print(f"分段数量: {len(segments)}")
        print(f"验证报告: {report}")

        # 验证
        assert len(segments) > 0, "应该至少有1个段落"
        assert all(seg.text.strip() for seg in segments), "所有段落都应该有内容"
        assert report.is_valid, f"分段应该有效: {report.issues}"

        # 打印每个段落的元数据
        for i, seg in enumerate(segments):
            print(f"\n段落 {i+1}:")
            print(f"  长度: {len(seg.text)} 字符")
            print(f"  年份: {seg.meta.years if seg.meta else '无'}")
            print(f"  地点: {seg.meta.locations if seg.meta else '无'}")
            print(f"  内容预览: {seg.text[:50]}...")

        print("\n✓ 分段测试通过")

    def test_budget_allocation(self):
        """测试2: 预算分配功能"""
        print("\n=== 测试2: 预算分配 ===")

        segments, _ = segment_memoir(TEST_MEMOIR, target_chars_per_segment=200)
        budgets = allocate_segment_budgets(segments, length_bucket="400-800")

        print(f"预算数量: {len(budgets)}")

        # 验证
        assert len(budgets) == len(segments), "预算数量应该等于段落数量"

        for i, budget in enumerate(budgets):
            print(f"\n段落 {i+1} 预算:")
            print(f"  长度提示: {budget.length_hint}")
            print(f"  最大tokens: {budget.max_tokens}")
            print(f"  目标字符数: {budget.target_chars}")

            assert budget.max_tokens > 0, "max_tokens应该大于0"
            assert budget.target_chars > 0, "target_chars应该大于0"

        print("\n✓ 预算分配测试通过")

    @pytest.mark.asyncio
    async def test_single_segment_retrieval(self, retriever):
        """测试3: 单段检索功能"""
        print("\n=== 测试3: 单段检索 ===")

        segments, _ = segment_memoir(TEST_MEMOIR, target_chars_per_segment=200)
        first_segment = segments[0]

        print(f"检索文本: {first_segment.text[:100]}...")

        # 执行检索
        result = await retriever.retrieve(
            first_segment.text,
            top_k=5,
            mode="hybrid"
        )

        print(f"\n检索结果:")
        print(f"  查询: {result.query}")
        print(f"  实体数量: {len(result.entities)}")
        print(f"  社区数量: {len(result.communities)}")
        print(f"  关系数量: {len(result.relationships)}")

        # 验证
        assert result.query, "应该有查询文本"
        assert result.has_results, "应该有检索结果"

        # 打印前3个实体
        if result.entities:
            print(f"\n前3个实体:")
            for entity in result.entities[:3]:
                name = entity.get("name", "未知")
                desc = entity.get("description", "")[:50]
                print(f"  - {name}: {desc}...")

        print("\n✓ 检索测试通过")

    @pytest.mark.asyncio
    async def test_single_segment_generation(self, retriever, llm_adapter):
        """测试4: 单段生成功能"""
        print("\n=== 测试4: 单段生成 ===")

        segments, _ = segment_memoir(TEST_MEMOIR, target_chars_per_segment=200)
        first_segment = segments[0]

        # 检索
        retrieval_result = await retriever.retrieve(
            first_segment.text,
            top_k=5,
            mode="hybrid"
        )

        # 生成
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        generation_result = await generator.generate(
            memoir_text=first_segment.text,
            retrieval_result=retrieval_result,
            style="standard",
            length_hint="200-400字",
            temperature=0.7,
            max_tokens=1024,
        )

        print(f"\n生成结果:")
        print(f"  提供商: {generation_result.provider}")
        print(f"  模型: {generation_result.model}")
        print(f"  内容长度: {len(generation_result.content)} 字符")
        print(f"  检索信息: {generation_result.retrieval_info}")
        print(f"\n生成内容:")
        print(generation_result.content)

        # 验证
        assert generation_result.content, "应该有生成内容"
        assert len(generation_result.content) > 50, "生成内容应该足够长"
        assert generation_result.provider, "应该有提供商信息"
        assert generation_result.model, "应该有模型信息"

        print("\n✓ 单段生成测试通过")

    @pytest.mark.asyncio
    async def test_long_form_generation(self, retriever, llm_adapter):
        """测试5: 长文生成功能（完整流程）"""
        print("\n=== 测试5: 长文生成（完整流程）===")

        generator = LiteraryGenerator(llm_adapter=llm_adapter)

        # 执行长文生成
        result = await run_long_form_generation(
            memoir_text=TEST_MEMOIR,
            retriever=retriever,
            generator=generator,
            length_bucket="400-800",
            style="standard",
            temperature=0.7,
            top_k=5,
            retrieval_mode="hybrid",
            enable_cross_chapter_context=True,
        )

        print(f"\n生成结果:")
        print(f"  章节数量: {len(result.chapters)}")
        print(f"  合并内容长度: {len(result.merged_content)} 字符")
        print(f"  分段报告: {result.segmentation_report}")

        # 验证
        assert len(result.chapters) > 0, "应该至少有1个章节"
        assert result.merged_content, "应该有合并内容"
        assert result.segmentation_report.is_valid, "分段应该有效"

        # 打印每个章节的信息
        for i, chapter in enumerate(result.chapters):
            print(f"\n章节 {i+1}:")
            print(f"  段落索引: {chapter.segment_index}")
            print(f"  内容长度: {len(chapter.generation.content)} 字符")
            print(f"  提供商: {chapter.generation.provider}")
            print(f"  警告: {chapter.warnings or '无'}")
            print(f"  内容预览: {chapter.generation.content[:100]}...")

        # 验证跨章节上下文
        if result.chapter_context and len(result.chapters) > 1:
            print(f"\n跨章节上下文:")
            print(f"  记录数量: {len(result.chapter_context._records)}")
            print(f"  关键短语总数: {len(result.chapter_context._all_key_phrases)}")

        print(f"\n完整生成内容:")
        print("=" * 60)
        print(result.merged_content)
        print("=" * 60)

        print("\n✓ 长文生成测试通过")

    @pytest.mark.asyncio
    async def test_cross_chapter_context(self, retriever, llm_adapter):
        """测试6: 跨章节上下文管理"""
        print("\n=== 测试6: 跨章节上下文管理 ===")

        generator = LiteraryGenerator(llm_adapter=llm_adapter)

        # 执行长文生成（启用跨章节上下文）
        result = await run_long_form_generation(
            memoir_text=TEST_MEMOIR,
            retriever=retriever,
            generator=generator,
            length_bucket="400-800",
            style="standard",
            temperature=0.7,
            top_k=5,
            retrieval_mode="hybrid",
            enable_cross_chapter_context=True,
        )

        # 验证跨章节上下文
        if len(result.chapters) > 1:
            assert result.chapter_context is not None, "应该有章节上下文"

            print(f"\n章节上下文信息:")
            print(f"  总章节数: {len(result.chapter_context._records)}")

            for i, record in enumerate(result.chapter_context._records):
                print(f"\n章节 {i+1} 记录:")
                print(f"  摘要: {record.brief}")
                print(f"  时间段: {record.time_period}")
                print(f"  实体: {record.entities[:5]}")  # 只显示前5个
                print(f"  关键短语数: {len(record.key_phrases)}")

            print("\n✓ 跨章节上下文测试通过")
        else:
            print("\n⚠ 只有1个章节，跳过跨章节上下文测试")


def main():
    """手动运行测试（不使用pytest）"""
    print("=" * 60)
    print("生成模块端到端测试")
    print("=" * 60)

    # 检查前置条件
    output_dir = Path("output")
    if not output_dir.exists():
        print("\n❌ 错误: 需要已构建的知识图谱（output/ 目录不存在）")
        print("请先运行知识图谱构建流程")
        return

    try:
        llm_adapter = create_llm_adapter()
    except Exception as e:
        print(f"\n❌ 错误: 无法创建LLM适配器: {e}")
        print("请检查LLM配置（环境变量或配置文件）")
        return

    # 创建测试实例
    test = TestGenerationModule()

    # 运行测试
    try:
        # 测试1: 分段
        test.test_memoir_segmentation()

        # 测试2: 预算分配
        test.test_budget_allocation()

        # 测试3-6: 需要异步运行
        async def run_async_tests():
            retriever = MemoirRetriever(index_dir=str(output_dir))
            await retriever.initialize()

            await test.test_single_segment_retrieval(retriever)
            await test.test_single_segment_generation(retriever, llm_adapter)
            await test.test_long_form_generation(retriever, llm_adapter)
            await test.test_cross_chapter_context(retriever, llm_adapter)

        asyncio.run(run_async_tests())

        print("\n" + "=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
