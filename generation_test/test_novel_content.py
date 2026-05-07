"""
Novel Content Generation Test - 新内容生成测试

验证生成模块能够正确引入 RAG 检索到的、输入中未提及的新知识。

测试策略：
1. 使用极简输入（只提到年份和事件，不包含任何历史背景）
2. 验证 RAG 检索到了新知识
3. 验证生成文本中包含了新知识
4. 验证新内容有 RAG 来源支撑（防幻觉）
"""

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from src.generation import (
    LiteraryGenerator,
    segment_memoir,
    extract_novel_content,
)
from src.retrieval import MemoirRetriever
from src.llm import create_llm_adapter
from src.evaluation import (
    information_gain_metric,
    expansion_grounding_metric,
)


# 极简测试输入：只提到年份和事件，不包含任何历史背景
TEST_MEMOIR_MINIMAL = """
1980年，我考上了大学。
"""


@pytest.fixture
def output_dir():
    """获取输出目录"""
    output_path = Path("data/graphrag_output")
    if not output_path.exists():
        pytest.skip("需要已构建的知识图谱（data/graphrag_output/ 目录不存在）")
    parquet_dir = output_path / "output"
    if not parquet_dir.exists() or not list(parquet_dir.glob("*.parquet")):
        pytest.skip("知识图谱索引文件不完整（缺少 parquet 文件）")
    return output_path


@pytest.fixture
def llm_adapter():
    """创建LLM适配器（带健康检查）"""
    try:
        adapter = create_llm_adapter()
        # 简单健康检查：尝试生成一个短文本
        asyncio.run(adapter.generate(
            prompt="测试",
            system_prompt="你是一个测试助手",
            temperature=0.1,
            max_tokens=10,
        ))
        return adapter
    except Exception as e:
        pytest.skip(f"LLM 服务不可用: {e}")


@pytest_asyncio.fixture
async def retriever(output_dir):
    """创建检索器"""
    try:
        retriever = MemoirRetriever(index_dir=str(output_dir))
        return retriever
    except Exception as e:
        pytest.skip(f"无法初始化检索器: {e}")


class TestNovelContentGeneration:
    """新内容生成测试套件"""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_novel_content_extraction(self, retriever):
        """测试1: 新内容提取功能"""
        print("\n=== 测试1: 新内容提取 ===")

        # 检索
        print(f"输入文本: {TEST_MEMOIR_MINIMAL.strip()}")
        retrieval_result = await retriever.retrieve(
            TEST_MEMOIR_MINIMAL,
            top_k=10,
            mode="hybrid"
        )

        print(f"\n检索结果:")
        print(f"  实体数量: {len(retrieval_result.entities)}")
        print(f"  关系数量: {len(retrieval_result.relationships)}")
        print(f"  社区数量: {len(retrieval_result.communities)}")

        # 提取新内容
        novel_brief = extract_novel_content(TEST_MEMOIR_MINIMAL, retrieval_result)

        print(f"\n新内容分类:")
        print(f"  新实体数量: {len(novel_brief.novel_entities)}")
        print(f"  对齐实体数量: {len(novel_brief.aligned_entities)}")
        print(f"  新关系数量: {len(novel_brief.novel_relationships)}")
        print(f"  新背景片段数量: {len(novel_brief.novel_snippets)}")
        print(f"  摘要: {novel_brief.summary}")

        # 验证
        assert novel_brief.has_novel_content, "应该有新知识可用"
        assert len(novel_brief.novel_entities) > 0, "应该有新实体"

        # 打印前5个新实体
        print(f"\n前5个新实体:")
        for entity in novel_brief.novel_entities[:5]:
            name = entity.get("name", entity.get("title", ""))
            desc = entity.get("description", "")[:80]
            print(f"  - {name}: {desc}")

        print("\n✓ 新内容提取测试通过")

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_novel_content_generation(self, retriever, llm_adapter):
        """测试2: 新内容生成功能"""
        print("\n=== 测试2: 新内容生成 ===")

        # 检索
        print(f"输入文本: {TEST_MEMOIR_MINIMAL.strip()}")
        retrieval_result = await retriever.retrieve(
            TEST_MEMOIR_MINIMAL,
            top_k=10,
            mode="hybrid"
        )

        # 提取新内容
        novel_brief = extract_novel_content(TEST_MEMOIR_MINIMAL, retrieval_result)
        print(f"\n可用新知识: {novel_brief.summary}")
        print(f"新实体: {', '.join(novel_brief.novel_entity_names[:5])}")

        # 生成
        print(f"\n开始生成...")
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        generation_result = await generator.generate(
            memoir_text=TEST_MEMOIR_MINIMAL,
            retrieval_result=retrieval_result,
            style="standard",
            length_hint="300-500字",
            temperature=0.7,
            max_tokens=1024,
        )

        print(f"\n生成结果:")
        print(f"  提供商: {generation_result.provider}")
        print(f"  模型: {generation_result.model}")
        print(f"  内容长度: {len(generation_result.content)} 字符")
        print(f"\n生成内容:")
        print("=" * 60)
        print(generation_result.content)
        print("=" * 60)

        # 验证：生成内容应该包含至少一个新实体
        generated_text = generation_result.content
        novel_entities_used = []
        for entity_name in novel_brief.novel_entity_names[:10]:
            if entity_name and entity_name in generated_text:
                novel_entities_used.append(entity_name)

        print(f"\n新实体使用情况:")
        print(f"  可用新实体: {len(novel_brief.novel_entity_names)}")
        print(f"  已使用新实体: {len(novel_entities_used)}")
        if novel_entities_used:
            print(f"  使用的新实体: {', '.join(novel_entities_used)}")

        assert len(novel_entities_used) > 0, "生成文本应该包含至少一个新实体"

        print("\n✓ 新内容生成测试通过")

    @pytest.mark.asyncio
    @pytest.mark.timeout(180)
    async def test_novel_content_metrics(self, retriever, llm_adapter):
        """测试3: 新内容评估指标"""
        print("\n=== 测试3: 新内容评估指标 ===")

        # 检索
        retrieval_result = await retriever.retrieve(
            TEST_MEMOIR_MINIMAL,
            top_k=10,
            mode="hybrid"
        )

        # 提取新内容
        novel_brief = extract_novel_content(TEST_MEMOIR_MINIMAL, retrieval_result)

        # 生成
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        generation_result = await generator.generate(
            memoir_text=TEST_MEMOIR_MINIMAL,
            retrieval_result=retrieval_result,
            style="standard",
            length_hint="300-500字",
            temperature=0.7,
            max_tokens=1024,
        )

        generated_text = generation_result.content

        # 计算新内容指标
        print(f"\n计算新内容指标...")

        ratio_metric = information_gain_metric(
            TEST_MEMOIR_MINIMAL,
            generated_text,
            novel_brief,
        )
        print(f"\n信息增量:")
        print(f"  分数: {ratio_metric.value:.2f} / {ratio_metric.max_value}")
        print(f"  说明: {ratio_metric.explanation}")

        grounding_metric = expansion_grounding_metric(
            TEST_MEMOIR_MINIMAL,
            generated_text,
            novel_brief,
        )
        print(f"\n扩展溯源率:")
        print(f"  分数: {grounding_metric.value:.2f} / {grounding_metric.max_value}")
        print(f"  说明: {grounding_metric.explanation}")

        # 验证
        assert ratio_metric.value > 0, "信息增量应该大于0（至少引入了一些新知识）"
        assert grounding_metric.value > 0.5, "扩展溯源率应该大于0.5（新内容有RAG来源支撑）"

        print("\n✓ 新内容评估指标测试通过")


def main():
    """手动运行测试（不使用pytest）"""
    print("=" * 60)
    print("新内容生成测试")
    print("=" * 60)

    # 检查前置条件
    output_dir = Path("data/graphrag_output")
    if not output_dir.exists():
        print("\n❌ 错误: 需要已构建的知识图谱（data/graphrag_output/ 目录不存在）")
        return

    try:
        llm_adapter = create_llm_adapter()
        # 健康检查
        asyncio.run(llm_adapter.generate(
            prompt="测试",
            system_prompt="你是一个测试助手",
            temperature=0.1,
            max_tokens=10,
        ))
    except Exception as e:
        print(f"\n❌ 错误: LLM 服务不可用: {e}")
        return

    # 创建测试实例
    test = TestNovelContentGeneration()

    # 运行测试
    try:
        async def run_async_tests():
            retriever = MemoirRetriever(index_dir=str(output_dir))

            await test.test_novel_content_extraction(retriever)
            await test.test_novel_content_generation(retriever, llm_adapter)
            await test.test_novel_content_metrics(retriever, llm_adapter)

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
