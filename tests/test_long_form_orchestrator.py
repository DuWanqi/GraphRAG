"""分章编排：mock 检索与生成，验证按段调用次数、合并结果、跨章上下文。"""

import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.memoir_retriever import RetrievalResult
from src.retrieval.memoir_parser import MemoirContext
from src.generation.literary_generator import GenerationResult
from src.generation.long_form_orchestrator import run_long_form_generation


class _MockRetriever:
    def __init__(self):
        self.retrieve_calls: list[str] = []

    async def retrieve(self, memoir_text: str, **kwargs):
        self.retrieve_calls.append(memoir_text[:80])
        ctx = MemoirContext(original_text=memoir_text)
        return RetrievalResult(query=f"q:{len(memoir_text)}", context=ctx, entities=[])


class _MockGenerator:
    def __init__(self):
        self.generate_calls = 0
        self.received_chapter_contexts: list[str] = []

    async def generate(self, memoir_text: str, retrieval_result: RetrievalResult, **kwargs):
        self.generate_calls += 1
        # 记录收到的 chapter_context
        self.received_chapter_contexts.append(kwargs.get("chapter_context", ""))
        return GenerationResult(
            content=f"第{self.generate_calls}章背景（段长{len(memoir_text)}）。",
            provider="mock",
            model="mock",
        )


def test_orchestrator_multi_segment_long_memoir():
    """多段输入时 retrieve/generate 与段数一致，合并含分隔符。"""
    body = "一九八八年夏天，我来到深圳。" * 120 + "\n\n" + "一九九二年春天，我又去了广州。" * 120
    assert len(body) >= 3000

    retriever = _MockRetriever()
    generator = _MockGenerator()

    lf = asyncio.run(
        run_long_form_generation(
            body,
            retriever,
            generator,
            length_bucket="400-800",
            retrieval_mode="keyword",
        )
    )

    assert len(lf.chapters) >= 2
    assert len(retriever.retrieve_calls) == len(lf.chapters)
    assert generator.generate_calls == len(lf.chapters)
    if len(lf.chapters) > 1:
        assert "---" in lf.merged_content
    assert all(ch.length_hint for ch in lf.chapters)


def test_orchestrator_has_segmentation_report():
    """编排结果应包含分段校验报告。"""
    body = "一九八八年夏天，我来到深圳。" * 120 + "\n\n" + "一九九二年春天，我又去了广州。" * 120
    retriever = _MockRetriever()
    generator = _MockGenerator()

    lf = asyncio.run(
        run_long_form_generation(body, retriever, generator)
    )
    assert lf.segmentation_report is not None
    assert lf.segmentation_report.segment_count == len(lf.chapters)


def test_orchestrator_cross_chapter_context():
    """启用跨章上下文时，第二章及以后应收到非空 chapter_context。"""
    body = "一九七二年秋天，我来到陕北。" * 100 + "\n\n" + "一九七七年十月，恢复高考消息传来。" * 100
    retriever = _MockRetriever()
    generator = _MockGenerator()

    lf = asyncio.run(
        run_long_form_generation(
            body,
            retriever,
            generator,
            enable_cross_chapter_context=True,
        )
    )
    assert len(lf.chapters) >= 2
    # 第一章 chapter_context 应只有位置指令（无前文概要）
    first_ctx = generator.received_chapter_contexts[0]
    assert "前文" not in first_ctx        # 第一章没有前文概要
    assert "开篇" in first_ctx            # 但有开篇位置指令
    # 第二章及以后应收到非空上下文，且包含前文概要
    for ctx in generator.received_chapter_contexts[1:]:
        assert ctx != ""
        assert "前文" in ctx or "概要" in ctx


def test_orchestrator_disabled_cross_chapter():
    """关闭跨章上下文时，所有章节 chapter_context 应为空。"""
    body = "一九八八年。" * 200 + "\n\n" + "一九九二年。" * 200
    retriever = _MockRetriever()
    generator = _MockGenerator()

    lf = asyncio.run(
        run_long_form_generation(
            body,
            retriever,
            generator,
            enable_cross_chapter_context=False,
        )
    )
    for ctx in generator.received_chapter_contexts:
        assert ctx == ""
