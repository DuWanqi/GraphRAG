"""分章编排：mock 检索与生成，验证按段调用次数与合并结果。"""

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

    async def generate(self, memoir_text: str, retrieval_result: RetrievalResult, **kwargs):
        self.generate_calls += 1
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
