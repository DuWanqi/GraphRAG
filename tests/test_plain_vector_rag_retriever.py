import asyncio
import json
from pathlib import Path

import pytest

from src.retrieval import (
    PLAIN_VECTOR_RAG_MODE,
    PlainVectorRAGEmbeddingError,
    PlainVectorRAGRetriever,
    RetrievalResult,
)


class FakeEmbedding:
    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    async def embed(self, text: str):
        self.single_calls += 1
        return self._vector(text)

    async def embed_batch(self, texts):
        self.batch_calls += 1
        return [self._vector(text) for text in texts]

    def _vector(self, text: str):
        text = text.lower()
        return [
            1.0 if ("深圳" in text or "shenzhen" in text) else 0.0,
            1.0 if "蛇口" in text else 0.0,
            1.0 if ("改革" in text or "开放" in text) else 0.0,
            1.0 if ("北京" in text or "beijing" in text) else 0.0,
        ]


class FailingEmbedding:
    async def embed_batch(self, texts):
        raise OSError("Cannot connect to host localhost:11434")


def _run(coro):
    return asyncio.run(coro)


def _make_retriever(input_dir: Path, cache_dir: Path, embedding: FakeEmbedding):
    return PlainVectorRAGRetriever(
        input_dir=input_dir,
        cache_dir=cache_dir,
        embedding_backend="fake",
        embedding_model="fake-model",
        embedding_client=embedding,
        chunk_size=120,
        chunk_overlap=20,
        batch_size=2,
    )


def test_plain_vector_rag_loads_txt_and_returns_text_units(tmp_path):
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "history.txt").write_text(
        "深圳蛇口在改革开放初期推进工业区建设，吸引企业和人才。\n\n"
        "北京举办了一次无关会议。",
        encoding="utf-8",
    )

    retriever = _make_retriever(input_dir, cache_dir, FakeEmbedding())
    result = _run(retriever.retrieve("我在蛇口工作，见证改革开放。", mode=PLAIN_VECTOR_RAG_MODE))

    assert isinstance(result, RetrievalResult)
    assert result.text_units
    assert "蛇口" in result.text_units[0]
    assert "普通RAG来源" in result.text_units[0]
    assert result.get_context_text()


def test_plain_vector_rag_extracts_json_event_fields(tmp_path):
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    data = [
        {
            "title": "蛇口工业区建设",
            "date": "1979年",
            "location": "深圳",
            "content": "蛇口工业区成为改革开放的重要窗口。",
            "keywords": ["蛇口", "改革开放"],
            "source": "测试来源",
        },
        {
            "title": "北京会议",
            "date": "1980年",
            "location": "北京",
            "content": "一次与深圳无关的会议。",
        },
    ]
    (input_dir / "events.json").write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )

    retriever = _make_retriever(input_dir, cache_dir, FakeEmbedding())
    result = _run(retriever.retrieve("深圳蛇口的改革开放经历", mode=PLAIN_VECTOR_RAG_MODE))

    assert result.text_units
    assert "标题: 蛇口工业区建设" in result.text_units[0]
    assert "日期: 1979年" in result.text_units[0]
    assert "地点: 深圳" in result.text_units[0]


def test_plain_vector_rag_cache_hit_skips_chunk_embedding(tmp_path):
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "history.txt").write_text(
        "深圳蛇口在改革开放初期承担了重要试验任务，工业区建设推动了制度创新。",
        encoding="utf-8",
    )

    first_embedding = FakeEmbedding()
    first = _make_retriever(input_dir, cache_dir, first_embedding)
    _run(first.retrieve("深圳蛇口", mode=PLAIN_VECTOR_RAG_MODE))
    assert first_embedding.batch_calls > 0

    second_embedding = FakeEmbedding()
    second = _make_retriever(input_dir, cache_dir, second_embedding)
    _run(second.retrieve("深圳蛇口", mode=PLAIN_VECTOR_RAG_MODE))

    assert second_embedding.batch_calls == 0
    assert second_embedding.single_calls == 1


def test_plain_vector_rag_rebuilds_when_input_changes(tmp_path):
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "history.txt").write_text(
        "深圳蛇口在改革开放初期承担了重要试验任务，工业区建设推动了制度创新。",
        encoding="utf-8",
    )

    first = _make_retriever(input_dir, cache_dir, FakeEmbedding())
    _run(first.retrieve("深圳蛇口", mode=PLAIN_VECTOR_RAG_MODE))

    (input_dir / "extra.txt").write_text(
        "北京召开了一次重要会议，讨论城市治理和经济建设相关议题。",
        encoding="utf-8",
    )
    changed_embedding = FakeEmbedding()
    changed = _make_retriever(input_dir, cache_dir, changed_embedding)
    _run(changed.retrieve("北京", mode=PLAIN_VECTOR_RAG_MODE))

    assert changed_embedding.batch_calls > 0


def test_plain_vector_rag_reports_ollama_connection_help(tmp_path):
    input_dir = tmp_path / "input"
    cache_dir = tmp_path / "cache"
    input_dir.mkdir()
    (input_dir / "history.txt").write_text(
        "A plain vector RAG source document with enough text to become one chunk.",
        encoding="utf-8",
    )

    retriever = PlainVectorRAGRetriever(
        input_dir=input_dir,
        cache_dir=cache_dir,
        embedding_backend="ollama",
        embedding_model="nomic-embed-text",
        embedding_client=FailingEmbedding(),
    )

    with pytest.raises(PlainVectorRAGEmbeddingError) as exc_info:
        _run(retriever.retrieve("query", mode=PLAIN_VECTOR_RAG_MODE))

    message = str(exc_info.value)
    assert "ollama serve" in message
    assert "ollama pull nomic-embed-text" in message
