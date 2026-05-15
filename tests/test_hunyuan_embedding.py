import pytest

from src.llm.hunyuan_embedding import HunyuanEmbedding, HunyuanEmbeddingConfig


def test_hunyuan_embedding_requires_api_key():
    with pytest.raises(ValueError, match="HUNYUAN_API_KEY"):
        HunyuanEmbedding(HunyuanEmbeddingConfig(api_key=None))


def test_hunyuan_embedding_extracts_openai_compatible_response():
    client = HunyuanEmbedding(HunyuanEmbeddingConfig(api_key="test-key"))

    vector = client._extract_embedding(
        {
            "data": [
                {
                    "index": 0,
                    "embedding": [0, "1.5", -2],
                }
            ]
        }
    )

    assert vector == [0.0, 1.5, -2.0]
