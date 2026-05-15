"""Tencent Hunyuan embedding client.

Uses the OpenAI-compatible Hunyuan `/v1/embeddings` endpoint.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp


@dataclass
class HunyuanEmbeddingConfig:
    api_key: Optional[str] = None
    base_url: str = "https://api.hunyuan.cloud.tencent.com/v1"
    model: str = "hunyuan-embedding"
    timeout: int = 60


class HunyuanEmbedding:
    """Embedding client for Tencent Hunyuan OpenAI-compatible API."""

    def __init__(self, config: HunyuanEmbeddingConfig):
        if not config.api_key:
            raise ValueError(
                "缺少 HUNYUAN_API_KEY，无法调用腾讯混元 embedding API。"
            )
        self.config = config
        self.config.base_url = self.config.base_url.rstrip("/")

    async def embed(self, text: str) -> List[float]:
        url = f"{self.config.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "input": text,
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"腾讯混元 embedding API 错误: {response.status} - {body}"
                    )
                data = await response.json()

        return self._extract_embedding(data)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Hunyuan OpenAI-compatible embedding currently documents a single
        # string input, so keep batching conservative and deterministic.
        return [await self.embed(text) for text in texts]

    def embed_sync(self, text: str) -> List[float]:
        return asyncio.run(self.embed(text))

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        return asyncio.run(self.embed_batch(texts))

    def _extract_embedding(self, data: Any) -> List[float]:
        items = data.get("data") if isinstance(data, dict) else None
        if not items:
            raise RuntimeError(f"腾讯混元 embedding 响应缺少 data: {data}")
        embedding = items[0].get("embedding") if isinstance(items[0], dict) else None
        if not isinstance(embedding, list):
            raise RuntimeError(f"腾讯混元 embedding 响应缺少 embedding: {data}")
        return [float(value) for value in embedding]


def get_hunyuan_embedding(
    api_key: str,
    base_url: str = "https://api.hunyuan.cloud.tencent.com/v1",
    model: str = "hunyuan-embedding",
) -> HunyuanEmbedding:
    return HunyuanEmbedding(
        HunyuanEmbeddingConfig(api_key=api_key, base_url=base_url, model=model)
    )
