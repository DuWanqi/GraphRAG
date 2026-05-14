#!/usr/bin/env python3
"""Build the plain vector RAG cache.

This script is intentionally separate from scripts/rebuild_index.py. It does
not run Microsoft GraphRAG and only creates data/plain_vector_rag_cache.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
import sys
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_settings
from src.retrieval import PLAIN_VECTOR_RAG_MODE, PlainVectorRAGRetriever


DEFAULT_SMOKE_QUERY = "土地改革 解放初期 新中国成立"


def _optional_path(value: Optional[str]) -> Optional[str]:
    return value if value and value.strip() else None


def _print_cache_files(cache_files: dict) -> None:
    for name, info in cache_files.items():
        status = "exists" if info.get("exists") else "missing"
        print(f"  {name}: {status}, {info.get('size', 0)} bytes, {info.get('path')}")


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    retriever = PlainVectorRAGRetriever(
        input_dir=_optional_path(args.input_dir),
        cache_dir=_optional_path(args.cache_dir),
        embedding_backend=args.embedding_backend or settings.plain_rag_embedding_backend,
        embedding_model=args.embedding_model or settings.plain_rag_embedding_model,
        batch_size=args.batch_size,
    )

    print("开始构建普通向量 RAG 索引/cache...")
    print(f"项目根目录: {ROOT}")
    print(f"输入目录: {retriever.input_dir}")
    print(f"缓存目录: {retriever.cache_dir}")
    print(f"Embedding: {retriever.embedding_backend}/{retriever.embedding_model}")
    print(f"Chunk: size={retriever.chunk_size}, overlap={retriever.chunk_overlap}")
    print(f"Force rebuild: {args.force}")

    started = time.monotonic()
    stats = await retriever.build_index(force=args.force)
    elapsed = time.monotonic() - started

    print("\n[OK] 普通向量 RAG 索引/cache 就绪")
    print(f"Cache hit: {stats['cache_hit']}")
    print(f"Chunk count: {stats['chunk_count']}")
    print(f"Embedding shape: {stats['embedding_shape']}")
    print(f"Elapsed: {elapsed:.2f}s")
    print("Cache files:")
    _print_cache_files(stats["cache_files"])

    if not args.no_smoke:
        query = args.query or DEFAULT_SMOKE_QUERY
        print(f"\nSmoke query: {query}")
        result = await retriever.retrieve(
            query,
            top_k=args.top_k,
            use_llm_parsing=False,
            mode=PLAIN_VECTOR_RAG_MODE,
        )
        meta = getattr(result, "_plain_rag_meta", {}) or {}
        top_scores = meta.get("top_scores") or []
        if not top_scores:
            print("  未返回 chunk。")
        for item in top_scores:
            print(
                "  "
                f"rank={item.get('rank')} "
                f"score={float(item.get('score', 0.0)):.4f} "
                f"source={item.get('source_file')} "
                f"chunk={item.get('chunk_index')} "
                f"title={item.get('title') or ''}"
            )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the plain vector RAG cache without rebuilding GraphRAG.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and rebuild.")
    parser.add_argument("--input-dir", help="Plain RAG input directory. Defaults to PLAIN_RAG_INPUT_DIR.")
    parser.add_argument("--cache-dir", help="Plain RAG cache directory. Defaults to PLAIN_RAG_CACHE_DIR.")
    parser.add_argument("--embedding-backend", help="Embedding backend, e.g. hunyuan or ollama.")
    parser.add_argument("--embedding-model", help="Embedding model name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Chunk embedding batch size.")
    parser.add_argument("--query", default=DEFAULT_SMOKE_QUERY, help="Smoke-test query.")
    parser.add_argument("--top-k", type=int, default=5, help="Smoke-test top-k chunks.")
    parser.add_argument("--no-smoke", action="store_true", help="Skip smoke retrieval after building.")
    args = parser.parse_args()

    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
