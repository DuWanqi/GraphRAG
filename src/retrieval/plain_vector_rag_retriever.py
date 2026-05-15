"""Plain text vector RAG retriever.

This retriever is intentionally independent from GraphRAG output artifacts. It
loads the original input files, chunks their text, embeds the chunks, and uses
cosine similarity over an in-memory numpy matrix.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..config import get_settings
from .memoir_parser import MemoirParser
from .memoir_retriever import RetrievalResult


logger = logging.getLogger(__name__)


PLAIN_VECTOR_RAG_MODE = "plain_vector_rag"
_CACHE_VERSION = 1
_JSON_TEXT_FIELDS = (
    "title",
    "date",
    "location",
    "keywords",
    "content",
    "summary",
    "description",
    "source",
    "url",
)
ProgressCallback = Callable[[Dict[str, Any]], None]


class PlainVectorRAGEmbeddingError(RuntimeError):
    """Raised when the plain RAG embedding backend cannot serve vectors."""


@dataclass
class PlainRAGChunk:
    """A source text chunk plus metadata used for display and cache."""

    text: str
    source_file: str
    chunk_index: int
    record_index: Optional[int] = None
    title: str = ""
    date: str = ""
    location: str = ""
    score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlainRAGChunk":
        return cls(
            text=str(data.get("text", "")),
            source_file=str(data.get("source_file", "")),
            chunk_index=int(data.get("chunk_index", 0)),
            record_index=data.get("record_index"),
            title=str(data.get("title", "")),
            date=str(data.get("date", "")),
            location=str(data.get("location", "")),
            score=data.get("score"),
            extra=dict(data.get("extra") or {}),
        )


class PlainVectorRAGRetriever:
    """Embedding-based plain RAG retriever over raw text/JSON inputs."""

    def __init__(
        self,
        input_dir: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        embedding_backend: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_client: Optional[Any] = None,
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        batch_size: int = 16,
    ) -> None:
        settings = get_settings()
        self.input_dir = Path(
            input_dir
            or getattr(settings, "plain_rag_input_dir", None)
            or (Path(settings.graphrag_output_dir) / "input")
        )
        self.cache_dir = Path(
            cache_dir
            or getattr(settings, "plain_rag_cache_dir", None)
            or (Path(__file__).resolve().parent.parent.parent / "data" / "plain_vector_rag_cache")
        )
        self.embedding_backend = (
            embedding_backend
            or getattr(settings, "plain_rag_embedding_backend", "ollama")
        ).strip().lower()
        self.embedding_model = (
            embedding_model
            or getattr(settings, "plain_rag_embedding_model", "nomic-embed-text")
        ).strip()
        self.ollama_api_base = getattr(settings, "ollama_api_base", "http://localhost:11434")
        self.embedding_client = embedding_client
        self.chunk_size = max(200, int(chunk_size))
        self.chunk_overlap = max(0, min(int(chunk_overlap), self.chunk_size - 1))
        self.batch_size = max(1, int(batch_size))

        self._chunks: List[PlainRAGChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index_ready = False
        self._build_lock = asyncio.Lock()

    async def retrieve(
        self,
        memoir_text: str,
        top_k: int = 10,
        use_llm_parsing: bool = False,
        mode: str = PLAIN_VECTOR_RAG_MODE,
    ) -> RetrievalResult:
        """Return top-k chunks as a standard RetrievalResult."""

        parser = MemoirParser()
        context = parser.parse(memoir_text, use_llm=use_llm_parsing)
        query = self._build_query(memoir_text, context)

        await self._ensure_index()

        result = RetrievalResult(context=context, query=query)
        if self._embeddings is None or self._embeddings.size == 0 or not self._chunks:
            result._plain_rag_meta = self._meta(top_scores=[])  # type: ignore[attr-defined]
            return result

        query_vector = await self._embed_one(query)
        query_vector_np = self._normalize_matrix(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._embeddings @ query_vector_np
        order = np.argsort(-scores)[: max(1, int(top_k))]

        top_scores = []
        text_units: List[str] = []
        for rank, idx in enumerate(order, start=1):
            chunk = self._chunks[int(idx)]
            score = float(scores[int(idx)])
            top_scores.append(
                {
                    "rank": rank,
                    "score": score,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                }
            )
            text_units.append(self._format_text_unit(chunk, score))

        result.text_units = text_units
        result._plain_rag_meta = self._meta(top_scores=top_scores)  # type: ignore[attr-defined]
        return result

    async def build_index(
        self,
        force: bool = False,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Build or load the plain vector RAG cache and return index stats."""
        if force:
            self._clear_cache_files()
            self._index_ready = False
            self._chunks = []
            self._embeddings = None

        cache_hit = await self._ensure_index(progress_callback=progress_callback)
        shape = list(self._embeddings.shape) if self._embeddings is not None else [0, 0]
        return {
            "input_dir": str(self.input_dir),
            "cache_dir": str(self.cache_dir),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chunk_count": len(self._chunks),
            "embedding_shape": shape,
            "cache_hit": bool(cache_hit),
            "cache_files": self._cache_file_status(),
        }

    async def _ensure_index(
        self,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> bool:
        if self._index_ready:
            return True

        async with self._build_lock:
            if self._index_ready:
                return True
            manifest = self._build_manifest()
            if self._load_cache(manifest):
                self._index_ready = True
                self._emit_progress(
                    progress_callback,
                    {
                        "event": "cache_loaded",
                        "chunk_count": len(self._chunks),
                    },
                )
                return True

            chunks = self._load_chunks()
            self._emit_progress(
                progress_callback,
                {
                    "event": "chunks_loaded",
                    "input_file_count": len(manifest.get("files") or []),
                    "chunk_count": len(chunks),
                },
            )
            embeddings = await self._embed_chunks(
                chunks,
                progress_callback=progress_callback,
            )
            self._chunks = chunks
            self._embeddings = embeddings
            self._save_cache(manifest)
            self._emit_progress(
                progress_callback,
                {
                    "event": "cache_saved",
                    "cache_files": self._cache_file_status(),
                },
            )
            self._index_ready = True
            return False

    def _build_manifest(self) -> Dict[str, Any]:
        files = []
        if self.input_dir.exists():
            for path in sorted(self.input_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in {".txt", ".json"}:
                    stat = path.stat()
                    files.append(
                        {
                            "path": path.name,
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    )

        return {
            "version": _CACHE_VERSION,
            "input_dir": str(self.input_dir.resolve()),
            "files": files,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def _load_cache(self, manifest: Dict[str, Any]) -> bool:
        manifest_path = self.cache_dir / "manifest.json"
        chunks_path = self.cache_dir / "chunks.jsonl"
        embeddings_path = self.cache_dir / "embeddings.npy"

        if not (manifest_path.exists() and chunks_path.exists() and embeddings_path.exists()):
            return False

        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest != manifest:
                return False
            chunks = []
            with chunks_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(PlainRAGChunk.from_dict(json.loads(line)))
            embeddings = np.load(embeddings_path)
            if len(chunks) != int(embeddings.shape[0]):
                return False
            self._chunks = chunks
            self._embeddings = embeddings.astype(np.float32, copy=False)
            logger.info("[PlainVectorRAG] loaded cache: %d chunks", len(chunks))
            return True
        except Exception as exc:
            logger.warning("[PlainVectorRAG] cache load failed: %s", exc)
            return False

    def _save_cache(self, manifest: Dict[str, Any]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (self.cache_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in self._chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            embeddings = self._embeddings
            if embeddings is None:
                embeddings = np.zeros((0, 0), dtype=np.float32)
            np.save(self.cache_dir / "embeddings.npy", embeddings.astype(np.float32, copy=False))
        except Exception as exc:
            logger.warning("[PlainVectorRAG] cache save failed: %s", exc)

    def _cache_paths(self) -> Dict[str, Path]:
        return {
            "manifest": self.cache_dir / "manifest.json",
            "chunks": self.cache_dir / "chunks.jsonl",
            "embeddings": self.cache_dir / "embeddings.npy",
        }

    def _cache_file_status(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for name, path in self._cache_paths().items():
            exists = path.exists()
            out[name] = {
                "path": str(path),
                "exists": exists,
                "size": path.stat().st_size if exists else 0,
            }
        return out

    def _clear_cache_files(self) -> None:
        for path in self._cache_paths().values():
            if path.exists():
                path.unlink()

    def _load_chunks(self) -> List[PlainRAGChunk]:
        if not self.input_dir.exists():
            logger.warning("[PlainVectorRAG] input dir does not exist: %s", self.input_dir)
            return []

        chunks: List[PlainRAGChunk] = []
        chunk_index = 0
        for path in sorted(self.input_dir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".txt":
                text = self._read_text(path)
                for chunk_text in self._split_text(text):
                    chunks.append(
                        PlainRAGChunk(
                            text=chunk_text,
                            source_file=path.name,
                            chunk_index=chunk_index,
                        )
                    )
                    chunk_index += 1
            elif suffix == ".json":
                for record_index, record in enumerate(self._load_json_records(path)):
                    text = self._record_to_text(record)
                    if not text:
                        continue
                    metadata = {
                        "title": self._stringify(record.get("title")),
                        "date": self._stringify(record.get("date")),
                        "location": self._stringify(record.get("location")),
                    }
                    for chunk_text in self._split_text(text):
                        chunks.append(
                            PlainRAGChunk(
                                text=chunk_text,
                                source_file=path.name,
                                chunk_index=chunk_index,
                                record_index=record_index,
                                title=metadata["title"],
                                date=metadata["date"],
                                location=metadata["location"],
                            )
                        )
                        chunk_index += 1

        logger.info("[PlainVectorRAG] loaded %d chunks from %s", len(chunks), self.input_dir)
        return chunks

    def _read_text(self, path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="ignore")

    def _load_json_records(self, path: Path) -> List[Dict[str, Any]]:
        try:
            raw = self._read_text(path)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("[PlainVectorRAG] skip invalid json %s: %s", path, exc)
            return []
        return list(self._iter_json_records(data))

    def _iter_json_records(self, data: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(data, list):
            for item in data:
                yield from self._iter_json_records(item)
            return

        if not isinstance(data, dict):
            return

        if any(field in data for field in _JSON_TEXT_FIELDS):
            yield data
            return

        for value in data.values():
            if isinstance(value, (list, dict)):
                yield from self._iter_json_records(value)

    def _record_to_text(self, record: Dict[str, Any]) -> str:
        parts = []
        labels = {
            "title": "标题",
            "date": "日期",
            "location": "地点",
            "keywords": "关键词",
            "content": "正文",
            "summary": "摘要",
            "description": "描述",
            "source": "来源",
            "url": "URL",
        }
        for field_name in _JSON_TEXT_FIELDS:
            value = self._stringify(record.get(field_name))
            if value:
                parts.append(f"{labels[field_name]}: {value}")
        return "\n".join(parts).strip()

    def _stringify(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            return "、".join(self._stringify(v) for v in value if self._stringify(v))
        if isinstance(value, dict):
            return " ".join(self._stringify(v) for v in value.values() if self._stringify(v))
        return re.sub(r"\s+", " ", str(value)).strip()

    def _split_text(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: List[str] = []
        current = ""
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._window_text(paragraph))
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = paragraph

        if current:
            chunks.append(current.strip())

        final_chunks: List[str] = []
        for chunk in chunks:
            final_chunks.extend(self._window_text(chunk) if len(chunk) > self.chunk_size else [chunk])
        return [c for c in final_chunks if len(c.strip()) >= 20]

    def _window_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()]

        step = self.chunk_size - self.chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start += step
        return chunks

    async def _embed_chunks(
        self,
        chunks: Sequence[PlainRAGChunk],
        progress_callback: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 0), dtype=np.float32)

        embeddings: List[List[float]] = []
        texts = [chunk.text for chunk in chunks]
        total_chunks = len(texts)
        total_batches = (total_chunks + self.batch_size - 1) // self.batch_size
        embedding_started = time.monotonic()
        for batch_index, start in enumerate(range(0, total_chunks, self.batch_size), start=1):
            batch = texts[start : start + self.batch_size]
            batch_started = time.monotonic()
            vectors = await self._embed_many(batch)
            embeddings.extend(vectors)
            now = time.monotonic()
            completed_chunks = len(embeddings)
            elapsed = now - embedding_started
            remaining_chunks = max(0, total_chunks - completed_chunks)
            eta = (elapsed / completed_chunks * remaining_chunks) if completed_chunks else None
            self._emit_progress(
                progress_callback,
                {
                    "event": "embedding_batch",
                    "batch_index": batch_index,
                    "batch_count": total_batches,
                    "batch_size": len(batch),
                    "completed_chunks": completed_chunks,
                    "total_chunks": total_chunks,
                    "batch_elapsed": now - batch_started,
                    "elapsed": elapsed,
                    "eta": eta,
                },
            )

        matrix = np.asarray(embeddings, dtype=np.float32)
        return self._normalize_matrix(matrix)

    def _emit_progress(
        self,
        progress_callback: Optional[ProgressCallback],
        event: Dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(dict(event))
        except Exception as exc:
            logger.warning("[PlainVectorRAG] progress callback failed: %s", exc)

    async def _embed_one(self, text: str) -> List[float]:
        try:
            client = self._get_embedding_client()
            if hasattr(client, "embed_async"):
                return list(await client.embed_async(text))
            if hasattr(client, "embed"):
                result = client.embed(text)
                if inspect.isawaitable(result):
                    result = await result
                return list(result)
            raise RuntimeError("Embedding client does not provide embed/embed_async")
        except PlainVectorRAGEmbeddingError:
            raise
        except Exception as exc:
            raise self._embedding_error(exc) from exc

    async def _embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        try:
            client = self._get_embedding_client()
            if hasattr(client, "embed_batch_async"):
                return [list(v) for v in await client.embed_batch_async(list(texts))]
            if hasattr(client, "embed_batch"):
                result = client.embed_batch(list(texts))
                if inspect.isawaitable(result):
                    result = await result
                return [list(v) for v in result]
        except PlainVectorRAGEmbeddingError:
            raise
        except Exception as exc:
            raise self._embedding_error(exc) from exc
        return [await self._embed_one(text) for text in texts]

    def _get_embedding_client(self) -> Any:
        if self.embedding_client is not None:
            return self.embedding_client

        if self.embedding_backend == "ollama":
            from ..llm.ollama_embedding import OllamaEmbedding, OllamaEmbeddingConfig

            settings = get_settings()
            self.embedding_client = OllamaEmbedding(
                OllamaEmbeddingConfig(
                    base_url=self.ollama_api_base,
                    model=self.embedding_model,
                )
            )
        elif self.embedding_backend in {"local", "sentence-transformers", "sentence_transformers"}:
            from ..llm.local_embedding import LocalEmbedding

            self.embedding_client = LocalEmbedding()
        elif self.embedding_backend in {"hunyuan", "tencent", "tencent-hunyuan"}:
            from ..llm.hunyuan_embedding import HunyuanEmbedding, HunyuanEmbeddingConfig

            settings = get_settings()
            self.embedding_client = HunyuanEmbedding(
                HunyuanEmbeddingConfig(
                    api_key=getattr(settings, "hunyuan_api_key", None),
                    base_url=(
                        getattr(settings, "hunyuan_api_base", None)
                        or "https://api.hunyuan.cloud.tencent.com/v1"
                    ),
                    model=self.embedding_model or "hunyuan-embedding",
                )
            )
        else:
            raise RuntimeError(
                f"Unsupported plain RAG embedding backend: {self.embedding_backend}"
            )
        return self.embedding_client

    def _embedding_error(self, exc: Exception) -> PlainVectorRAGEmbeddingError:
        raw = str(exc)
        if self.embedding_backend == "ollama":
            connection_markers = (
                "Cannot connect to host",
                "Connect call failed",
                "Connection refused",
                "WinError 10061",
                "远程计算机拒绝网络连接",
            )
            if any(marker in raw for marker in connection_markers):
                return PlainVectorRAGEmbeddingError(
                    "普通向量 RAG 无法连接 Ollama embedding 服务 "
                    f"({self.ollama_api_base})。请先运行 `ollama serve`，并确认已安装模型 "
                    f"`ollama pull {self.embedding_model}`。原始错误: {raw}"
                )
            return PlainVectorRAGEmbeddingError(
                "普通向量 RAG 调用 Ollama embedding 失败，"
                f"请确认模型 `{self.embedding_model}` 可用。原始错误: {raw}"
            )
        if self.embedding_backend in {"local", "sentence-transformers", "sentence_transformers"}:
            return PlainVectorRAGEmbeddingError(
                "普通向量 RAG 本地 embedding 加载/调用失败，"
                "请确认已安装 sentence-transformers 且模型缓存可用。"
                f"原始错误: {raw}"
            )
        if self.embedding_backend in {"hunyuan", "tencent", "tencent-hunyuan"}:
            return PlainVectorRAGEmbeddingError(
                "普通向量 RAG 调用腾讯混元 embedding API 失败，"
                "请确认已配置 `HUNYUAN_API_KEY`，且 `PLAIN_RAG_EMBEDDING_MODEL` "
                f"为可用模型（当前 `{self.embedding_model}`）。原始错误: {raw}"
            )
        return PlainVectorRAGEmbeddingError(
            f"普通向量 RAG embedding 失败 ({self.embedding_backend}): {raw}"
        )

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32, copy=False)

    def _build_query(self, memoir_text: str, context: Any) -> str:
        parts = []
        if getattr(context, "year", None):
            parts.append(str(context.year))
        if getattr(context, "location", None):
            parts.append(str(context.location))
        keywords = getattr(context, "keywords", None) or []
        parts.extend(str(k) for k in keywords[:8] if k)
        parts.append((memoir_text or "").strip())
        return "\n".join(p for p in parts if p).strip()

    def _format_text_unit(self, chunk: PlainRAGChunk, score: float) -> str:
        meta = [
            f"source={chunk.source_file}",
            f"chunk={chunk.chunk_index}",
            f"score={score:.4f}",
        ]
        if chunk.title:
            meta.append(f"title={chunk.title}")
        if chunk.date:
            meta.append(f"date={chunk.date}")
        if chunk.location:
            meta.append(f"location={chunk.location}")
        return f"【普通RAG来源】{' | '.join(meta)}\n{chunk.text}"

    def _meta(self, top_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "backend": "plain_vector_rag",
            "input_dir": str(self.input_dir),
            "cache_dir": str(self.cache_dir),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "chunk_count": len(self._chunks),
            "top_scores": top_scores,
        }
