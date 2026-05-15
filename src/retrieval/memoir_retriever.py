"""
回忆录历史背景检索器
基于GraphRAG进行知识图谱检索
支持关键词检索、向量检索、混合检索三种模式
"""

import os
import asyncio
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .memoir_parser import MemoirParser, MemoirContext


class RetrievalMode(Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """检索结果"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    text_units: List[str] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    context: Optional[MemoirContext] = None
    query: str = ""

    @property
    def has_results(self) -> bool:
        """Return whether any retrieval channel produced evidence."""
        return bool(self.entities or self.relationships or self.text_units or self.communities)

    def get_context_text(self, max_chars: int = 8000) -> str:
        """Build compact evidence text used by evaluators and fact checkers."""
        parts: List[str] = []

        if self.text_units:
            parts.append("【相关文本】")
            for i, text in enumerate(self.text_units[:8], 1):
                cleaned = str(text).strip()
                if cleaned:
                    parts.append(f"{i}. {cleaned}")

        if self.entities:
            parts.append("【相关实体】")
            for i, entity in enumerate(self.entities[:10], 1):
                name = str(entity.get("name", "")).strip()
                desc = str(entity.get("description", "")).strip()
                if name or desc:
                    parts.append(f"{i}. {name}: {desc}".strip())

        if self.relationships:
            parts.append("【相关关系】")
            for i, rel in enumerate(self.relationships[:8], 1):
                source = str(rel.get("source", "")).strip()
                target = str(rel.get("target", "")).strip()
                desc = str(rel.get("description", "")).strip()
                label = " -> ".join(x for x in [source, target] if x)
                parts.append(f"{i}. {label}: {desc}".strip())

        if self.communities:
            parts.append("【社区摘要】")
            for i, community in enumerate(self.communities[:3], 1):
                title = str(community.get("title", "")).strip()
                summary = str(community.get("summary", "")).strip()
                if title or summary:
                    parts.append(f"{i}. {title}: {summary}".strip())

        context = "\n".join(parts)
        return context[:max_chars]
    
    def merge(self, other: 'RetrievalResult') -> 'RetrievalResult':
        """合并另一个检索结果"""
        self.entities.extend(other.entities)
        self.relationships.extend(other.relationships)
        self.text_units.extend(other.text_units)
        self.communities.extend(other.communities)
        return self


class MemoirRetriever:
    """回忆录检索器"""
    
    def __init__(self, llm_adapter=None, index_dir: Optional[Union[str, Path]] = None):
        self.llm_adapter = llm_adapter
        default_root = Path(__file__).resolve().parent.parent.parent / "data" / "graphrag_output"
        self.index_dir = Path(index_dir) if index_dir is not None else default_root
        
        # 缓存数据
        self._entities_df = None
        self._relationships_df = None
        self._text_units_df = None
        self._communities_df = None
        self._vector_retriever = None
        self._entity_rows_by_id: Dict[str, Dict[str, Any]] = {}
        self._text_unit_rows_by_id: Dict[str, Dict[str, Any]] = {}
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        # 兼容 GraphRAG 1.x 和 2.x
        entity_files = [
            output_dir / "entities.parquet",
            output_dir / "create_final_entities.parquet"
        ]
        relationship_files = [
            output_dir / "relationships.parquet",
            output_dir / "create_final_relationships.parquet"
        ]
        text_unit_files = [
            output_dir / "text_units.parquet",
            output_dir / "create_final_text_units.parquet"
        ]
        community_files = [
            output_dir / "community_reports.parquet",
            output_dir / "communities.parquet",
            output_dir / "create_final_community_reports.parquet"
        ]
        
        for file_path in entity_files:
            if file_path.exists():
                self._entities_df = pd.read_parquet(file_path)
                break
        
        for file_path in relationship_files:
            if file_path.exists():
                self._relationships_df = pd.read_parquet(file_path)
                break
        
        for file_path in text_unit_files:
            if file_path.exists():
                self._text_units_df = pd.read_parquet(file_path)
                break
        
        for file_path in community_files:
            if file_path.exists():
                self._communities_df = pd.read_parquet(file_path)
                break

        self._entity_rows_by_id = self._build_row_map(self._entities_df)
        self._text_unit_rows_by_id = self._build_row_map(self._text_units_df)
        
        print(f"[DEBUG] 加载实体: {len(self._entities_df) if self._entities_df is not None else 0} 条")
        print(f"[DEBUG] 加载关系: {len(self._relationships_df) if self._relationships_df is not None else 0} 条")
        print(f"[DEBUG] 加载文本单元: {len(self._text_units_df) if self._text_units_df is not None else 0} 条")
    
    def is_index_ready(self) -> bool:
        """检查索引是否就绪"""
        output_dir = self.index_dir / "output"
        new_format = output_dir / "entities.parquet"
        old_format = output_dir / "create_final_entities.parquet"
        return new_format.exists() or old_format.exists()
    
    async def retrieve(
        self,
        memoir_text: str,
        top_k: int = 10,
        use_llm_parsing: bool = False,
        mode: str = "hybrid"
    ) -> RetrievalResult:
        """执行检索"""
        # 解析回忆录文本，提取上下文信息
        parser = MemoirParser()
        context = parser.parse(memoir_text, use_llm=use_llm_parsing)
        
        # 根据模式执行检索
        if mode == "keyword":
            return await self._keyword_retrieve(context, top_k)
        elif mode == "vector":
            return await self._vector_retrieve(context, top_k)
        else:
            return await self._hybrid_retrieve(context, top_k)
    
    def retrieve_sync(self, memoir_text: str, top_k: int = 10) -> RetrievalResult:
        """同步版本的检索"""
        return asyncio.run(self.retrieve(memoir_text, top_k, use_llm_parsing=False))
    
    async def _keyword_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """关键词检索"""
        result = RetrievalResult()
        result.context = context
        result.query = context.to_query()
        
        # 搜索实体、关系、文本单元和社区报告
        result.entities = self._search_entities(context, top_k)
        result.relationships = self._search_relationships(context, top_k)
        result.text_units = self._search_text_units(context, top_k)
        result.communities = self._search_communities(context, top_k)
        
        return result
    
    async def _vector_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """GraphRAG Local Search 风格检索。

        入口是实体描述向量召回；随后沿图谱扩展相关关系、文本单元和社区。
        这比普通 chunk 向量检索多利用了 GraphRAG 产出的结构化图谱。
        """
        result = RetrievalResult(context=context, query=self._build_local_search_query(context))
        vector_retriever = self._get_vector_retriever()

        if not vector_retriever.is_ready():
            raise RuntimeError("GraphRAG 向量索引未就绪：未找到 entity_description LanceDB 表。")

        entity_limit = max(top_k * 8, 60)
        text_limit = max(top_k * 3, 20)

        entity_hits = await vector_retriever.search_entities(result.query, top_k=entity_limit)
        keyword_entity_seeds = self._search_entities(context, min(top_k, 10))
        for seed in keyword_entity_seeds:
            seed["source"] = "graphrag_local_keyword_seed"
            seed["distance"] = 0.7

        vector_text_records = await vector_retriever.search_text_unit_records(
            result.query, top_k=text_limit
        )
        keyword_text_records = [
            {
                "id": "",
                "text": text,
                "score": 0.7,
                "distance": 0.7,
                "source": "keyword_seed",
                "metadata": {},
            }
            for text in self._search_text_units(context, min(top_k, 8))
        ]
        vector_text_records = vector_text_records + keyword_text_records
        seed_text_ids = {
            str(r.get("id", "")).strip() for r in vector_text_records if str(r.get("id", "")).strip()
        }
        text_entity_ids = self._entity_ids_from_text_records(vector_text_records)
        text_neighbor_entities = self._entities_from_ids(
            text_entity_ids,
            source="graphrag_local_text_unit_neighbor",
        )

        entity_hits = self._dedupe_entities(
            entity_hits + keyword_entity_seeds + text_neighbor_entities
        )
        entity_hits = self._rerank_entities(entity_hits, context)
        selected_entities = entity_hits[: min(top_k, 10)]
        selected_entity_ids = {
            str(e.get("id", "")).strip() for e in selected_entities if str(e.get("id", "")).strip()
        } | text_entity_ids
        selected_entity_names = {
            str(e.get("name", "")).strip() for e in selected_entities if str(e.get("name", "")).strip()
        }
        text_records = self._collect_local_text_units(
            vector_text_records,
            selected_entity_ids,
            context,
            top_k=top_k,
        )
        selected_text_ids = {
            str(r.get("id", "")).strip() for r in text_records if str(r.get("id", "")).strip()
        } | seed_text_ids

        relationships = self._collect_local_relationships(
            selected_entity_names,
            selected_text_ids,
            context,
            top_k=top_k,
        )
        selected_relationship_ids = {
            str(r.get("id", "")).strip() for r in relationships if str(r.get("id", "")).strip()
        }

        # 再用关系补充文本单元，贴近官方 Local Search 的 graph neighborhood 思路。
        text_records = self._collect_local_text_units(
            text_records,
            selected_entity_ids,
            context,
            top_k=top_k,
            relationship_ids=selected_relationship_ids,
        )

        result.entities = selected_entities
        result.relationships = relationships
        result.text_units = [self._format_local_text_unit(r) for r in text_records]
        result.communities = self._collect_local_communities(
            selected_entity_ids,
            selected_relationship_ids,
            context,
            top_k=min(top_k, 3),
        )

        return result
    
    async def _hybrid_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """混合检索：GraphRAG Local Search + 关键词规则补充。"""
        vector_result = await self._vector_retrieve(context, top_k)
        keyword_result = await self._keyword_retrieve(context, top_k)
        return self._merge_retrieval_results(vector_result, keyword_result, top_k)

    @staticmethod
    def _build_row_map(df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        if df is None or df.empty or "id" not in df.columns:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_id = str(row_dict.get("id", "")).strip()
            if row_id:
                out[row_id] = row_dict
        return out

    @staticmethod
    def _as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, (tuple, set)):
            return list(value)
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            return [s]
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        return [value]

    def _get_vector_retriever(self):
        if self._vector_retriever is None:
            from .vector_retriever import VectorRetriever

            self._vector_retriever = VectorRetriever(
                index_dir=str(self.index_dir),
                llm_adapter=self.llm_adapter,
            )
        return self._vector_retriever

    def _build_local_search_query(self, context: MemoirContext) -> str:
        parts: List[str] = []
        if context.original_text:
            parts.append(context.original_text[:500])
        if context.year:
            parts.append(f"时间：{context.year}年")
        if context.location:
            parts.append(f"地点：{context.location}")
        keywords = [str(k) for k in (context.keywords or [])[:8] if k]
        if keywords:
            parts.append("关键词：" + "、".join(keywords))
        return "\n".join(parts).strip() or context.to_query()

    def _context_overlap_score(self, text: str, context: MemoirContext) -> float:
        haystack = (text or "").upper()
        score = 0.0
        if context.year and str(context.year).upper() in haystack:
            score += 25.0
        location_aliases = {
            "深圳": ["SHENZHEN"],
            "广州": ["GUANGZHOU", "CANTON"],
            "广东": ["GUANGDONG"],
            "香港": ["HONG KONG"],
            "上海": ["SHANGHAI"],
            "北京": ["BEIJING"],
            "佛山": ["FOSHAN"],
            "东莞": ["DONGGUAN"],
            "柳州": ["LIUZHOU"],
            "广西": ["GUANGXI"],
        }
        if context.location:
            loc_terms = [str(context.location), *location_aliases.get(str(context.location), [])]
            if any(term.upper() in haystack for term in loc_terms):
                score += 20.0
        for kw in (context.keywords or [])[:8]:
            if kw and str(kw).upper() in haystack:
                score += 8.0
        return score

    def _entity_ids_from_text_records(self, records: List[Dict[str, Any]]) -> Set[str]:
        entity_ids: Set[str] = set()
        for record in records:
            metadata = record.get("metadata") or {}
            for entity_id in self._as_list(metadata.get("entity_ids")):
                entity_id = str(entity_id).strip()
                if entity_id:
                    entity_ids.add(entity_id)
        return entity_ids

    def _entities_from_ids(self, entity_ids: Set[str], source: str) -> List[Dict[str, Any]]:
        entities = []
        for entity_id in entity_ids:
            row = self._entity_rows_by_id.get(entity_id)
            if not row:
                continue
            entities.append({
                "id": entity_id,
                "name": row.get("title") or row.get("name", ""),
                "description": row.get("description", ""),
                "type": row.get("type", "unknown"),
                "score": 0.8,
                "distance": 0.8,
                "source": source,
            })
        return entities

    @staticmethod
    def _dedupe_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids = set()
        seen_names = set()
        out = []
        for entity in entities:
            entity_id = str(entity.get("id", "")).strip()
            name = str(entity.get("name", "")).strip().lower()
            key = entity_id or name
            if not key or key in seen_ids or name in seen_names:
                continue
            seen_ids.add(key)
            if name:
                seen_names.add(name)
            out.append(entity)
        return out

    def _rerank_entities(
        self,
        entities: List[Dict[str, Any]],
        context: MemoirContext,
    ) -> List[Dict[str, Any]]:
        reranked = []
        for rank, entity in enumerate(entities):
            e = dict(entity)
            if self._is_low_value_local_entity(e):
                continue
            distance = float(e.get("distance", e.get("score", 1.0)) or 1.0)
            text = f"{e.get('name', '')} {e.get('description', '')}"
            relevance = 100.0 / (1.0 + max(distance, 0.0))
            local_score = relevance + self._context_overlap_score(text, context) - rank * 0.01
            e["distance"] = distance
            e["score"] = round(local_score, 4)
            e["source"] = "graphrag_local_entity_vector"
            reranked.append(e)
        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reranked

    @staticmethod
    def _is_low_value_local_entity(entity: Dict[str, Any]) -> bool:
        name = str(entity.get("name", "")).strip()
        desc = str(entity.get("description", "")).strip()
        entity_type = str(entity.get("type", "")).upper()
        if not name:
            return True
        if re.fullmatch(r"\d{4}(?:年)?(?:\d{1,2}月\d{1,2}日)?", name):
            return True
        if re.fullmatch(r"\d{1,2}月\d{1,2}日", name):
            return True
        if entity_type in {"DATE", "TIME"} and len(desc) < 40:
            return True
        generic_entities = {
            "中国", "中华人民共和国", "国家", "政府", "中央", "国务院",
            "北京", "上海", "STATE COUNCIL", "CPC CENTRAL COMMITTEE",
        }
        return name.upper() in {g.upper() for g in generic_entities}

    def _collect_local_text_units(
        self,
        seed_records: List[Dict[str, Any]],
        entity_ids: Set[str],
        context: MemoirContext,
        top_k: int,
        relationship_ids: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        relationship_ids = relationship_ids or set()
        candidates: Dict[str, Dict[str, Any]] = {}

        def add_record(record: Dict[str, Any], graph_bonus: float = 0.0) -> None:
            text = str(record.get("text", "")).strip()
            if not text:
                return
            text_id = str(record.get("id", "")).strip()
            key = text_id or text[:120]
            distance = float(record.get("distance", record.get("score", 1.0)) or 1.0)
            vector_score = 100.0 / (1.0 + max(distance, 0.0))
            score = vector_score + graph_bonus + self._context_overlap_score(text, context)
            existing = candidates.get(key)
            if existing is None or score > existing.get("score", 0.0):
                item = dict(record)
                item["id"] = text_id
                item["score"] = round(score, 4)
                item["distance"] = distance
                candidates[key] = item

        for record in seed_records:
            add_record(record, graph_bonus=5.0 if record.get("source") == "vector" else 0.0)

        for entity_id in entity_ids:
            entity_row = self._entity_rows_by_id.get(entity_id)
            if not entity_row:
                continue
            for text_unit_id in self._as_list(entity_row.get("text_unit_ids"))[:4]:
                text_row = self._text_unit_rows_by_id.get(str(text_unit_id))
                if text_row:
                    add_record({
                        "id": str(text_unit_id),
                        "text": text_row.get("text", ""),
                        "distance": 1.0,
                        "source": "graph_entity_neighbor",
                        "metadata": text_row,
                    }, graph_bonus=35.0)

        if relationship_ids and self._relationships_df is not None:
            rel_df = self._relationships_df
            if "id" in rel_df.columns:
                rel_df = rel_df[rel_df["id"].astype(str).isin(relationship_ids)]
            for _, rel in rel_df.iterrows():
                for text_unit_id in self._as_list(rel.get("text_unit_ids"))[:3]:
                    text_row = self._text_unit_rows_by_id.get(str(text_unit_id))
                    if text_row:
                        add_record({
                            "id": str(text_unit_id),
                            "text": text_row.get("text", ""),
                            "distance": 1.0,
                            "source": "graph_relationship_neighbor",
                            "metadata": text_row,
                        }, graph_bonus=25.0)

        out = list(candidates.values())
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out[: min(top_k, 10)]

    def _collect_local_relationships(
        self,
        entity_names: Set[str],
        text_unit_ids: Set[str],
        context: MemoirContext,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if self._relationships_df is None or self._relationships_df.empty:
            return []

        candidates: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        entity_names_norm = {n.upper() for n in entity_names if n}

        for _, row in self._relationships_df.iterrows():
            source = str(row.get("source", row.get("source_title", ""))).strip()
            target = str(row.get("target", row.get("target_title", ""))).strip()
            desc = str(row.get("description", "")).strip()
            rel_text_units = {str(x) for x in self._as_list(row.get("text_unit_ids"))}

            source_hit = source.upper() in entity_names_norm
            target_hit = target.upper() in entity_names_norm
            text_hit = bool(rel_text_units & text_unit_ids)
            if not (source_hit or target_hit or text_hit):
                continue

            weight = float(row.get("weight", 1.0) or 1.0)
            graph_score = (20.0 if source_hit else 0.0) + (20.0 if target_hit else 0.0)
            graph_score += 15.0 if text_hit else 0.0
            graph_score += min(weight, 10.0)
            graph_score += self._context_overlap_score(f"{source} {target} {desc}", context)

            key = (source, target, desc[:80])
            item = {
                "id": str(row.get("id", "")).strip(),
                "source": source,
                "target": target,
                "description": desc,
                "weight": weight,
                "score": round(graph_score, 4),
                "retrieval_source": "graphrag_local_relationship",
            }
            if key not in candidates or graph_score > candidates[key].get("score", 0.0):
                candidates[key] = item

        out = list(candidates.values())
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out[: min(top_k, 10)]

    def _collect_local_communities(
        self,
        entity_ids: Set[str],
        relationship_ids: Set[str],
        context: MemoirContext,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if self._communities_df is None or self._communities_df.empty:
            return []

        candidates = []
        for _, row in self._communities_df.iterrows():
            row_entity_ids = {str(x) for x in self._as_list(row.get("entity_ids"))}
            row_relationship_ids = {str(x) for x in self._as_list(row.get("relationship_ids"))}
            entity_hits = row_entity_ids & entity_ids
            relationship_hits = row_relationship_ids & relationship_ids
            if not entity_hits and not relationship_hits:
                continue

            title = str(row.get("title", "") or f"Community {row.get('community', '')}").strip()
            summary = str(
                row.get("summary", "")
                or row.get("full_content", "")
                or self._summarize_community_row(row, entity_hits)
            ).strip()
            score = len(entity_hits) * 10.0 + len(relationship_hits) * 4.0
            score += self._context_overlap_score(f"{title} {summary}", context)
            candidates.append({
                "title": title,
                "summary": summary[:600],
                "score": round(score, 4),
                "id": str(row.get("id", "")).strip(),
                "source": "graphrag_local_community",
            })

        candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return candidates[:top_k]

    def _summarize_community_row(self, row: pd.Series, entity_hits: Set[str]) -> str:
        entity_names = []
        for entity_id in list(entity_hits)[:8]:
            entity_row = self._entity_rows_by_id.get(entity_id)
            if entity_row:
                title = entity_row.get("title") or entity_row.get("name")
                if title:
                    entity_names.append(str(title))
        size = row.get("size", "")
        period = row.get("period", "")
        parts = []
        if entity_names:
            parts.append("相关实体：" + "、".join(entity_names))
        if size:
            parts.append(f"社区规模：{size}")
        if period:
            parts.append(f"索引时期：{period}")
        return "；".join(parts) or str(row.get("title", ""))

    @staticmethod
    def _format_local_text_unit(record: Dict[str, Any]) -> str:
        text = str(record.get("text", "")).strip()
        source = record.get("source", "graphrag_local")
        score = record.get("score")
        text_id = str(record.get("id", "")).strip()
        meta = [f"source={source}"]
        if score is not None:
            meta.append(f"score={float(score):.2f}")
        if text_id:
            meta.append(f"id={text_id[:12]}")
        return f"【GraphRAG Local Search】{' | '.join(meta)}\n{text}"

    def _merge_retrieval_results(
        self,
        primary: RetrievalResult,
        secondary: RetrievalResult,
        top_k: int,
    ) -> RetrievalResult:
        merged = RetrievalResult(context=primary.context, query=primary.query or secondary.query)

        def dedupe_dicts(items: List[Dict[str, Any]], key_fields: Tuple[str, ...], limit: int) -> List[Dict[str, Any]]:
            seen = set()
            out = []
            for item in items:
                key = tuple(str(item.get(f, "")).lower() for f in key_fields)
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
                if len(out) >= limit:
                    break
            return out

        merged.entities = dedupe_dicts(
            primary.entities + secondary.entities,
            ("name",),
            min(top_k, 10),
        )
        merged.relationships = dedupe_dicts(
            primary.relationships + secondary.relationships,
            ("source", "target", "description"),
            min(top_k, 10),
        )
        seen_text = set()
        for text in primary.text_units + secondary.text_units:
            key = str(text).strip()[:160]
            if key and key not in seen_text:
                seen_text.add(key)
                merged.text_units.append(text)
            if len(merged.text_units) >= min(top_k, 10):
                break
        merged.communities = dedupe_dicts(
            primary.communities + secondary.communities,
            ("title", "summary"),
            min(top_k, 3),
        )
        return merged
    
    def _search_entities(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关实体（改进版：加权评分+主题相关性过滤）"""
        if self._entities_df is None or self._entities_df.empty:
            print(f"[_search_entities] 实体数据为空")
            return []
        
        # 需要过滤的通用实体（太泛泛，没有针对性）
        GENERIC_ENTITIES = {
            '中国', '中华人民共和国', '北京', '国务院', 'STATE COUNCIL',
            '中共中央', 'CPC CENTRAL COMMITTEE', '中央政府', '中央',
            'EUROPEAN UNION', '欧盟', 'BRICS',
        }
        
        # 需要过滤的不相关地点实体（与目标地点无关）
        UNRELATED_LOCATIONS = {
            'SHANGHAI', '北京', 'BEIJING', '上海',
            '天津', 'TIANJIN', '重庆', 'CHONGQING',
            '成都', 'CHENGDU', '武汉', 'WUHAN', '杭州', 'HANGZHOU',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高
        if context.location:
            search_terms.append((context.location, 20))
            location_map = {
                "深圳": "SHENZHEN", "北京": "BEIJING", "上海": "SHANGHAI",
                "广州": "GUANGZHOU", "香港": "HONG KONG",
                "广东": "GUANGDONG", "佛山": "FOSHAN", "东莞": "DONGGUAN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 20))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_entities] 搜索词: {search_terms}")
        print(f"[_search_entities] 实体总数: {len(self._entities_df)}")
        
        # 构建目标地点集合（用于严格匹配）
        target_locations = set()
        if context.location:
            target_locations.add(context.location)
            target_locations.add(context.location.upper())
            if context.location in location_map:
                target_locations.add(location_map[context.location])
                target_locations.add(location_map[context.location].lower())
        
        for _, row in self._entities_df.iterrows():
            entity_name = str(row.get("title", row.get("name", "")))
            entity_desc = str(row.get("description", ""))
            search_text = f"{entity_name} {entity_desc}".upper()
            
            # 过滤通用实体
            if entity_name in GENERIC_ENTITIES:
                continue
            
            # 过滤不相关地点实体
            if entity_name in UNRELATED_LOCATIONS:
                continue
            
            score = 0
            year_matched = False
            exact_year_matched = False  # 是否匹配目标年份
            location_matched = False
            topic_matched = False
            
            for term, weight in search_terms:
                term_upper = str(term).upper()
                if term_upper in search_text:
                    score += weight
                    if str(term) == str(context.year):
                        year_matched = True
                        exact_year_matched = True  # 精确匹配目标年份
                    elif str(term) == str(context.location) or term_upper in target_locations:
                        location_matched = True
                    elif term in topic_keywords:
                        topic_matched = True
            
            # 过滤策略：必须匹配年份
            if not year_matched:
                continue
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 15
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 25
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append({
                "id": row.get("id", ""),
                "name": entity_name,
                "type": row.get("type", "unknown"),
                "description": entity_desc,
                "text_unit_ids": row.get("text_unit_ids", []),
                "score": score,
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 去重：移除重复的实体（中英文变体）
        seen_names = set()
        unique_results = []
        for r in results:
            name_lower = r["name"].lower()
            if name_lower in seen_names:
                continue
            is_duplicate = False
            for seen in seen_names:
                if name_lower in seen or seen in name_lower:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen_names.add(name_lower)
                unique_results.append(r)
        
        print(f"[_search_entities] 匹配结果数: {len(results)}, 去重后: {len(unique_results)}")
        if unique_results:
            print(f"[_search_entities] 前3个结果: {[(r['name'], r['score']) for r in unique_results[:3]]}")
        out = unique_results[: min(top_k, 10)]  # 限制最多返回10个
        return out
    
    def _search_relationships(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关关系（改进版：加权评分+主题相关性过滤）"""
        if self._relationships_df is None or self._relationships_df.empty:
            return []
        
        # 需要过滤的不相关关系关键词
        UNRELATED_KEYWORDS = {
            '嫦娥', 'CHANG', '探月', '月球', 'BRICS', '金砖',
            '维和', '平潭', '滨海新区', '塘沽', '汉沽', '大港',
            '里斯本', '欧盟', 'EKATERINBURG', '嫦娥一号',
            'BINHAI', 'DAGANG', 'TANGGU', 'HANGU', 'LISBON',
        }
        
        # 需要过滤的不相关地点
        UNRELATED_LOCATIONS = {
            '上海', 'SHANGHAI', '北京', 'BEIJING', '天津', 'TIANJIN',
            '成都', 'CHENGDU', '武汉', 'WUHAN', '杭州', 'HANGZHOU',
            '莫斯科', 'MOSCOW', '叶卡捷琳堡',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高
        if context.location:
            search_terms.append((context.location, 20))
            location_map = {
                "广州": "GUANGZHOU", "广东": "GUANGDONG",
                "佛山": "FOSHAN", "东莞": "DONGGUAN", "深圳": "SHENZHEN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 20))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_relationships] 搜索词: {search_terms}")
        
        for _, row in self._relationships_df.iterrows():
            source = str(row.get("source", row.get("source_title", "")))
            target = str(row.get("target", row.get("target_title", "")))
            desc = str(row.get("description", ""))
            
            search_text = f"{source} {target} {desc}".upper()
            
            # 过滤不相关关键词
            if any(kw.upper() in search_text for kw in UNRELATED_KEYWORDS):
                continue
            
            # 过滤不相关地点
            if any(loc.upper() in search_text for loc in UNRELATED_LOCATIONS):
                continue
            
            score = 0
            year_matched = False
            exact_year_matched = False  # 是否匹配目标年份
            location_matched = False
            topic_matched = False
            
            for term, weight in search_terms:
                term_upper = str(term).upper()
                if term_upper in search_text:
                    score += weight
                    if str(term) == str(context.year):
                        year_matched = True
                        exact_year_matched = True  # 精确匹配目标年份
                    elif str(term) == str(context.location):
                        location_matched = True
                    elif term in topic_keywords:
                        topic_matched = True
            
            # 过滤策略：必须匹配年份
            if not year_matched:
                continue
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 15
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 25
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append({
                "id": row.get("id", ""),
                "source": source,
                "target": target,
                "description": desc,
                "weight": row.get("weight", 1),
                "text_unit_ids": row.get("text_unit_ids", []),
                "score": score,
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 去重：移除重复的关系
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['source']}_{r['target']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        print(f"[_search_relationships] 匹配结果数: {len(results)}, 去重后: {len(unique_results)}")
        return unique_results[:min(top_k, 8)]  # 限制最多返回8个
    
    def _search_text_units(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[str]:
        """搜索相关文本单元（改进版：加权评分+主题相关性过滤）"""
        if self._text_units_df is None or self._text_units_df.empty:
            return []
        
        # 需要过滤的不相关文本关键词
        UNRELATED_TEXT_KEYWORDS = {
            '嫦娥', '探月', '月球', 'BRICS', '金砖',
            '维和', '平潭', '滨海新区', '里斯本', '欧盟',
            'HU JINTAO', 'JIANG ZEMIN', 'DENG XIAOPING',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高（提高权重）
        if context.location:
            search_terms.append((context.location, 35))  # 提高到35
            location_map = {
                "广州": "GUANGZHOU", "广东": "GUANGDONG",
                "佛山": "FOSHAN", "东莞": "DONGGUAN", "深圳": "SHENZHEN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 35))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_text_units] 搜索词: {search_terms}")
        
        for _, row in self._text_units_df.iterrows():
            text = str(row.get("text", row.get("content", "")))
            
            # 过滤过短的文本
            if len(text) < 50:
                continue
            
            # 过滤不相关文本
            if any(keyword in text for keyword in UNRELATED_TEXT_KEYWORDS):
                continue
            
            # 年份匹配：允许目标年份及邻近年份
            year_matched = False
            exact_year_matched = False
            found_year = None
            if context.year:
                target_year = int(context.year)
                allowed_years = {str(target_year - 1), str(target_year), str(target_year + 1)}
                
                first_part = text[:300]
                
                for year in allowed_years:
                    year_pattern = rf'{year}[\s年]'
                    match = re.search(year_pattern, first_part)
                    if match:
                        match_str = match.group(0)
                        # 简化检查：匹配到年份数字后面跟空格或"年"就认为是有效的年份引用
                        # 检查是否是完整的年份（避免误匹配如"20090"这样的数字）
                        if len(match_str) >= 5:
                            year_matched = True
                            found_year = year
                            if year == str(target_year):
                                exact_year_matched = True
                            break
                
                if not year_matched:
                    continue
            
            # 地点匹配检查（硬性要求：必须包含目标地点或邻近城市/地区）
            location_matched = False
            if context.location:
                # 直接匹配目标地点
                if context.location in text:
                    location_matched = True
                else:
                    # 检查邻近城市
                    nearby_cities = ["深圳", "东莞", "佛山", "珠海", "中山"]
                    for city in nearby_cities:
                        if city in text:
                            location_matched = True
                            break
                    # 检查省份和区域
                    if not location_matched:
                        regions = ["广东", "珠三角", "珠江三角洲", "粤港澳"]
                        for region in regions:
                            if region in text:
                                location_matched = True
                                break
            
            # 硬性要求：必须同时匹配年份和地点
            if not (year_matched and location_matched):
                continue
            
            score = 0
            topic_matched = False
            
            for term, weight in search_terms:
                if term in text:
                    score += weight
                    if term in topic_keywords:
                        topic_matched = True
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 30  # 提高加分
            
            # 额外加分：直接匹配目标地点（而不是邻近城市）
            if context.location and context.location in text:
                score += 20  # 直接匹配广州加20分
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 30
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append((text, score, exact_year_matched, location_matched, topic_matched))
        
        # 排序策略：首先按是否精确匹配目标年份排序，然后按分数降序
        # 这样确保2009年的内容优先于2010年的内容
        results.sort(key=lambda x: (x[2], x[1], x[3] and x[4], x[3], x[4]), reverse=True)
        
        print(f"[_search_text_units] 匹配结果数: {len(results)}")
        
        # 确保同时包含目标年份和邻近年份的内容
        # 先取前5个2009年的，再取3个2010年的
        year_2009 = [r for r in results if r[2]]  # 精确匹配2009年
        year_other = [r for r in results if not r[2]]  # 2008或2010年
        
        # 混合结果：优先2009年，但也要包含一些2010年的内容
        mixed_results = year_2009[:5] + year_other[:3]
        
        return [r[0] for r in mixed_results[:min(top_k, 8)]]  # 限制最多返回8个
    
    def _search_communities(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关社区报告"""
        if self._communities_df is None or self._communities_df.empty:
            return []
        
        results = []
        search_terms = [context.year, context.location] + context.keywords
        search_terms = [t for t in search_terms if t]
        
        for _, row in self._communities_df.iterrows():
            title = str(row.get("title", ""))
            summary = str(row.get("summary", ""))
            
            score = 0
            for term in search_terms:
                if str(term) in title or str(term) in summary:
                    score += 1
            
            if score > 0:
                results.append({
                    "title": title,
                    "summary": summary,
                    "score": score,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:min(top_k, 3)]  # 限制最多返回3个
