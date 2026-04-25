"""
回忆录历史背景检索器
基于GraphRAG进行知识图谱检索
支持关键词检索、向量检索、混合检索三种模式
"""

import os
import asyncio
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .memoir_parser import MemoirParser, MemoirContext
from .vector_retriever import VectorRetriever, RetrievalMode
from ..config import get_settings
from ..llm import LLMAdapter, create_llm_adapter


@dataclass
class RetrievalResult:
    """检索结果"""
    query: str
    context: MemoirContext
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    text_units: List[str] = field(default_factory=list)
    
    @property
    def has_results(self) -> bool:
        """是否有检索结果"""
        return bool(self.entities or self.text_units)
    
    def get_context_text(self) -> str:
        """获取上下文文本（用于LLM生成）"""
        parts = []
        
        if self.entities:
            parts.append("## 相关历史实体")
            for entity in self.entities[:10]:
                name = entity.get("name", entity.get("title", "未知"))
                desc = entity.get("description", "")
                parts.append(f"- {name}: {desc[:200]}")
        
        if self.communities:
            parts.append("\n## 相关历史背景")
            for comm in self.communities[:3]:
                summary = comm.get("summary", comm.get("full_content", ""))
                parts.append(summary[:500])
        
        if self.text_units:
            parts.append("\n## 相关历史文本")
            for text in self.text_units[:5]:
                parts.append(text[:300])
        
        return "\n".join(parts)


class MemoirRetriever:
    """
    回忆录历史背景检索器
    
    功能：
    1. 解析回忆录文本，提取查询上下文
    2. 在知识图谱中进行本地检索（实体、关系）
    3. 进行全局检索（社区报告）
    4. 返回相关的历史背景信息
    
    检索模式：
    - keyword: 关键词匹配（默认，快速）
    - vector: 向量相似度检索（需要embedding）
    - hybrid: 混合检索（关键词+向量融合）
    """
    
    def __init__(
        self,
        index_dir: Optional[str] = None,
        llm_adapter: Optional[LLMAdapter] = None,
    ):
        """
        初始化检索器
        
        Args:
            index_dir: GraphRAG索引目录
            llm_adapter: LLM适配器
        """
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.graphrag_output_dir)
        self.llm_adapter = llm_adapter
        self.parser = MemoirParser(llm_adapter)
        self.vector_retriever = VectorRetriever(index_dir=str(self.index_dir), llm_adapter=llm_adapter)
        
        # 缓存加载的数据
        self._entities_df = None
        self._relationships_df = None
        self._communities_df = None
        self._text_units_df = None
    
    def _load_index_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        # GraphRAG 2.x 使用新的文件名格式
        # 尝试新格式，如果不存在则尝试旧格式（兼容 1.x）
        
        # 加载实体
        entities_file = output_dir / "entities.parquet"
        if not entities_file.exists():
            entities_file = output_dir / "create_final_entities.parquet"
        if entities_file.exists():
            self._entities_df = pd.read_parquet(entities_file)
            print(f"[DEBUG] 加载实体: {len(self._entities_df)} 条")
        
        # 加载关系
        rel_file = output_dir / "relationships.parquet"
        if not rel_file.exists():
            rel_file = output_dir / "create_final_relationships.parquet"
        if rel_file.exists():
            self._relationships_df = pd.read_parquet(rel_file)
            print(f"[DEBUG] 加载关系: {len(self._relationships_df)} 条")
        
        # 加载社区报告
        comm_file = output_dir / "community_reports.parquet"
        if not comm_file.exists():
            comm_file = output_dir / "create_final_community_reports.parquet"
        if comm_file.exists():
            self._communities_df = pd.read_parquet(comm_file)
            print(f"[DEBUG] 加载社区报告: {len(self._communities_df)} 条")
        
        # 加载文本单元
        text_file = output_dir / "text_units.parquet"
        if not text_file.exists():
            text_file = output_dir / "create_final_text_units.parquet"
        if text_file.exists():
            self._text_units_df = pd.read_parquet(text_file)
            print(f"[DEBUG] 加载文本单元: {len(self._text_units_df)} 条")
    
    async def retrieve(
        self,
        memoir_text: str,
        top_k: int = 10,
        use_llm_parsing: bool = True,
        mode: str = "keyword",
    ) -> RetrievalResult:
        """
        检索与回忆录相关的历史背景
        
        Args:
            memoir_text: 回忆录文本
            top_k: 返回结果数量
            use_llm_parsing: 是否使用LLM解析回忆录
            mode: 检索模式 (keyword/vector/hybrid)
            
        Returns:
            RetrievalResult: 检索结果
        """
        timing = os.getenv("TEMP_TIMING") == "1"
        t0 = time.perf_counter()

        # 解析回忆录
        t_parse0 = time.perf_counter()
        if use_llm_parsing and self.llm_adapter:
            context = await self.parser.parse_with_llm(memoir_text)
        else:
            context = self.parser.parse(memoir_text, use_llm=False)
        t_parse = time.perf_counter() - t_parse0
        if timing:
            print(f"[TEMP_TIMING] retriever.parse={t_parse:.3f}s use_llm_parsing={use_llm_parsing} mode={mode}")
        
        # 生成查询
        query = context.to_query()
        
        # 确保索引数据已加载
        t_load0 = time.perf_counter()
        if self._entities_df is None:
            self._load_index_data()
        t_load = time.perf_counter() - t_load0
        if timing:
            print(f"[TEMP_TIMING] retriever.load_index={t_load:.3f}s")
        
        # 执行检索
        result = RetrievalResult(query=query, context=context)
        
        # 根据模式选择检索策略
        if mode == "vector" and self.vector_retriever.is_ready():
            # 纯向量检索（实体、社区、文本单元用向量，关系用关键词）
            t_v0 = time.perf_counter()
            result.entities = await self.vector_retriever.search_entities(query, top_k)
            result.communities = await self.vector_retriever.search_communities(query, top_k // 2)
            result.text_units = await self.vector_retriever.search_text_units(query, top_k)
            # 关系检索使用关键词（向量索引中没有关系）
            if self._relationships_df is not None:
                result.relationships = self._search_relationships(context, top_k)
            if timing:
                print(f"[TEMP_TIMING] retriever.vector_search={time.perf_counter()-t_v0:.3f}s")
            
        elif mode == "hybrid" and self.vector_retriever.is_ready():
            # 混合检索：关键词 + 向量融合
            t_h0 = time.perf_counter()
            keyword_entities = self._search_entities(context, top_k)
            vector_entities = await self.vector_retriever.search_entities(query, top_k)
            result.entities = self._merge_results(keyword_entities, vector_entities, top_k)
            
            keyword_communities = self._search_communities(context, top_k // 2)
            vector_communities = await self.vector_retriever.search_communities(query, top_k // 2)
            result.communities = self._merge_results(keyword_communities, vector_communities, top_k // 2)
            
            keyword_texts = self._search_text_units(context, top_k)
            vector_texts = await self.vector_retriever.search_text_units(query, top_k)
            result.text_units = self._merge_text_results(keyword_texts, vector_texts, top_k)
            if timing:
                print(f"[TEMP_TIMING] retriever.hybrid_search={time.perf_counter()-t_h0:.3f}s")
            
        else:
            # 关键词检索（默认）
            t_k0 = time.perf_counter()
            if self._entities_df is not None:
                result.entities = self._search_entities(context, top_k)
            if self._relationships_df is not None:
                result.relationships = self._search_relationships(context, top_k)
            if self._communities_df is not None:
                result.communities = self._search_communities(context, top_k // 2)
            if self._text_units_df is not None:
                result.text_units = self._search_text_units(context, top_k)
            if timing:
                print(f"[TEMP_TIMING] retriever.keyword_search={time.perf_counter()-t_k0:.3f}s")
        
        if timing:
            total = time.perf_counter() - t0
            print(
                f"[TEMP_TIMING] retriever.total={total:.3f}s "
                f"entities={len(result.entities)} rel={len(result.relationships)} comm={len(result.communities)} texts={len(result.text_units)}"
            )
        return result
    
    def _merge_results(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """融合关键词和向量检索结果"""
        merged = {}
        
        for item in keyword_results:
            name = item.get("name", item.get("title", ""))
            if name:
                merged[name] = item.copy()
                merged[name]["score"] = item.get("score", 1) * 0.5
                merged[name]["source"] = "keyword"
        
        for item in vector_results:
            name = item.get("name", item.get("title", ""))
            if name:
                if name in merged:
                    merged[name]["score"] += item.get("score", 1) * 0.5
                    merged[name]["source"] = "hybrid"
                else:
                    merged[name] = item.copy()
                    merged[name]["score"] = item.get("score", 1) * 0.5
                    merged[name]["source"] = "vector"
        
        sorted_results = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:top_k]
    
    def _merge_text_results(
        self,
        keyword_texts: List[str],
        vector_texts: List[str],
        top_k: int,
    ) -> List[str]:
        """融合文本结果"""
        seen = set()
        merged = []
        
        for text in keyword_texts:
            if text not in seen:
                seen.add(text)
                merged.append(text)
        
        for text in vector_texts:
            if text not in seen:
                seen.add(text)
                merged.append(text)
        
        return merged[:top_k]
    
    def retrieve_sync(
        self,
        memoir_text: str,
        top_k: int = 10,
    ) -> RetrievalResult:
        """同步版本的检索"""
        return asyncio.run(self.retrieve(memoir_text, top_k, use_llm_parsing=False))
    
    def _search_entities(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关实体"""
        if self._entities_df is None or self._entities_df.empty:
            return []
        
        results = []
        
        # 构建搜索词（包含中英文变体）
        search_terms = []
        if context.year:
            search_terms.append(context.year)
            search_terms.append(str(context.year))
        if context.location:
            search_terms.append(context.location)
            # 添加英文变体
            location_map = {
                "深圳": "SHENZHEN", "北京": "BEIJING", "上海": "SHANGHAI",
                "广州": "GUANGZHOU", "香港": "HONG KONG"
            }
            if context.location in location_map:
                search_terms.append(location_map[context.location])
        search_terms.extend(context.keywords)
        
        for _, row in self._entities_df.iterrows():
            # GraphRAG 2.x 使用 title 字段
            entity_name = str(row.get("title", row.get("name", "")))
            entity_desc = str(row.get("description", ""))
            
            # 计算匹配分数（不区分大小写）
            score = 0
            search_text = f"{entity_name} {entity_desc}".upper()
            for term in search_terms:
                if term and term.upper() in search_text:
                    score += 1
            
            if score > 0:
                results.append({
                    "name": entity_name,
                    "type": row.get("type", "unknown"),
                    "description": entity_desc,
                    "score": score,
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _search_relationships(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关关系"""
        if self._relationships_df is None or self._relationships_df.empty:
            return []
        
        results = []
        search_terms = [context.year, context.location] + context.keywords
        search_terms = [t for t in search_terms if t]
        
        for _, row in self._relationships_df.iterrows():
            source = str(row.get("source", ""))
            target = str(row.get("target", ""))
            desc = str(row.get("description", ""))
            
            score = 0
            for term in search_terms:
                if term in source or term in target or term in desc:
                    score += 1
            
            if score > 0:
                results.append({
                    "source": source,
                    "target": target,
                    "description": desc,
                    "weight": row.get("weight", 1),
                    "score": score,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
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
            content = str(row.get("full_content", ""))
            
            score = 0
            text = f"{title} {summary} {content}"
            for term in search_terms:
                if term in text:
                    score += 1
            
            if score > 0:
                results.append({
                    "title": title,
                    "summary": summary,
                    "full_content": content[:1000],
                    "level": row.get("level", 0),
                    "score": score,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _search_text_units(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[str]:
        """搜索相关文本单元"""
        if self._text_units_df is None or self._text_units_df.empty:
            return []
        
        results = []
        search_terms = [context.year, context.location] + context.keywords
        search_terms = [t for t in search_terms if t]
        
        for _, row in self._text_units_df.iterrows():
            text = str(row.get("text", ""))
            
            score = 0
            for term in search_terms:
                if term in text:
                    score += 1
            
            if score > 0:
                results.append((text, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:top_k]]
    
    def is_index_ready(self) -> bool:
        """检查索引是否就绪"""
        output_dir = self.index_dir / "output"
        # 兼容 GraphRAG 1.x 和 2.x
        new_format = output_dir / "entities.parquet"
        old_format = output_dir / "create_final_entities.parquet"
        return new_format.exists() or old_format.exists()
