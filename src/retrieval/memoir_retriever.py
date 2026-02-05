"""
回忆录历史背景检索器
基于GraphRAG进行知识图谱检索
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd

from .memoir_parser import MemoirParser, MemoirContext
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
        
        # 缓存加载的数据
        self._entities_df = None
        self._relationships_df = None
        self._communities_df = None
        self._text_units_df = None
    
    def _load_index_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        # 加载实体
        entities_file = output_dir / "create_final_entities.parquet"
        if entities_file.exists():
            self._entities_df = pd.read_parquet(entities_file)
        
        # 加载关系
        rel_file = output_dir / "create_final_relationships.parquet"
        if rel_file.exists():
            self._relationships_df = pd.read_parquet(rel_file)
        
        # 加载社区报告
        comm_file = output_dir / "create_final_community_reports.parquet"
        if comm_file.exists():
            self._communities_df = pd.read_parquet(comm_file)
        
        # 加载文本单元
        text_file = output_dir / "create_final_text_units.parquet"
        if text_file.exists():
            self._text_units_df = pd.read_parquet(text_file)
    
    async def retrieve(
        self,
        memoir_text: str,
        top_k: int = 10,
        use_llm_parsing: bool = True,
    ) -> RetrievalResult:
        """
        检索与回忆录相关的历史背景
        
        Args:
            memoir_text: 回忆录文本
            top_k: 返回结果数量
            use_llm_parsing: 是否使用LLM解析回忆录
            
        Returns:
            RetrievalResult: 检索结果
        """
        # 解析回忆录
        if use_llm_parsing and self.llm_adapter:
            context = await self.parser.parse_with_llm(memoir_text)
        else:
            context = self.parser.parse(memoir_text, use_llm=False)
        
        # 生成查询
        query = context.to_query()
        
        # 确保索引数据已加载
        if self._entities_df is None:
            self._load_index_data()
        
        # 执行检索
        result = RetrievalResult(query=query, context=context)
        
        # 本地检索：实体匹配
        if self._entities_df is not None:
            result.entities = self._search_entities(context, top_k)
        
        # 本地检索：关系匹配
        if self._relationships_df is not None:
            result.relationships = self._search_relationships(context, top_k)
        
        # 全局检索：社区报告
        if self._communities_df is not None:
            result.communities = self._search_communities(context, top_k // 2)
        
        # 文本单元检索
        if self._text_units_df is not None:
            result.text_units = self._search_text_units(context, top_k)
        
        return result
    
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
        
        # 构建搜索词
        search_terms = []
        if context.year:
            search_terms.append(context.year)
        if context.location:
            search_terms.append(context.location)
        search_terms.extend(context.keywords)
        
        for _, row in self._entities_df.iterrows():
            entity_name = str(row.get("name", row.get("title", "")))
            entity_desc = str(row.get("description", ""))
            
            # 计算匹配分数
            score = 0
            for term in search_terms:
                if term and (term in entity_name or term in entity_desc):
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
        entities_file = output_dir / "create_final_entities.parquet"
        return entities_file.exists()
