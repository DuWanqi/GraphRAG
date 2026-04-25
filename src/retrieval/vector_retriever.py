"""
向量检索模块
基于LanceDB实现向量相似度检索
支持Ollama本地embedding模型（nomic-embed-text）
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..config import get_settings
from ..llm import LLMAdapter
from ..llm.ollama_embedding import OllamaEmbedding, OllamaEmbeddingConfig


class RetrievalMode(Enum):
    """检索模式"""
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class VectorSearchResult:
    """向量检索结果"""
    id: str
    name: str
    description: str
    score: float
    vector: Optional[List[float]] = None


class VectorRetriever:
    """
    向量检索器
    
    使用LanceDB存储的向量索引进行相似度检索
    支持Ollama本地embedding模型（nomic-embed-text），无需外部API
    """
    
    def __init__(
        self,
        index_dir: Optional[str] = None,
        llm_adapter: Optional[LLMAdapter] = None,
        use_ollama_embedding: bool = True,
    ):
        """
        初始化向量检索器
        
        Args:
            index_dir: GraphRAG索引目录
            llm_adapter: LLM适配器（不再需要，保留兼容性）
            use_ollama_embedding: 是否使用Ollama embedding（默认True）
        """
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.graphrag_output_dir)
        self.llm_adapter = llm_adapter
        self.use_ollama_embedding = use_ollama_embedding
        
        self._lancedb = None
        self._entity_table = None
        self._community_table = None
        self._text_unit_table = None
        
        # 使用Ollama embedding
        if use_ollama_embedding:
            self._embedding = OllamaEmbedding()
        else:
            self._embedding = None
    
    def _connect(self):
        """连接LanceDB"""
        if self._lancedb is not None:
            return
        
        lancedb_path = self.index_dir / "output" / "lancedb"
        if not lancedb_path.exists():
            return
        
        try:
            import lancedb
            self._lancedb = lancedb.connect(str(lancedb_path))
            
            table_names = self._lancedb.table_names()
            
            for name in table_names:
                if "entity" in name.lower():
                    self._entity_table = self._lancedb.open_table(name)
                elif "community" in name.lower():
                    self._community_table = self._lancedb.open_table(name)
                elif "text" in name.lower():
                    self._text_unit_table = self._lancedb.open_table(name)
                
        except Exception as e:
            print(f"[VectorRetriever] 连接LanceDB失败: {e}")
    
    def is_ready(self) -> bool:
        """检查向量索引是否就绪"""
        self._connect()
        return self._entity_table is not None
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """获取查询的向量表示（使用Ollama）"""
        try:
            if self.use_ollama_embedding and self._embedding:
                return await self._embedding.embed(query)
            return None
        except Exception as e:
            print(f"[VectorRetriever] 获取embedding失败: {e}")
            return None
    
    async def search_entities(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        向量检索实体
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        self._connect()
        
        if self._entity_table is None:
            return []
        
        embedding = await self._get_query_embedding(query)
        if embedding is None:
            return []
        
        try:
            query_vector = np.array(embedding, dtype=np.float32)
            
            results = self._entity_table.search(query_vector).limit(top_k).to_pandas()
            
            entities = []
            for _, row in results.iterrows():
                # LanceDB返回的列名是text，包含实体名称和描述
                text = row.get("text", "")
                
                # 尝试多种方式解析实体名称
                name = ""
                description = text
                
                if ":" in text:
                    # 格式: "实体名称: 描述"
                    parts = text.split(":", 1)
                    name = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else text
                elif "\n" in text:
                    # 格式: "实体名称\n描述"
                    parts = text.split("\n", 1)
                    name = parts[0].strip()
                    description = parts[1].strip() if len(parts) > 1 else text
                else:
                    # 取前50个字符作为名称
                    name = text[:50].strip()
                
                # 如果名称为空，使用默认值
                if not name:
                    name = "未知实体"
                
                entities.append({
                    "name": name,
                    "description": description if description else text,
                    "type": row.get("type", "unknown"),
                    "score": float(row.get("_distance", 1.0)),
                    "source": "vector",
                })
            
            return entities
            
        except Exception as e:
            print(f"[VectorRetriever] 实体检索失败: {e}")
            return []
    
    async def search_communities(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        向量检索社区报告
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            检索结果列表
        """
        self._connect()
        
        if self._community_table is None:
            return []
        
        embedding = await self._get_query_embedding(query)
        if embedding is None:
            return []
        
        try:
            query_vector = np.array(embedding, dtype=np.float32)
            
            results = self._community_table.search(query_vector).limit(top_k).to_pandas()
            
            communities = []
            for _, row in results.iterrows():
                communities.append({
                    "title": row.get("title", ""),
                    "summary": row.get("summary", ""),
                    "full_content": row.get("full_content", "")[:1000],
                    "score": float(row.get("_distance", 0)),
                    "source": "vector",
                })
            
            return communities
            
        except Exception as e:
            print(f"[VectorRetriever] 社区检索失败: {e}")
            return []
    
    async def search_text_units(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[str]:
        """
        向量检索单元文本
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            文本列表
        """
        self._connect()
        
        if self._text_unit_table is None:
            return []
        
        embedding = await self._get_query_embedding(query)
        if embedding is None:
            return []
        
        try:
            query_vector = np.array(embedding, dtype=np.float32)
            
            results = self._text_unit_table.search(query_vector).limit(top_k).to_pandas()
            
            texts = []
            for _, row in results.iterrows():
                text = row.get("text", "")
                if text:
                    texts.append(text)
            
            return texts
            
        except Exception as e:
            print(f"[VectorRetriever] 文本检索失败: {e}")
            return []
