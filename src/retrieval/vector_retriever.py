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


class EmbeddingError(Exception):
    """Embedding 获取失败异常"""
    def __init__(self, message: str, is_ollama_error: bool = False):
        self.message = message
        self.is_ollama_error = is_ollama_error
        super().__init__(self.message)


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
        
        # 加载实体和关系数据（用于关联向量检索结果）
        self._entities_df = None
        self._relationships_df = None
        self._text_units_df = None
        self._communities_df = None
        self._load_graph_data()
        
        # 使用Ollama embedding
        if use_ollama_embedding:
            self._embedding = OllamaEmbedding()
        else:
            self._embedding = None
    
    def _load_graph_data(self):
        """加载图谱数据（实体、关系等）用于关联向量检索结果"""
        import pandas as pd
        
        output_dir = self.index_dir / "output"
        if not output_dir.exists():
            return
        
        # 加载实体数据
        entities_path = output_dir / "entities.parquet"
        if entities_path.exists():
            try:
                self._entities_df = pd.read_parquet(entities_path)
                print(f"[VectorRetriever] 加载 {len(self._entities_df)} 个实体")
            except Exception as e:
                print(f"[VectorRetriever] 加载实体数据失败: {e}")
        
        # 加载关系数据
        relationships_path = output_dir / "relationships.parquet"
        if relationships_path.exists():
            try:
                self._relationships_df = pd.read_parquet(relationships_path)
                print(f"[VectorRetriever] 加载 {len(self._relationships_df)} 个关系")
            except Exception as e:
                print(f"[VectorRetriever] 加载关系数据失败: {e}")

        text_units_path = output_dir / "text_units.parquet"
        if text_units_path.exists():
            try:
                self._text_units_df = pd.read_parquet(text_units_path)
                print(f"[VectorRetriever] 加载 {len(self._text_units_df)} 个文本单元")
            except Exception as e:
                print(f"[VectorRetriever] 加载文本单元失败: {e}")

        community_paths = [
            output_dir / "community_reports.parquet",
            output_dir / "communities.parquet",
        ]
        for communities_path in community_paths:
            if communities_path.exists():
                try:
                    self._communities_df = pd.read_parquet(communities_path)
                    print(f"[VectorRetriever] 加载 {len(self._communities_df)} 个社区")
                    break
                except Exception as e:
                    print(f"[VectorRetriever] 加载社区数据失败: {e}")
    
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
            error_msg = str(e)
            print(f"[VectorRetriever] 获取embedding失败: {e}")
            # 检测是否是 Ollama 连接错误
            if "Cannot connect to host" in error_msg or "远程计算机拒绝网络连接" in error_msg:
                raise EmbeddingError(
                    "Ollama 服务未运行，请启动 Ollama 服务后再试。\n"
                    "启动命令: ollama serve",
                    is_ollama_error=True
                )
            raise EmbeddingError(f"获取 embedding 失败: {e}", is_ollama_error=False)
    
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
            print(f"[VectorRetriever] 向量检索返回 {len(results)} 个实体")
            
            entities = []
            for _, row in results.iterrows():
                entity_id = row.get("id", "")
                score = float(row.get("_distance", 1.0))
                
                # 通过ID关联实体数据
                entity_name = "未知实体"
                entity_desc = ""
                entity_type = "unknown"
                
                if self._entities_df is not None and entity_id:
                    entity_row = self._entities_df[self._entities_df['id'] == entity_id]
                    if not entity_row.empty:
                        entity_name = entity_row.iloc[0].get('title', '未知实体')
                        entity_desc = entity_row.iloc[0].get('description', '')
                        entity_type = entity_row.iloc[0].get('type', 'unknown')
                    else:
                        print(f"[VectorRetriever] 警告: 找不到ID为 {entity_id[:20]}... 的实体")
                else:
                    if self._entities_df is None:
                        print(f"[VectorRetriever] 警告: 实体数据未加载")
                    if not entity_id:
                        print(f"[VectorRetriever] 警告: 实体ID为空")
                
                entities.append({
                    "name": entity_name,
                    "description": entity_desc,
                    "type": entity_type,
                    "score": score,
                    "source": "vector",
                    "id": entity_id,
                })
            
            print(f"[VectorRetriever] 成功解析 {len(entities)} 个实体")
            return entities
            
        except Exception as e:
            print(f"[VectorRetriever] 实体检索失败: {e}")
            import traceback
            print(traceback.format_exc())
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
        
        records = await self.search_text_unit_records(query, top_k=top_k)
        return [r["text"] for r in records if r.get("text")]

    async def search_text_unit_records(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """向量检索文本单元，并保留 GraphRAG text_unit id 与距离。"""
        self._connect()

        if self._text_unit_table is None:
            return []

        embedding = await self._get_query_embedding(query)
        if embedding is None:
            return []

        try:
            query_vector = np.array(embedding, dtype=np.float32)
            results = self._text_unit_table.search(query_vector).limit(top_k).to_pandas()

            records = []
            for _, row in results.iterrows():
                text_id = row.get("id", "")
                distance = float(row.get("_distance", 1.0))
                text = row.get("text", "")
                metadata: Dict[str, Any] = {}

                if (not text) and self._text_units_df is not None and text_id:
                    text_row = self._text_units_df[self._text_units_df["id"] == text_id]
                    if not text_row.empty:
                        text = text_row.iloc[0].get("text", "")
                        metadata = text_row.iloc[0].to_dict()
                elif self._text_units_df is not None and text_id:
                    text_row = self._text_units_df[self._text_units_df["id"] == text_id]
                    if not text_row.empty:
                        metadata = text_row.iloc[0].to_dict()

                if text:
                    records.append({
                        "id": text_id,
                        "text": text,
                        "score": distance,
                        "distance": distance,
                        "source": "vector",
                        "metadata": metadata,
                    })

            return records

        except Exception as e:
            print(f"[VectorRetriever] 文本检索失败: {e}")
            return []
