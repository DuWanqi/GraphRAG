"""
Microsoft GraphRAG 官方 API 检索器
使用 graphrag.api.local_search 进行检索
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field

import pandas as pd
import yaml

from .memoir_parser import MemoirContext


@dataclass
class OfficialRetrievalResult:
    """官方 GraphRAG 检索结果"""
    response: str = ""  # LLM 生成的回答
    context_data: Dict[str, Any] = field(default_factory=dict)  # 上下文数据
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    text_units: List[str] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    context: Optional[MemoirContext] = None
    query: str = ""


class OfficialGraphRAGRetriever:
    """
    使用 Microsoft GraphRAG 官方 API 的检索器
    调用 graphrag.api.local_search 进行检索
    """
    
    def __init__(self, index_dir: Optional[Union[str, Path]] = None):
        default_root = Path(__file__).resolve().parent.parent.parent / "data" / "graphrag_output"
        self.index_dir = Path(index_dir) if index_dir is not None else default_root
        
        # 加载配置
        self.config = None
        self._load_config()
        
        # 加载数据
        self._entities_df = None
        self._relationships_df = None
        self._text_units_df = None
        self._communities_df = None
        self._community_reports_df = None
        self._covariates_df = None
        self._load_data()
    
    def _load_config(self):
        """加载 GraphRAG 配置"""
        settings_path = self.index_dir / "settings.yaml"
        if not settings_path.exists():
            print(f"[OfficialGraphRAGRetriever] 警告: 配置文件不存在 {settings_path}")
            return
        
        try:
            from graphrag.config.models.graph_rag_config import GraphRagConfig
            self.config = GraphRagConfig.from_yaml(str(settings_path))
            print(f"[OfficialGraphRAGRetriever] 配置加载成功")
        except Exception as e:
            print(f"[OfficialGraphRAGRetriever] 加载配置失败: {e}")
    
    def _load_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        # 兼容 GraphRAG 1.x 和 2.x
        file_mappings = {
            "entities": ["entities.parquet", "create_final_entities.parquet"],
            "relationships": ["relationships.parquet", "create_final_relationships.parquet"],
            "text_units": ["text_units.parquet", "create_final_text_units.parquet"],
            "communities": ["communities.parquet", "create_final_communities.parquet"],
            "community_reports": ["community_reports.parquet", "create_final_community_reports.parquet"],
            "covariates": ["covariates.parquet", "create_final_covariates.parquet"],
        }
        
        def load_first_existing(file_list):
            for file_name in file_list:
                file_path = output_dir / file_name
                if file_path.exists():
                    try:
                        return pd.read_parquet(file_path)
                    except Exception as e:
                        print(f"[OfficialGraphRAGRetriever] 加载 {file_name} 失败: {e}")
            return None
        
        self._entities_df = load_first_existing(file_mappings["entities"])
        self._relationships_df = load_first_existing(file_mappings["relationships"])
        self._text_units_df = load_first_existing(file_mappings["text_units"])
        self._communities_df = load_first_existing(file_mappings["communities"])
        self._community_reports_df = load_first_existing(file_mappings["community_reports"])
        self._covariates_df = load_first_existing(file_mappings["covariates"])
        
        print(f"[OfficialGraphRAGRetriever] 加载实体: {len(self._entities_df) if self._entities_df is not None else 0} 条")
        print(f"[OfficialGraphRAGRetriever] 加载关系: {len(self._relationships_df) if self._relationships_df is not None else 0} 条")
        print(f"[OfficialGraphRAGRetriever] 加载文本单元: {len(self._text_units_df) if self._text_units_df is not None else 0} 条")
        print(f"[OfficialGraphRAGRetriever] 加载社区报告: {len(self._community_reports_df) if self._community_reports_df is not None else 0} 条")
    
    def is_ready(self) -> bool:
        """检查是否就绪"""
        return (
            self.config is not None and
            self._entities_df is not None and
            self._text_units_df is not None
        )
    
    async def search(
        self,
        query: str,
        community_level: int = 2,
        response_type: str = "Multiple Paragraphs",
        verbose: bool = False
    ) -> OfficialRetrievalResult:
        """
        使用官方 local_search API 进行检索
        
        Args:
            query: 查询文本
            community_level: 社区层级（默认2）
            response_type: 响应类型
            verbose: 是否输出详细信息
            
        Returns:
            OfficialRetrievalResult 检索结果
        """
        from graphrag.api import local_search
        
        result = OfficialRetrievalResult()
        result.query = query
        
        if not self.is_ready():
            print("[OfficialGraphRAGRetriever] 未就绪，无法执行检索")
            return result
        
        try:
            print(f"[OfficialGraphRAGRetriever] 执行 local_search: {query[:50]}...")
            
            # 调用官方 API
            response, context_data = await local_search(
                config=self.config,
                entities=self._entities_df,
                communities=self._communities_df if self._communities_df is not None else pd.DataFrame(),
                community_reports=self._community_reports_df if self._community_reports_df is not None else pd.DataFrame(),
                text_units=self._text_units_df,
                relationships=self._relationships_df if self._relationships_df is not None else pd.DataFrame(),
                covariates=self._covariates_df,
                community_level=community_level,
                response_type=response_type,
                query=query,
                verbose=verbose
            )
            
            result.response = response if isinstance(response, str) else str(response)
            result.context_data = context_data if isinstance(context_data, dict) else {}
            
            # 从 context_data 中提取实体、关系等信息
            if isinstance(context_data, dict):
                # 提取实体
                if "entities" in context_data and isinstance(context_data["entities"], pd.DataFrame):
                    for _, row in context_data["entities"].head(10).iterrows():
                        result.entities.append({
                            "name": row.get("title", row.get("name", "")),
                            "description": row.get("description", ""),
                            "type": row.get("type", "unknown"),
                            "source": "official_local_search"
                        })
                
                # 提取关系
                if "relationships" in context_data and isinstance(context_data["relationships"], pd.DataFrame):
                    for _, row in context_data["relationships"].head(10).iterrows():
                        result.relationships.append({
                            "source": row.get("source", ""),
                            "target": row.get("target", ""),
                            "description": row.get("description", ""),
                            "source": "official_local_search"
                        })
                
                # 提取文本单元
                if "text_units" in context_data and isinstance(context_data["text_units"], pd.DataFrame):
                    for _, row in context_data["text_units"].head(10).iterrows():
                        text = row.get("text", "")
                        if text:
                            result.text_units.append(text)
                
                # 提取社区报告
                if "community_reports" in context_data and isinstance(context_data["community_reports"], pd.DataFrame):
                    for _, row in context_data["community_reports"].head(5).iterrows():
                        result.communities.append({
                            "title": row.get("title", ""),
                            "summary": row.get("summary", ""),
                            "full_content": row.get("full_content", "")[:1000],
                            "source": "official_local_search"
                        })
            
            print(f"[OfficialGraphRAGRetriever] 检索完成: {len(result.entities)} 实体, {len(result.relationships)} 关系, {len(result.text_units)} 文本")
            
        except Exception as e:
            print(f"[OfficialGraphRAGRetriever] 检索失败: {e}")
            import traceback
            traceback.print_exc()
        
        return result
