"""
本地Embedding模型封装
使用 sentence-transformers 加载 BGE 中文模型
"""

import asyncio
from typing import List, Optional
from pathlib import Path
import os


class LocalEmbedding:
    """
    本地Embedding模型封装类
    
    使用 BAAI/bge-base-zh-v1.5 模型（768维，与GraphRAG索引一致）
    - 完全本地化，无API依赖
    - 中文效果优秀
    - 向量维度: 768
    """
    
    MODEL_NAME = "BAAI/bge-base-zh-v1.5"
    CACHE_DIR = Path.home() / ".cache" / "huggingface"
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """单例模式，避免重复加载模型"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[LocalEmbedding] 加载模型: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
            print(f"[LocalEmbedding] 模型加载成功，向量维度: {self._model.get_sentence_embedding_dimension()}")
        except ImportError:
            raise ImportError(
                "请安装 sentence-transformers: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._model.get_sentence_embedding_dimension()
    
    def embed(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示（同步）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量表示（同步）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]
    
    async def embed_async(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示（异步）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        return await asyncio.to_thread(self.embed, text)
    
    async def embed_batch_async(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量表示（异步）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        return await asyncio.to_thread(self.embed_batch, texts)


def get_local_embedding() -> LocalEmbedding:
    """获取本地embedding实例"""
    return LocalEmbedding()


if __name__ == "__main__":
    embedding = LocalEmbedding()
    test_text = "1988年深圳经济特区成立"
    vector = embedding.embed(test_text)
    print(f"文本: {test_text}")
    print(f"向量维度: {len(vector)}")
    print(f"向量前10维: {vector[:10]}")
