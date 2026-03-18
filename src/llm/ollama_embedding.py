"""
Ollama Embedding封装
使用Ollama API获取文本向量
"""

import asyncio
import aiohttp
import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class OllamaEmbeddingConfig:
    """Ollama Embedding配置"""
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout: int = 60


class OllamaEmbedding:
    """
    Ollama Embedding封装类
    
    通过Ollama API调用本地embedding模型
    - 支持nomic-embed-text等模型
    - 完全本地化，无外部API依赖
    """
    
    def __init__(self, config: Optional[OllamaEmbeddingConfig] = None):
        """
        初始化Ollama Embedding
        
        Args:
            config: 配置对象，默认使用nomic-embed-text模型
        """
        self.config = config or OllamaEmbeddingConfig()
        # 允许用环境变量覆盖（与 LLM 侧保持一致）
        self.config.base_url = os.getenv("OLLAMA_API_BASE", self.config.base_url)
        self._dimension = None
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        if self._dimension is None:
            # nomic-embed-text的维度是768
            self._dimension = 768
        return self._dimension
    
    async def embed(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示（异步）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        url = f"{self.config.base_url}/api/embeddings"
        
        payload = {
            "model": self.config.model,
            "prompt": text
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API错误: {response.status} - {error_text}")
                
                result = await response.json()
                return result["embedding"]
    
    def embed_sync(self, text: str) -> List[float]:
        """
        获取单个文本的向量表示（同步）
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        return asyncio.run(self.embed(text))
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量表示（异步）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量表示（同步）
        
        Args:
            texts: 文本列表
            
        Returns:
            List[List[float]]: 向量列表
        """
        return asyncio.run(self.embed_batch(texts))
    
    async def is_available(self) -> bool:
        """检查Ollama服务是否可用"""
        try:
            url = f"{self.config.base_url}/api/tags"
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    return response.status == 200
        except Exception:
            return False


def get_ollama_embedding(model: str = "nomic-embed-text") -> OllamaEmbedding:
    """获取Ollama embedding实例"""
    config = OllamaEmbeddingConfig(model=model)
    return OllamaEmbedding(config)


if __name__ == "__main__":
    import asyncio
    
    async def test():
        embedding = OllamaEmbedding()
        
        # 检查服务是否可用
        available = await embedding.is_available()
        print(f"Ollama服务可用: {available}")
        
        if available:
            # 测试embedding
            test_text = "深圳经济特区成立"
            vector = await embedding.embed(test_text)
            print(f"文本: {test_text}")
            print(f"向量维度: {len(vector)}")
            print(f"向量前5维: {vector[:5]}")
    
    asyncio.run(test())
