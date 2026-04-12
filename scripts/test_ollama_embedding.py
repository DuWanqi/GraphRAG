"""测试Ollama Embedding"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from src.llm import OllamaEmbedding

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
