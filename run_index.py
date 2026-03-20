#!/usr/bin/env python3
"""
运行索引构建
使用GLM作为LLM，Ollama作为Embedding
"""

import sys
import os
from pathlib import Path

# 确保能导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

from src.indexing import GraphBuilder

def main():
    print("开始构建知识图谱索引...")
    print("LLM: GLM (智谱)")
    print("Embedding: Ollama (nomic-embed-text)")
    
    # 禁用Python版本警告
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # 使用GLM作为LLM，Ollama作为Embedding
    builder = GraphBuilder(
        llm_provider='glm',
        embedding_provider='ollama',
        embedding_model='nomic-embed-text'
    )
    result = builder.build_index_sync()
    
    if result.success:
        print("\n✅ 索引构建成功！")
        print(f"\n统计信息:")
        for key, value in result.stats.items():
            print(f"  {key}: {value}")
        print(f"\n输出目录: {result.output_dir}")
    else:
        print("\n❌ 索引构建失败：")
        print(f"  {result.message}")

if __name__ == "__main__":
    main()
