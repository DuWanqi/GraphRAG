#!/usr/bin/env python3
"""
使用本地Ollama模型重建知识图谱索引
"""

import os
import sys
from pathlib import Path

# 确保能导入本地模块
sys.path.insert(0, str(Path(__file__).parent.parent))

def rebuild_index_with_ollama():
    """使用Ollama本地模型重建索引"""
    print("开始使用Ollama本地模型重建知识图谱索引...")
    
    # 检查Ollama是否可用
    try:
        import subprocess
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Ollama服务未启动或未安装")
            print("请先安装Ollama并启动服务，然后下载模型")
            print("安装地址: https://ollama.com/download")
            print("下载模型: ollama pull qwen3")
            return
        print("✅ Ollama服务可用")
    except FileNotFoundError:
        print("❌ Ollama未安装")
        return
    
    # 导入GraphBuilder
    from src.indexing.graph_builder import GraphBuilder
    
    # 初始化构建器，使用Ollama
    builder = GraphBuilder(
        input_dir='D:\\projects\\Capstone\\GraphRAG\\data\\input',
        output_dir='D:\\projects\\Capstone\\GraphRAG\\data\\graphrag_output',
        llm_provider='openai',  # 使用openai兼容接口
        llm_model='qwen3',
        embedding_provider='ollama',  # 使用本地embedding
        embedding_model='nomic-embed-text',
    )
    
    # 构建索引
    print("\n正在构建索引...")
    print("这可能需要一段时间，取决于数据量和硬件性能...")
    
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
    rebuild_index_with_ollama()
