#!/usr/bin/env python3
"""
重构知识图谱索引
"""

import sys
sys.path.insert(0, 'D:\\projects\\Capstone\\GraphRAG\\src')

from indexing.graph_builder import GraphBuilder

def main():
    print("开始重构知识图谱索引...")
    
    # 初始化构建器
    builder = GraphBuilder(
        input_dir='D:\\projects\\Capstone\\GraphRAG\\data\\input',
        output_dir='D:\\projects\\Capstone\\GraphRAG\\data\\graphrag_output',
        llm_provider='gemini',  # 可以根据需要切换到其他提供商
    )
    
    # 构建索引
    print("正在构建索引...")
    print("这可能需要一段时间，取决于数据量和网络速度...")
    
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
