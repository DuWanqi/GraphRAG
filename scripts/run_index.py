#!/usr/bin/env python3
"""
运行索引构建
"""

from src.indexing import GraphBuilder

def main():
    print("开始构建知识图谱索引...")
    builder = GraphBuilder()
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
