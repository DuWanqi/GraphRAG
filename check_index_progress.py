#!/usr/bin/env python3
"""检查索引构建进度"""
import os
import json

# 检查缓存文件数量
cache_dir = "data/graphrag_output/cache/extract_graph"
if os.path.exists(cache_dir):
    cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('_v4')])
    print(f"已缓存的实体提取结果: {cache_files} 个")

# 检查输出文件
output_dir = "data/graphrag_output/output"
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    print(f"\n已生成的输出文件:")
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  - {f}: {size/1024:.1f} KB")

# 检查文档总数
with open("data/graphrag_output/input_json/all_documents.json", 'r', encoding='utf-8') as f:
    docs = json.load(f)
    print(f"\n总文档数: {len(docs)}")
    print(f"处理进度: {cache_files}/{len(docs)} ({cache_files/len(docs)*100:.1f}%)")
