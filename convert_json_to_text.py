#!/usr/bin/env python3
"""
将 JSON 文件转换为文本格式，以便 GraphRAG 处理
"""
import json
import os
from pathlib import Path

input_dir = Path("data/graphrag_output/input")
output_dir = Path("data/graphrag_output/input_text")
output_dir.mkdir(exist_ok=True)

# 复制现有的 txt 文件
for txt_file in input_dir.glob("*.txt"):
    output_file = output_dir / txt_file.name
    output_file.write_text(txt_file.read_text(encoding='utf-8'), encoding='utf-8')
    print(f"复制: {txt_file.name}")

# 转换 JSON 文件
for json_file in input_dir.glob("*.json"):
    print(f"\n处理: {json_file.name}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取所有文本内容
        texts = []

        def extract_text(obj, depth=0):
            """递归提取文本"""
            if depth > 10:  # 防止无限递归
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        # 如果是描述性字段，提取内容
                        if key in ['description', 'content', 'text', 'event', 'title', 'summary']:
                            texts.append(value)
                    elif isinstance(value, (dict, list)):
                        extract_text(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, depth + 1)

        extract_text(data)

        # 写入文本文件
        output_file = output_dir / f"{json_file.stem}.txt"
        output_file.write_text('\n\n'.join(texts), encoding='utf-8')
        print(f"  提取了 {len(texts)} 段文本")
        print(f"  输出: {output_file.name}")

    except Exception as e:
        print(f"  错误: {e}")

print("\