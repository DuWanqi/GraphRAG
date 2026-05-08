#!/usr/bin/env python3
"""调试转换脚本"""

import json
from pathlib import Path

input_file = Path("data/graphrag_output/input/1960_1978.json")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"总条目数: {len(data)}")

# 检查1972年条目
count_1972 = 0
for i, item in enumerate(data):
    if isinstance(item, dict):
        date = item.get('date', '')
        content = item.get('content', '')
        if '1972' in date or '1972' in content:
            count_1972 += 1
            if count_1972 <= 3:
                print(f"\n条目 {i}:")
                print(f"  title: {item.get('title')}")
                print(f"  date: {date}")
                print(f"  content: {content[:60]}...")

print(f"\n包含1972的条目数: {count_1972}")

# 模拟转换脚本的处理
print("\n--- 模拟转换脚本处理 ---")
converted = []
for item in data:
    if isinstance(item, dict):
        text = item.get('content', item.get('text', item.get('description', '')))
        if text:
            converted.append(text)

print(f"转换后文档数: {len(converted)}")
matches = [t for t in converted if '1972' in t]
print(f"包含1972的文档数: {len(matches)}")

# 检查前几个转换后的内容
print("\n前5个转换后的内容:")
for i, t in enumerate(converted[:5]):
    has_1972 = '1972' in t
    print(f"{i+1}. [1972={has_1972}] {t[:60]}...")
