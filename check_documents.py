#!/usr/bin/env python3
import json
from collections import Counter

with open('data/graphrag_output/input_json/all_documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'总文档数: {len(data)}')

# 按来源统计
sources = Counter([d.get('source', 'unknown') for d in data])
print('\n按来源统计:')
for source, count in sources.most_common():
    print(f'  {source}: {count}')

# 检查1972
matches = [d for d in data if '1972' in d['text']]
print(f'\n包含1972的文档数: {len(matches)}')

# 显示前几个文档的内容示例
print('\n前5个文档示例:')
for i, d in enumerate(data[:5]):
    print(f'{i+1}. [{d["source"]}] {d["text"][:60]}...')
