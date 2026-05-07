#!/usr/bin/env python3
import json

# 检查原始JSON文件
with open('data/graphrag_output/input/1960_1978.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'1960_1978.json 总条目数: {len(data)}')

# 查找1972年的条目
matches = [d for d in data if '1972' in d.get('date', '') or '1972' in d.get('content', '')]
print(f'包含1972的条目数: {len(matches)}')

if matches:
    print('\n前5个1972年条目:')
    for i, d in enumerate(matches[:5]):
        print(f'{i+1}. [{d.get("date")}] {d.get("title")}')
        print(f'   内容: {d.get("content")[:80]}...')
else:
    print('\n没有找到1972年的数据！')
    # 检查日期字段
    print('\n检查前10个条目的日期:')
    for i, d in enumerate(data[:10]):
        print(f'{i+1}. date={d.get("date")}, title={d.get("title")}')
