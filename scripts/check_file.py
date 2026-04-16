#!/usr/bin/env python3
"""检查文件内容"""

with open(r'd:\projects\Capstone\GraphRAG\data\output\input\shenzhen.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    non_empty = [l for l in lines if l.strip()]
    print(f'总行数: {len(lines)}')
    print(f'非空行数: {len(non_empty)}')
    print(f'\n前5行:')
    for i, line in enumerate(non_empty[:5]):
        print(f'  {i+1}. {line[:80]}...')
