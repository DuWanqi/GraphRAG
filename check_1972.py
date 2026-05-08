#!/usr/bin/env python3
import json

with open('data/graphrag_output/input_json/all_documents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

matches = [d for d in data if '1972' in d['text']]
print(f'包含1972的文档数: {len(matches)}')
print('\n前10个:')
for i, d in enumerate(matches[:10]):
    print(f'{i+1}. {d["text"][:80]}...')
