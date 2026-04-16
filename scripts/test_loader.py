#!/usr/bin/env python3
"""测试数据加载器"""

import sys
sys.path.insert(0, 'D:\\projects\\Capstone\\GraphRAG\\src')

from indexing.data_loader import DataLoader

loader = DataLoader('D:\\projects\\Capstone\\GraphRAG\\data\\input')
events = loader.load_txt('D:\\projects\\Capstone\\GraphRAG\\data\\input\\gaigekaifang_processed.txt')

print(f"成功加载 {len(events)} 个事件\n")

# 显示前3个事件
for i, event in enumerate(events[:3]):
    print(f"事件 {i+1}:")
    print(f"  标题: {event.title[:60]}...")
    print(f"  时间: {event.date}")
    print(f"  地点: {event.location}")
    print(f"  内容: {event.content[:80]}...")
    print()
