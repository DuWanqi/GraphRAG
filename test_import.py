#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试导入"""

print("开始导入测试...")

try:
    from src.indexing import GraphBuilder
    print("✅ GraphBuilder 导入成功")
    
    builder = GraphBuilder()
    print("✅ GraphBuilder 实例化成功")
    
    stats = builder.get_index_stats()
    print(f"✅ 索引状态: {stats}")
    
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("测试完成")
