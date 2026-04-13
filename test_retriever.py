#!/usr/bin/env python3
import os
import sys

# 设置环境变量
os.environ['GRAPHRAG_OUTPUT_DIR'] = './data/graphrag_output/output'
os.environ['GLM_API_KEY'] = '7b5a8b76396b4f0fa8126e7b046ad583.VcowEIPLMsJnOKMp'

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.retrieval import MemoirRetriever
from src.llm import create_llm_adapter

# 创建检索器
print("创建检索器...")
retriever = MemoirRetriever()

# 手动加载索引数据
print("加载索引数据...")
retriever._load_index_data()

# 检查加载的数据
print(f"\n实体数据框: {retriever._entities_df is not None}")
if retriever._entities_df is not None:
    print(f"实体数量: {len(retriever._entities_df)}")
    print(f"实体列名: {list(retriever._entities_df.columns)}")

print(f"\n关系数据框: {retriever._relationships_df is not None}")
if retriever._relationships_df is not None:
    print(f"关系数量: {len(retriever._relationships_df)}")

print(f"\n社区数据框: {retriever._communities_df is not None}")
if retriever._communities_df is not None:
    print(f"社区数量: {len(retriever._communities_df)}")

print(f"\n文本单元数据框: {retriever._text_units_df is not None}")
if retriever._text_units_df is not None:
    print(f"文本单元数量: {len(retriever._text_units_df)}")

# 测试搜索
print("\n\n测试搜索 '1990年 深圳'...")
from src.retrieval.memoir_parser import MemoirParser

parser = MemoirParser()
context = parser.parse("1990年我在深圳的车间工作", use_llm=False)
print(f"解析结果: 时间={context.year}, 地点={context.location}, 关键词={context.keywords}")

# 搜索实体
if retriever._entities_df is not None:
    entities = retriever._search_entities(context, top_k=10)
    print(f"\n找到实体: {len(entities)} 个")
    for e in entities[:5]:
        print(f"  - {e['name']} ({e['type']}): score={e['score']}")
