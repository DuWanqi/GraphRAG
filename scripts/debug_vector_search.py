"""调试向量检索"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import lancedb
import numpy as np

# 连接LanceDB
db = lancedb.connect("data/output/output/lancedb")
table = db.open_table("default-entity-description")
df = table.to_pandas()

print(f"实体数量: {len(df)}")
print(f"列: {df.columns.tolist()}")
print(f"向量维度: {len(df['vector'].iloc[0])}")

# 测试本地embedding
from src.llm import LocalEmbedding
embedding = LocalEmbedding()

query = "深圳经济特区"
query_vector = embedding.embed(query)
print(f"\n查询: {query}")
print(f"查询向量维度: {len(query_vector)}")

# 测试向量搜索
try:
    query_array = np.array(query_vector, dtype=np.float32)
    results = table.search(query_array).limit(5).to_pandas()
    print(f"\n搜索结果:")
    for _, row in results.iterrows():
        print(f"  - {row['text'][:50]}... (distance={row.get('_distance', 'N/A')})")
except Exception as e:
    print(f"搜索失败: {e}")
