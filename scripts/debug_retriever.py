"""调试检索器"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.retrieval import MemoirRetriever, MemoirParser

output_dir = Path("data/output/output")

entities_file = output_dir / "entities.parquet"
df = pd.read_parquet(entities_file)
print("所有实体:")
for _, row in df.iterrows():
    print(f"  - {row['title']} ({row['type']})")

print("\n" + "="*60)

retriever = MemoirRetriever()
parser = MemoirParser()

test_text = "1988年夏天，我从大学毕业，怀揣着梦想来到了深圳。"
context = parser.parse(test_text, use_llm=False)

print(f"\n解析结果:")
print(f"  时间: {context.year}")
print(f"  地点: {context.location}")
print(f"  关键词: {context.keywords}")
print(f"  查询: {context.to_query()}")

print("\n检索测试:")
results = retriever._search_entities(context, 10)
print(f"  检索到 {len(results)} 个实体")
for r in results:
    print(f"    - {r['name']} (score={r['score']})")
