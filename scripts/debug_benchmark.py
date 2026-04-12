"""调试评测指标计算"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.evaluation.retrieval_benchmark import RetrievalBenchmark

# 创建评测器
benchmark = RetrievalBenchmark()

# 打印测试用例
print("=" * 60)
print("测试用例:")
print("=" * 60)
for tc in benchmark.test_cases:
    print(f"  Query: {tc.memoir_text[:40]}...")
    print(f"  Ground Truth: {tc.ground_truth_entities}")
    print()

# 打印索引中的实体
print("=" * 60)
print("索引中的实体:")
print("=" * 60)
if benchmark._entities_df is not None:
    for _, row in benchmark._entities_df.head(10).iterrows():
        print(f"  - {row['title']} ({row['type']})")

# 测试单个检索
print("\n" + "=" * 60)
print("检索测试:")
print("=" * 60)

test_case = benchmark.test_cases[0]
print(f"查询: {test_case.memoir_text}")
print(f"Ground Truth: {test_case.ground_truth_entities}")

# 执行关键词检索
from src.retrieval import MemoirParser
parser = MemoirParser()
context = parser.parse(test_case.memoir_text)
print(f"解析结果: year={context.year}, location={context.location}, keywords={context.keywords}")

results = benchmark._strategy_keyword_only(context, 10)
print(f"\n检索结果 ({len(results)} 个):")
for r in results:
    print(f"  - {r['name']} (score={r['score']})")

# 计算匹配
print("\n匹配分析:")
gt_variants = benchmark._normalize_entity_name(test_case.ground_truth_entities[0])
print(f"  Ground Truth 变体: {gt_variants}")

for r in results[:5]:
    name = r['name']
    variants = benchmark._normalize_entity_name(name)
    match = any(v.lower() in [g.lower() for g in gt_variants] for v in variants)
    print(f"  '{name}' -> 变体: {variants[:3]}... 匹配: {match}")
