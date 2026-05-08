"""
Quick verification script for novel content generation implementation
"""

# Test 1: Import verification
print("=" * 60)
print("Test 1: Import Verification")
print("=" * 60)

try:
    from src.generation import extract_novel_content, NovelContentBrief
    print("✓ extract_novel_content imported")
    print("✓ NovelContentBrief imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

try:
    from src.evaluation import (
        novel_content_ratio_metric,
        novel_content_grounding_metric,
        expansion_depth_metric,
    )
    print("✓ novel_content_ratio_metric imported")
    print("✓ novel_content_grounding_metric imported")
    print("✓ expansion_depth_metric imported")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Basic functionality
print("\n" + "=" * 60)
print("Test 2: Basic Functionality")
print("=" * 60)

from src.retrieval import RetrievalResult, MemoirContext

memoir_text = "1980年，我考上了大学。"

# Create a mock retrieval result
mock_context = MemoirContext(
    original_text=memoir_text,
    year="1980",
    location="北京"
)
mock_retrieval = RetrievalResult(
    query=memoir_text,
    context=mock_context,
    entities=[
        {"name": "恢复高考", "description": "1977年恢复高考制度"},
        {"name": "改革开放", "description": "1978年开始的改革开放政策"},
        {"name": "大学", "description": "高等教育机构"},
    ],
    relationships=[
        {"source": "恢复高考", "target": "邓小平", "description": "邓小平推动恢复高考"},
    ],
    communities=[
        {"summary": "1977年恢复高考制度，标志着中国教育改革的开始"},
    ],
    text_units=["1980年代初期，改革开放政策全面展开"],
)

# Test extract_novel_content
print(f"\n输入文本: {memoir_text}")
novel_brief = extract_novel_content(memoir_text, mock_retrieval)

print(f"\n新内容提取结果:")
print(f"  新实体数量: {len(novel_brief.novel_entities)}")
print(f"  对齐实体数量: {len(novel_brief.aligned_entities)}")
print(f"  新关系数量: {len(novel_brief.novel_relationships)}")
print(f"  新背景片段数量: {len(novel_brief.novel_snippets)}")
print(f"  摘要: {novel_brief.summary}")

if novel_brief.novel_entities:
    print(f"\n新实体:")
    for entity in novel_brief.novel_entities:
        name = entity.get("name", "")
        print(f"    - {name}")

if novel_brief.aligned_entities:
    print(f"\n对齐实体:")
    for entity in novel_brief.aligned_entities:
        name = entity.get("name", "")
        print(f"    - {name}")

# Test format_for_prompt
formatted = novel_brief.format_for_prompt()
print(f"\nPrompt 格式化结果:")
print(f"  aligned_context 长度: {len(formatted['aligned_context'])} 字符")
print(f"  novel_context 长度: {len(formatted['novel_context'])} 字符")

print(f"\naligned_context 内容:")
print(formatted['aligned_context'][:200])

print(f"\nnovel_context 内容:")
print(formatted['novel_context'][:200])

# Test 3: Metrics
print("\n" + "=" * 60)
print("Test 3: Metrics Calculation")
print("=" * 60)

generated_text = """
一九八零年，我收到了大学录取通知书。距离恢复高考已经三年，
整个国家都在改革开放的浪潮中焕发生机。父亲激动得一夜没睡，
他说："你赶上了好时候，国家需要人才。"母亲在一旁默默流泪。
"""

print(f"\n生成文本: {generated_text.strip()}")

ratio_metric = novel_content_ratio_metric(memoir_text, generated_text, novel_brief)
print(f"\n新内容引入率:")
print(f"  分数: {ratio_metric.value:.2f} / {ratio_metric.max_value}")
print(f"  说明: {ratio_metric.explanation}")

grounding_metric = novel_content_grounding_metric(memoir_text, generated_text, novel_brief)
print(f"\n新内容溯源率:")
print(f"  分数: {grounding_metric.value:.2f} / {grounding_metric.max_value}")
print(f"  说明: {grounding_metric.explanation}")

depth_metric = expansion_depth_metric(memoir_text, generated_text, novel_brief)
print(f"\n扩展深度:")
print(f"  分数: {depth_metric.value:.2f} / {depth_metric.max_value}")
print(f"  说明: {depth_metric.explanation}")

# Test 4: Integration with calculate_all_metrics
print("\n" + "=" * 60)
print("Test 4: Integration with calculate_all_metrics")
print("=" * 60)

from src.evaluation import calculate_all_metrics

metrics = calculate_all_metrics(
    memoir_text,
    generated_text,
    reference_entities=["大学"],
    reference_year="1980",
    novel_content_brief=novel_brief,
)

print(f"\n所有指标:")
for name, metric in metrics.items():
    print(f"  {name}: {metric.value:.2f} / {metric.max_value} - {metric.explanation}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
