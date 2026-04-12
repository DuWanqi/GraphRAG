"""测试改进后的关键词提取"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import MemoirParser

parser = MemoirParser()

test_cases = [
    "1988年夏天，我从大学毕业，怀揣着梦想来到了深圳。",
    "1992年，邓小平南巡讲话后，深圳迎来了新的发展机遇。",
    "2001年中国加入WTO后，外贸行业蓬勃发展。",
    "我参加了高考，考上了北京大学。",
    "2010年上海世博会期间，我作为志愿者参与工作。",
]

print("=" * 60)
print("关键词提取测试（jieba TF-IDF）")
print("=" * 60)

for text in test_cases:
    context = parser.parse(text)
    print(f"\n文本: {text[:40]}...")
    print(f"  年份: {context.year}")
    print(f"  地点: {context.location}")
    print(f"  关键词: {context.keywords}")
    print(f"  查询: {context.to_query()}")
