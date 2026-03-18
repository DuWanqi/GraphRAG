"""测试本地embedding"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("测试本地embedding模型...")

from src.llm import LocalEmbedding, get_local_embedding

embedding = get_local_embedding()
print(f"向量维度: {embedding.dimension}")

test_texts = [
    "1988年深圳经济特区成立",
    "邓小平南巡讲话",
    "改革开放的历史意义",
]

print("\n测试embedding:")
for text in test_texts:
    vector = embedding.embed(text)
    print(f"  '{text[:20]}...' -> 向量维度: {len(vector)}, 前5维: {vector[:5]}")

print("\n测试批量embedding:")
vectors = embedding.embed_batch(test_texts)
print(f"  批量处理 {len(vectors)} 条文本")

print("\n本地embedding测试成功!")
