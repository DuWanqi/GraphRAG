"""测试向量检索模块"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import MemoirRetriever, VectorRetriever, RetrievalMode

print("Import successful")

vr = VectorRetriever()
print(f"Vector index ready: {vr.is_ready()}")

retriever = MemoirRetriever()
print(f"MemoirRetriever created")
