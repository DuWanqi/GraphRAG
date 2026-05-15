"""检索效果对比测试脚本"""
import asyncio
import time
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import MemoirRetriever
from src.retrieval.official_graphrag_retriever import OfficialGraphRAGRetriever


async def test_keyword_retrieval(retriever: MemoirRetriever, query: str, top_k: int = 10):
    """测试关键词检索"""
    start_time = time.time()
    result = await retriever.retrieve(query, top_k=top_k, mode="keyword")
    elapsed = time.time() - start_time
    return result, elapsed


async def test_vector_retrieval(retriever: MemoirRetriever, query: str, top_k: int = 10):
    """测试自定义向量检索"""
    start_time = time.time()
    result = await retriever.retrieve(query, top_k=top_k, mode="vector")
    elapsed = time.time() - start_time
    return result, elapsed


async def test_hybrid_retrieval(retriever: MemoirRetriever, query: str, top_k: int = 10):
    """测试混合检索"""
    start_time = time.time()
    result = await retriever.retrieve(query, top_k=top_k, mode="hybrid")
    elapsed = time.time() - start_time
    return result, elapsed


async def test_official_retrieval(official_retriever, query: str, top_k: int = 10):
    """测试官方 API 检索"""
    start_time = time.time()
    result = await official_retriever.search(query=query)
    elapsed = time.time() - start_time
    return result, elapsed


def print_result_comparison(name: str, result, elapsed: float, top_n: int = 5):
    """打印检索结果对比"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"耗时: {elapsed:.2f}秒")
    print(f"实体数: {len(result.entities)}")
    print(f"关系数: {len(result.relationships)}")
    print(f"文本单元数: {len(result.text_units)}")
    print(f"社区报告数: {len(result.communities)}")

    if hasattr(result, 'response') and result.response:
        print(f"\n官方 API 回答:\n{result.response[:500]}...")

    if result.entities:
        print(f"\n前 {top_n} 个实体:")
        for i, e in enumerate(result.entities[:top_n]):
            name_str = e.get("name", "N/A")
            score = e.get("score", 0)
            source = e.get("source", "unknown")
            print(f"  {i+1}. {name_str} (score: {score:.2f}, source: {source})")

    if result.relationships:
        print(f"\n前 {top_n} 个关系:")
        for i, r in enumerate(result.relationships[:top_n]):
            src = r.get("source", "")
            tgt = r.get("target", "")
            print(f"  {i+1}. {src} -> {tgt}")


async def main():
    """主函数"""
    print("=" * 60)
    print("GraphRAG 检索效果对比测试")
    print("=" * 60)

    # 测试查询
    query = "1992年邓小平在深圳"
    top_k = 10

    print(f"\n测试查询: {query}")
    print(f"top_k: {top_k}")

    # 初始化检索器
    print("\n初始化检索器...")
    memoir_retriever = MemoirRetriever()

    # 初始化官方检索器
    official_retriever = OfficialGraphRAGRetriever()

    # 检查官方检索器是否就绪
    if not official_retriever.is_ready():
        print("[WARNING] 官方检索器未就绪，可能缺少配置或数据")

    # 测试关键词检索
    print("\n" + "=" * 60)
    print("开始测试...")
    print("=" * 60)

    keyword_result, keyword_time = await test_keyword_retrieval(memoir_retriever, query, top_k)
    print_result_comparison("关键词检索 (Keyword)", keyword_result, keyword_time)

    vector_result, vector_time = await test_vector_retrieval(memoir_retriever, query, top_k)
    print_result_comparison("自定义向量检索 (Vector)", vector_result, vector_time)

    hybrid_result, hybrid_time = await test_hybrid_retrieval(memoir_retriever, query, top_k)
    print_result_comparison("混合检索 (Hybrid)", hybrid_result, hybrid_time)

    # 官方 API 检索
    if official_retriever.is_ready():
        official_result, official_time = await test_official_retrieval(official_retriever, query, top_k)
        print_result_comparison("官方 GraphRAG API (local_search)", official_result, official_time)
    else:
        print("\n[跳过] 官方 API 检索（未就绪）")
        official_result = None
        official_time = 0

    # 总结
    print("\n" + "=" * 60)
    print("性能总结")
    print("=" * 60)
    print(f"关键词检索:    {keyword_time:.2f}秒 | 实体: {len(keyword_result.entities)} | 关系: {len(keyword_result.relationships)}")
    print(f"自定义向量检索: {vector_time:.2f}秒 | 实体: {len(vector_result.entities)} | 关系: {len(vector_result.relationships)}")
    print(f"混合检索:      {hybrid_time:.2f}秒 | 实体: {len(hybrid_result.entities)} | 关系: {len(hybrid_result.relationships)}")
    if official_result:
        print(f"官方 API:     {official_time:.2f}秒 | 实体: {len(official_result.entities)} | 关系: {len(official_result.relationships)}")

    # 实体召回对比
    print("\n" + "=" * 60)
    print("实体召回对比")
    print("=" * 60)

    keyword_entities = set(e.get("name", "").lower() for e in keyword_result.entities)
    vector_entities = set(e.get("name", "").lower() for e in vector_result.entities)
    hybrid_entities = set(e.get("name", "").lower() for e in hybrid_result.entities)

    print(f"关键词检索召回实体: {len(keyword_entities)}")
    print(f"自定义向量检索召回实体: {len(vector_entities)}")
    print(f"混合检索召回实体: {len(hybrid_entities)}")

    if official_result:
        official_entities = set(e.get("name", "").lower() for e in official_result.entities)
        print(f"官方 API 召回实体: {len(official_entities)}")

        # 计算交集
        if official_entities:
            vector_overlap = len(vector_entities & official_entities)
            hybrid_overlap = len(hybrid_entities & official_entities)
            keyword_overlap = len(keyword_entities & official_entities)
            print(f"\n与官方 API 的重叠率:")
            print(f"  关键词检索: {keyword_overlap}/{len(official_entities)} = {keyword_overlap/len(official_entities)*100:.1f}%")
            print(f"  自定义向量检索: {vector_overlap}/{len(official_entities)} = {vector_overlap/len(official_entities)*100:.1f}%")
            print(f"  混合检索: {hybrid_overlap}/{len(official_entities)} = {hybrid_overlap/len(official_entities)*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
