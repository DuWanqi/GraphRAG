"""
对比修复后的混合检索 vs 普通RAG
"""
import sys
from pathlib import Path
import asyncio
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import MemoirRetriever
from src.evaluation.retrieval_benchmark import TestCase


async def compare_strategies():
    """对比混合检索"""
    
    # 测试用例
    test_cases = [
        TestCase(
            query_id="001",
            memoir_text="1988年夏天，我从大学毕业，怀揣着梦想来到了深圳。",
            ground_truth_entities=["深圳", "经济特区"],
            ground_truth_time="1988",
            ground_truth_location="深圳",
            query_type="simple"
        ),
        TestCase(
            query_id="002",
            memoir_text="1992年，邓小平南巡讲话后，深圳迎来了新的发展机遇。",
            ground_truth_entities=["邓小平", "南巡", "深圳"],
            ground_truth_time="1992",
            ground_truth_location="深圳",
            query_type="simple"
        ),
        TestCase(
            query_id="003",
            memoir_text="1990年上海浦东新区开发开放，我参与了早期的建设工作。",
            ground_truth_entities=["上海", "浦东新区"],
            ground_truth_time="1990",
            ground_truth_location="上海",
            query_type="simple"
        ),
        TestCase(
            query_id="004",
            memoir_text="改革开放初期，我从农村来到城市，经历了从计划经济到市场经济的转变。",
            ground_truth_entities=["改革开放", "计划经济", "市场经济"],
            ground_truth_time=None,
            ground_truth_location=None,
            query_type="multi_hop"
        ),
        TestCase(
            query_id="005",
            memoir_text="2001年中国加入WTO后，外贸行业蓬勃发展，我创办了自己的出口公司。",
            ground_truth_entities=["WTO", "外贸"],
            ground_truth_time="2001",
            ground_truth_location=None,
            query_type="temporal"
        ),
        TestCase(
            query_id="006",
            memoir_text="2008年北京奥运会期间，我作为志愿者参与了赛事服务工作。",
            ground_truth_entities=["北京", "奥运会"],
            ground_truth_time="2008",
            ground_truth_location="北京",
            query_type="simple"
        ),
        TestCase(
            query_id="007",
            memoir_text="1997年香港回归祖国，我在电视前见证了这一历史时刻。",
            ground_truth_entities=["香港", "回归"],
            ground_truth_time="1997",
            ground_truth_location="香港",
            query_type="simple"
        ),
        TestCase(
            query_id="008",
            memoir_text="2010年上海世博会举办，我带着家人参观了各国展馆。",
            ground_truth_entities=["上海", "世博会"],
            ground_truth_time="2010",
            ground_truth_location="上海",
            query_type="simple"
        ),
    ]
    
    print("=" * 70)
    print("对比测试：混合检索 (修复后)")
    print("=" * 70)
    
    # 初始化检索器
    print("\n初始化检索器...")
    hybrid_retriever = MemoirRetriever()
    print("混合检索就绪!")
    
    # 对比每个测试用例
    print("\n" + "=" * 70)
    print("开始对比测试")
    print("=" * 70)
    
    total_recall = 0
    
    for tc in test_cases:
        print(f"\n{'-' * 70}")
        print(f"查询 [{tc.query_id}]: {tc.memoir_text[:40]}...")
        print(f"期望实体: {tc.ground_truth_entities}")
        
        # 混合检索
        start = time.time()
        hybrid_result = await hybrid_retriever.retrieve(tc.memoir_text, top_k=10, mode='hybrid')
        hybrid_time = time.time() - start
        hybrid_entities = [e.get('name', '') for e in hybrid_result.entities[:5]]
        
        # 计算召回率
        matched = set(hybrid_entities) & set(tc.ground_truth_entities)
        hybrid_recall = len(matched) / len(tc.ground_truth_entities)
        total_recall += hybrid_recall
        
        print(f"\n  混合检索 ({hybrid_time:.2f}s):")
        print(f"    召回实体: {hybrid_entities}")
        print(f"    匹配实体: {list(matched)}")
        print(f"    召回率: {hybrid_recall:.2f}")
    
    avg_recall = total_recall / len(test_cases)
    print("\n" + "=" * 70)
    print(f"平均召回率: {avg_recall:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(compare_strategies())
