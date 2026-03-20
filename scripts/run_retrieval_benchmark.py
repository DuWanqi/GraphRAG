"""
检索策略对比评测脚本
独立运行，不修改原有代码

使用方法：
python scripts/run_retrieval_benchmark.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import run_benchmark, RetrievalBenchmark, TestCase


def main():
    """运行检索策略对比评测"""
    print("=" * 60)
    print("检索策略对比评测")
    print("=" * 60)
    
    custom_test_cases = [
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
    
    output_dir = project_root / "benchmark_results"
    
    results = run_benchmark(
        test_cases=custom_test_cases,
        output_dir=str(output_dir),
        llm_provider="hunyuan",
        use_llm_judge=True,
    )
    
    if results:
        print("\n" + "=" * 60)
        print("评测完成！")
        print(f"结果已保存到: {output_dir}")
        print("=" * 60)
        
        print("\n各策略F1得分对比:")
        for name, result in sorted(results.items(), key=lambda x: x[1].avg_f1, reverse=True):
            print(f"  {name}: {result.avg_f1:.4f}")
    else:
        print("\n评测失败：请先构建GraphRAG索引")
        print("运行命令: python -c \"from src.indexing import GraphBuilder; GraphBuilder().build_index_sync()\"")


if __name__ == "__main__":
    main()
