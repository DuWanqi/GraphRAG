"""
使用真实测试集对比混合检索 vs 普通RAG
测试集: data/memoirs/segments/
"""
import sys
from pathlib import Path
import asyncio
import time
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval import MemoirRetriever
from src.retrieval.plain_vector_rag_retriever import PlainVectorRAGRetriever


def load_test_cases():
    """加载测试集"""
    segments_dir = project_root / "data" / "memoirs" / "segments"
    test_cases = []
    
    for json_file in segments_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                test_cases.append({
                    'id': item['id'],
                    'chapter': item['chapter'],
                    'text': item['original_text'],
                    'ground_truth_entities': item.get('historical_tag', []),
                })
    
    return test_cases


async def compare_strategies():
    """对比混合检索和普通RAG"""
    
    # 加载测试集
    print("加载测试集...")
    test_cases = load_test_cases()
    print(f"共加载 {len(test_cases)} 个测试用例")
    
    # 只测试前10个作为示例
    test_cases = test_cases[:10]
    
    print("=" * 70)
    print(f"对比测试：混合检索 vs 普通RAG (前{len(test_cases)}个样本)")
    print("=" * 70)
    
    # 初始化检索器
    print("\n初始化检索器...")
    hybrid_retriever = MemoirRetriever()
    
    # 普通RAG
    plain_rag = PlainVectorRAGRetriever()
    try:
        await plain_rag.build_index()
        plain_rag_ready = True
        print("普通RAG就绪!")
    except Exception as e:
        print(f"普通RAG初始化失败: {e}")
        plain_rag_ready = False
    
    print("混合检索就绪!")
    
    # 对比每个测试用例
    print("\n" + "=" * 70)
    print("开始对比测试")
    print("=" * 70)
    
    hybrid_total_recall = 0
    plain_total_recall = 0
    
    for tc in test_cases:
        print(f"\n{'-' * 70}")
        print(f"查询 [{tc['id']}]: {tc['text'][:50]}...")
        print(f"期望实体: {tc['ground_truth_entities']}")
        
        # 混合检索
        start = time.time()
        hybrid_result = await hybrid_retriever.retrieve(tc['text'], top_k=10, mode='hybrid')
        hybrid_time = time.time() - start
        hybrid_entities = [e.get('name', '') for e in hybrid_result.entities[:10]]
        
        # 计算召回率
        hybrid_matched = set(hybrid_entities) & set(tc['ground_truth_entities'])
        hybrid_recall = len(hybrid_matched) / len(tc['ground_truth_entities']) if tc['ground_truth_entities'] else 0
        hybrid_total_recall += hybrid_recall
        
        print(f"\n  混合检索 ({hybrid_time:.2f}s):")
        print(f"    召回实体: {hybrid_entities[:5]}")
        print(f"    匹配实体: {list(hybrid_matched)}")
        print(f"    召回率: {hybrid_recall:.2f}")
        
        # 普通RAG
        if plain_rag_ready:
            start = time.time()
            plain_result = await plain_rag.retrieve(tc['text'], top_k=10)
            plain_time = time.time() - start
            
            # 从text_units中匹配实体
            plain_matched = set()
            for text in plain_result.text_units[:10]:
                for entity in tc['ground_truth_entities']:
                    if entity in text:
                        plain_matched.add(entity)
            
            plain_recall = len(plain_matched) / len(tc['ground_truth_entities']) if tc['ground_truth_entities'] else 0
            plain_total_recall += plain_recall
            
            print(f"\n  普通RAG ({plain_time:.2f}s):")
            print(f"    匹配实体: {list(plain_matched)}")
            print(f"    召回率: {plain_recall:.2f}")
    
    # 计算平均召回率
    hybrid_avg = hybrid_total_recall / len(test_cases)
    print("\n" + "=" * 70)
    print(f"混合检索平均召回率: {hybrid_avg:.2f}")
    
    if plain_rag_ready:
        plain_avg = plain_total_recall / len(test_cases)
        print(f"普通RAG平均召回率: {plain_avg:.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(compare_strategies())
