"""
测试FActScore集成
验证FActScoreChecker是否正常工作
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from src.llm import create_llm_adapter
from src.evaluation import FActScoreChecker, FactCheckResult


async def test_factscore_checker():
    """测试FActScoreChecker"""
    
    print("=" * 60)
    print("测试FActScoreChecker集成")
    print("=" * 60)
    
    # 创建LLM适配器
    print("\n1. 创建LLM适配器...")
    try:
        llm_adapter = create_llm_adapter(provider="gemini")
        print("✅ LLM适配器创建成功")
    except Exception as e:
        print(f"❌ LLM适配器创建失败: {e}")
        return
    
    # 创建FActScoreChecker
    print("\n2. 创建FActScoreChecker...")
    try:
        checker = FActScoreChecker(llm_adapter=llm_adapter)
        print("✅ FActScoreChecker创建成功")
    except Exception as e:
        print(f"❌ FActScoreChecker创建失败: {e}")
        return
    
    # 测试事实性检查
    print("\n3. 测试事实性检查...")
    memoir_text = "1988年夏天，我从大学毕业，怀揣着梦想来到了深圳。"
    generated_text = "1988年，深圳经济特区正处于快速发展阶段。这一年，深圳的GDP增长率达到了30%以上，成为中国改革开放的窗口和试验田。"
    
    print(f"回忆录: {memoir_text}")
    print(f"生成文本: {generated_text}")
    
    try:
        result = await checker.check(
            memoir_text=memoir_text,
            generated_text=generated_text,
            retrieval_result=None,
            use_llm=True,
        )
        
        print("\n✅ 事实性检查完成")
        print(f"是否事实一致: {result.is_factual}")
        print(f"置信度: {result.confidence:.2%}")
        print(f"实体覆盖率: {result.entity_coverage:.2%}")
        print(f"证据支持度: {result.evidence_support:.2%}")
        print(f"总结: {result.summary}")
        
        if result.inconsistencies:
            print(f"\n发现 {len(result.inconsistencies)} 个问题:")
            for inc in result.inconsistencies:
                print(f"  - {inc.type.value}: {inc.explanation}")
        
    except Exception as e:
        print(f"❌ 事实性检查失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！FActScore集成成功")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_factscore_checker())
