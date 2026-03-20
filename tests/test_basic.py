"""
基础功能测试
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestMemoirParser:
    """回忆录解析器测试"""
    
    def test_extract_year(self):
        """测试年份提取"""
        from src.retrieval.memoir_parser import MemoirParser
        
        parser = MemoirParser()
        
        # 测试用例
        test_cases = [
            ("1988年夏天，我从大学毕业", "1988"),
            ("那是88年的事情", "1988"),
            ("2020年疫情爆发", "2020"),
            ("记得是在1992年春天", "1992"),
        ]
        
        for text, expected_year in test_cases:
            result = parser.parse(text)
            assert result.year == expected_year, f"Expected {expected_year}, got {result.year}"
    
    def test_extract_location(self):
        """测试地点提取"""
        from src.retrieval.memoir_parser import MemoirParser
        
        parser = MemoirParser()
        
        # 测试用例
        test_cases = [
            ("我来到了深圳", "深圳"),
            ("在北京的日子", "北京"),
            ("前往上海出差", "上海"),
        ]
        
        for text, expected_location in test_cases:
            result = parser.parse(text)
            assert expected_location in (result.location or ""), \
                f"Expected {expected_location} in {result.location}"
    
    def test_to_query(self):
        """测试查询生成"""
        from src.retrieval.memoir_parser import MemoirParser
        
        parser = MemoirParser()
        result = parser.parse("1988年夏天，我从大学毕业，来到了深圳创业")
        
        query = result.to_query()
        assert "1988" in query or "深圳" in query


class TestLiteraryMetrics:
    """文学性指标测试"""
    
    def test_length_score(self):
        """测试长度评分"""
        from src.evaluation.metrics import LiteraryMetrics
        
        # 最佳长度范围
        text_optimal = "这是一段适中长度的文本。" * 30
        result = LiteraryMetrics.length_score(text_optimal)
        assert result.value >= 0.7
        
        # 过短
        text_short = "太短"
        result = LiteraryMetrics.length_score(text_short)
        assert result.value < 0.5
    
    def test_transition_usage(self):
        """测试过渡词使用"""
        from src.evaluation.metrics import LiteraryMetrics
        
        # 有过渡词
        text_with_transition = "那时候的深圳，正是改革开放的前沿。当时的年轻人都怀揣梦想。"
        result = LiteraryMetrics.transition_usage(text_with_transition)
        assert result.value >= 0.7
        
        # 无过渡词
        text_without = "深圳是一个城市。这里有很多人。"
        result = LiteraryMetrics.transition_usage(text_without)
        assert result.value < 0.7


class TestDataLoader:
    """数据加载器测试"""
    
    def test_parse_txt_event(self):
        """测试TXT事件解析"""
        from src.indexing.data_loader import DataLoader
        
        loader = DataLoader()
        
        text = """事件标题：改革开放
时间：1978年
地点：北京

这是改革开放的描述内容。"""
        
        event = loader._parse_txt_event(text)
        
        assert event is not None
        assert event.title == "改革开放"
        assert event.date == "1978年"
        assert event.location == "北京"
        assert "改革开放" in event.content


class TestConfig:
    """配置测试"""
    
    def test_settings_load(self):
        """测试配置加载"""
        from src.config import get_settings
        
        settings = get_settings()
        
        # 检查默认值
        assert settings.default_llm_provider is not None
        assert settings.graphrag_input_dir is not None


class TestLLMAdapter:
    """LLM适配器测试"""
    
    def test_provider_enum(self):
        """测试提供商枚举"""
        from src.llm import LLMProvider
        
        assert LLMProvider.DEEPSEEK.value == "deepseek"
        assert LLMProvider.QWEN.value == "qwen"
        assert LLMProvider.HUNYUAN.value == "hunyuan"
        assert LLMProvider.GEMINI.value == "gemini"
    
    def test_available_providers(self):
        """测试获取可用提供商"""
        from src.llm import get_available_providers
        
        available = get_available_providers()
        
        assert isinstance(available, dict)
        assert "deepseek" in available
        assert "qwen" in available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
