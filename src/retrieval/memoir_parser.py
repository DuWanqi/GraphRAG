"""
回忆录文本解析器
从回忆录文本中提取时间、地点、事件关键词
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ..llm import LLMAdapter, create_llm_adapter


@dataclass
class MemoirContext:
    """回忆录上下文信息"""
    original_text: str
    year: Optional[str] = None
    month: Optional[str] = None
    location: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    
    def to_query(self) -> str:
        """转换为检索查询"""
        parts = []
        if self.year:
            parts.append(f"{self.year}年")
        if self.location:
            parts.append(self.location)
        if self.keywords:
            parts.extend(self.keywords[:3])
        return " ".join(parts) if parts else self.original_text[:100]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "original_text": self.original_text,
            "year": self.year,
            "month": self.month,
            "location": self.location,
            "keywords": self.keywords,
            "entities": self.entities,
        }


class MemoirParser:
    """
    回忆录解析器
    
    功能：
    1. 从回忆录文本中提取时间信息（年份、月份）
    2. 提取地点信息
    3. 提取关键事件和实体
    4. 使用LLM进行深度语义理解
    """
    
    # 年份正则表达式
    YEAR_PATTERNS = [
        r'(\d{4})\s*年',                     # 1988年
        r'(\d{2})\s*年',                     # 88年
        r'一九([零一二三四五六七八九]{2})年', # 一九八八年
        r'二[零〇]([零一二三四五六七八九]{2})年', # 二零二零年
        r'(\d{4})',                          # 纯数字年份
    ]
    
    # 月份正则表达式
    MONTH_PATTERNS = [
        r'(\d{1,2})\s*月',                   # 8月
        r'([一二三四五六七八九十]+)月',        # 八月
    ]
    
    # 常见地点关键词
    LOCATION_KEYWORDS = [
        '省', '市', '县', '区', '镇', '村',
        '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '重庆',
        '香港', '澳门', '台湾',
    ]
    
    def __init__(self, llm_adapter: Optional[LLMAdapter] = None):
        """
        初始化解析器
        
        Args:
            llm_adapter: LLM适配器，用于深度语义理解
        """
        self.llm_adapter = llm_adapter
    
    def parse(self, text: str, use_llm: bool = True) -> MemoirContext:
        """
        解析回忆录文本
        
        Args:
            text: 回忆录文本
            use_llm: 是否使用LLM进行深度解析
            
        Returns:
            MemoirContext: 解析后的上下文信息
        """
        context = MemoirContext(original_text=text)
        
        # 规则提取
        context.year = self._extract_year(text)
        context.month = self._extract_month(text)
        context.location = self._extract_location(text)
        context.keywords = self._extract_keywords(text)
        
        return context
    
    async def parse_with_llm(self, text: str) -> MemoirContext:
        """
        使用LLM进行深度解析
        
        Args:
            text: 回忆录文本
            
        Returns:
            MemoirContext: 解析后的上下文信息
        """
        # 先进行规则提取
        context = self.parse(text, use_llm=False)
        
        if not self.llm_adapter:
            return context
        
        # 使用LLM提取更多信息
        prompt = f"""请从以下回忆录片段中提取关键信息，以JSON格式返回：

回忆录片段：
{text}

请提取：
1. year: 年份（如"1988"）
2. month: 月份（如"8"）
3. location: 地点（城市或更具体的地点）
4. keywords: 3-5个关键词（描述主要事件或主题）
5. entities: 涉及的重要实体列表，每个实体包含name和type（人物/组织/事件/地点）

只返回JSON，不要其他内容。如果某项信息无法提取，设为null。
"""
        
        try:
            response = await self.llm_adapter.generate(
                prompt=prompt,
                system_prompt="你是一个专业的文本分析助手，擅长从回忆录中提取时间、地点和关键事件信息。",
                temperature=0.1,
                max_tokens=500
            )
            
            # 解析LLM响应
            import json
            result = json.loads(response.content)
            
            # 合并LLM提取的信息
            if result.get("year") and not context.year:
                context.year = str(result["year"])
            if result.get("month") and not context.month:
                context.month = str(result["month"])
            if result.get("location") and not context.location:
                context.location = result["location"]
            if result.get("keywords"):
                context.keywords = list(set(context.keywords + result["keywords"]))
            if result.get("entities"):
                context.entities = result["entities"]
                
        except Exception as e:
            # LLM解析失败时，使用规则提取的结果
            pass
        
        return context
    
    def _extract_year(self, text: str) -> Optional[str]:
        """提取年份"""
        for pattern in self.YEAR_PATTERNS:
            match = re.search(pattern, text)
            if match:
                year = match.group(1)
                # 处理两位数年份
                if len(year) == 2:
                    prefix = "19" if int(year) > 50 else "20"
                    year = prefix + year
                # 处理汉字年份
                elif not year.isdigit():
                    year = self._chinese_to_number(year)
                return year
        return None
    
    def _extract_month(self, text: str) -> Optional[str]:
        """提取月份"""
        for pattern in self.MONTH_PATTERNS:
            match = re.search(pattern, text)
            if match:
                month = match.group(1)
                if not month.isdigit():
                    month = self._chinese_to_number(month)
                return month
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """提取地点"""
        # 尝试匹配"在XX"、"到XX"、"来到XX"等模式
        location_patterns = [
            r'(?:在|到|来到|去|前往|抵达)\s*([^\s,，。！？\n]{2,10}(?:省|市|县|区|镇|村)?)',
            r'([^\s,，。！？\n]{2,6}(?:省|市|县|区))',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        # 检查是否包含知名城市
        for city in ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '重庆']:
            if city in text:
                return city
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = []
        
        # 历史事件相关关键词
        event_patterns = [
            r'(改革开放)',
            r'(经济特区)',
            r'(大学毕业)',
            r'(金融危机)',
            r'(南方谈话)',
            r'(下海创业)',
            r'(国企改革)',
            r'(市场经济)',
            r'(包分配)',
            r'(双向选择)',
        ]
        
        for pattern in event_patterns:
            if re.search(pattern, text):
                keywords.append(re.search(pattern, text).group(1))
        
        return keywords[:5]  # 最多返回5个关键词
    
    def _chinese_to_number(self, chinese: str) -> str:
        """汉字数字转阿拉伯数字"""
        mapping = {
            '零': '0', '〇': '0', '一': '1', '二': '2', '三': '3',
            '四': '4', '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10',
        }
        result = ""
        for char in chinese:
            if char in mapping:
                result += mapping[char]
        return result or chinese
