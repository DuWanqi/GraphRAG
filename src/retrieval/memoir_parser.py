"""
回忆录文本解析器
从回忆录文本中提取时间、地点、事件关键词
"""

import re
import jieba
import jieba.analyse
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
    
    # 常见城市名称（用于优先匹配）
    CITY_NAMES = [
        '北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都', '重庆',
        '天津', '苏州', '西安', '郑州', '长沙', '青岛', '沈阳', '大连',
        '佛山', '东莞', '宁波', '无锡', '合肥', '昆明', '厦门', '济南',
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
4. keywords: 3-5个关键词，要求：
   - 优先提取历史事件、机构、政策、重要概念（如"恢复高考"、"大学"、"入学"）
   - 避免提取叙事性词汇（如"梦想"、"憧憬"、"激动"）
   - 避免提取动词和形容词（如"发生变化"、"没睡"、"聊着"）
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

            # 去除可能的 markdown 代码块标记
            content = response.content.strip()
            if content.startswith("```"):
                # 移除开头的 ```json 或 ```
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # 移除结尾的 ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            result = json.loads(content)

            print(f"[DEBUG] LLM 提取结果: {result}")

            # 合并LLM提取的信息
            if result.get("year") and not context.year:
                context.year = str(result["year"])
            if result.get("month") and not context.month:
                context.month = str(result["month"])
            if result.get("location") and not context.location:
                context.location = result["location"]
            if result.get("keywords"):
                # 优先使用 LLM 提取的关键词（质量更高）
                # 只有在 LLM 关键词不足时才补充 TF-IDF 的结果
                llm_keywords = result["keywords"]
                if len(llm_keywords) >= 3:
                    context.keywords = llm_keywords[:5]  # 直接使用 LLM 结果
                else:
                    # LLM 关键词不足，补充 TF-IDF 结果
                    context.keywords = llm_keywords + [k for k in context.keywords if k not in llm_keywords]
                    context.keywords = context.keywords[:5]
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
                # 处理汉字年份（包括两位汉字如"七二"、"八八"）
                if not year.isdigit():
                    year = self._chinese_to_number(year)
                # 处理两位数年份
                if year.isdigit() and len(year) == 2:
                    prefix = "19" if int(year) > 50 else "20"
                    year = prefix + year
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
        """
        提取地点
        
        策略：
        1. 优先从文本中提取常见城市名称
        2. 然后使用正则匹配"在XX"、"到XX"等模式
        3. 最后使用jieba词性标注识别地名
        """
        import jieba.posseg as pseg
        
        # 策略1：优先匹配常见城市名称
        for city in self.CITY_NAMES:
            if city in text:
                # 检查是否是来源地（如"从广州来"）
                pattern = rf'从{city}(?:来|出发|离开)'
                if re.search(pattern, text):
                    continue  # 跳过来源地
                return city
        
        # 策略2：匹配"在XX"、"到XX"等模式（事件发生地）
        location_patterns = [
            r'(?:在|到|来到|去|前往|抵达|位于)\s*([^\s,，。！？\n]{2,6}(?:省|市|区|镇|村)?)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text)
            if match:
                loc = match.group(1)
                # 过滤掉不相关的词
                if not any(word in loc for word in ['老家', '家乡', '故乡', '屋里', '房子', '房间']):
                    # 如果匹配到的是区名，尝试找到对应的城市
                    if loc.endswith('区'):
                        for city in self.CITY_NAMES:
                            if city in text and loc in text:
                                return city
                    return loc
        
        # 策略3：使用jieba词性标注识别地名（ns表示地名）
        words = pseg.cut(text)
        locations = []
        for word, flag in words:
            if flag in ['ns', 'nsf']:  # ns=地名, nsf=音译地名
                # 过滤掉"老家"等词
                if word not in ['老家', '家乡', '故乡'] and '老家' not in word:
                    locations.append(word)
        
        if locations:
            return locations[0]
        
        return None
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（使用jieba TF-IDF + 过滤）
        
        过滤掉无意义的通用词，保留有检索价值的事件相关词
        """
        # 需要过滤的无意义词（包括地名、无检索价值的词）
        STOP_WORDS = {
            '广州', '深圳', '上海', '北京', '湖南', '四川', '湖北', '河南', '河北',
            '中国', '广东', '广西', '福建', '江西', '安徽', '江苏', '浙江',
            '天河', '白云', '越秀', '海珠', '番禺', '珠海', '东莞', '佛山',
            '香港', '澳门', '台湾', '老家', '家乡', '故乡',
            '到处', '处都', '那个', '什么', '怎么', '为什么', '哪里',
            '时候', '那年', '每天', '早上', '晚上', '周末', '年底', '春天', '夏天', '秋天', '冬天',
            '云吞面', '肠粉', '美食', '骑楼', '步行街', '图书馆',
            '老板', '同事', '员工', '公司', '客户', '外商',
            # 新增：人称代词和无意义词
            '我们', '他们', '你们', '我', '他', '她', '它', '这', '那', '此',
            '但', '但是', '却', '而', '和', '与', '及', '以及',
            '的', '了', '是', '在', '有', '不', '人', '都', '一', '上', '也',
            '很', '到', '说', '要', '去', '会', '着', '没有', '看', '好',
            '自己', '知道', '下', '还', '就', '把', '给', '做', '让', '能',
            '可以', '应该', '觉得', '时间', '事情', '东西', '问题', '方面',
            # 数字和无意义词
            '800', '100', '10', '5', '20', '30', '1', '2', '3', '4', '5',
            '但离', '不远', '心里', '充满', '干劲', '努力', '介绍', '产品',
            '意向', '表扬', '高兴', '温暖', '活力', '未来', '开始', '过来',
        }
        
        keywords = []
        
        # 使用jieba的TF-IDF提取关键词
        try:
            tfidf_keywords = jieba.analyse.extract_tags(text, topK=15, withWeight=False)
            for kw in tfidf_keywords:
                # 过滤条件：不在停用词中，长度>=2，不是纯数字
                if kw not in STOP_WORDS and len(kw) >= 2 and not kw.isdigit():
                    keywords.append(kw)
        except Exception:
            pass
        
        # 补充：基于规则提取重要事件相关词（这些词应该优先）
        event_keywords = [
            '外贸', '出口', '金融危机', '亚运会', '亚运', '广交会', '交易会',
            '经济', '创业', '改革', '开放', '建设', '发展', '投资', '产业',
            '制造业', '电子', '纺织', '服装', '科技', '互联网', '电商',
            '创业板', '股票', '金融', '房地产', '楼市', '政策', '政策变化',
            '地铁', '交通', '基建', '城市建设', '环境', '环保', '污染',
        ]
        
        for kw in event_keywords:
            if kw in text and kw not in keywords and kw not in STOP_WORDS:
                keywords.append(kw)
        
        # 去重并保留前10个关键词
        keywords = list(dict.fromkeys(keywords))[:10]
        
        return keywords
    
    def _chinese_to_number(self, chinese_num: str) -> str:
        """
        将中文数字转换为阿拉伯数字
        
        Args:
            chinese_num: 中文数字字符串（如"八八"、"二零"）
            
        Returns:
            str: 阿拉伯数字字符串
        """
        digit_map = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
        }
        
        result = ''
        for char in chinese_num:
            if char in digit_map:
                result += digit_map[char]
            else:
                result += char
        
        return result
