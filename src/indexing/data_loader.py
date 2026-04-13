"""
历史数据加载器
支持多种数据格式的加载和预处理
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class HistoricalEvent:
    """历史事件数据结构"""
    title: str
    date: str
    location: str
    content: str
    keywords: Optional[List[str]] = None
    source: Optional[str] = None
    
    def to_text(self) -> str:
        """转换为文本格式"""
        parts = [
            f"事件标题：{self.title}",
            f"时间：{self.date}",
            f"地点：{self.location}",
            "",
            self.content,
        ]
        if self.keywords:
            parts.append(f"\n关键词：{', '.join(self.keywords)}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "title": self.title,
            "date": self.date,
            "location": self.location,
            "content": self.content,
            "keywords": self.keywords,
            "source": self.source,
        }


class DataLoader:
    """
    历史数据加载器
    
    支持的格式：
    - JSON: 结构化的历史事件数据
    - CSV: 表格形式的历史事件数据
    - TXT: 纯文本格式的历史事件数据
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir) if data_dir else Path("./data/input")
    
    def load_json(self, filepath: str) -> List[HistoricalEvent]:
        """
        从JSON文件加载历史事件
        
        JSON格式示例：
        [
            {
                "title": "事件标题",
                "date": "1988年",
                "location": "深圳",
                "content": "事件描述...",
                "keywords": ["关键词1", "关键词2"]
            }
        ]
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        events = []
        for item in data:
            event = HistoricalEvent(
                title=item.get("title", "未命名事件"),
                date=item.get("date", "未知时间"),
                location=item.get("location", "未知地点"),
                content=item.get("content", ""),
                keywords=item.get("keywords"),
                source=item.get("source"),
            )
            events.append(event)
        
        return events
    
    def load_csv(self, filepath: str) -> List[HistoricalEvent]:
        """
        从CSV文件加载历史事件
        
        CSV格式要求：
        - 必须包含列：title, date, location, content
        - 可选列：keywords (逗号分隔), source
        """
        events = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                keywords = None
                if row.get("keywords"):
                    keywords = [k.strip() for k in row["keywords"].split(",")]
                
                event = HistoricalEvent(
                    title=row.get("title", "未命名事件"),
                    date=row.get("date", "未知时间"),
                    location=row.get("location", "未知地点"),
                    content=row.get("content", ""),
                    keywords=keywords,
                    source=row.get("source"),
                )
                events.append(event)
        
        return events
    
    def load_txt(self, filepath: str) -> List[HistoricalEvent]:
        """
        从TXT文件加载历史事件
        
        TXT格式：事件之间用 "---" 分隔
        每个事件包含：
        - 事件标题：xxx
        - 时间：xxx
        - 地点：xxx
        - 正文内容
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按分隔符拆分事件
        event_texts = content.split("---")
        events = []
        
        for text in event_texts:
            text = text.strip()
            if not text:
                continue
            
            event = self._parse_txt_event(text)
            if event:
                events.append(event)
        
        return events
    
    def _parse_txt_event(self, text: str) -> Optional[HistoricalEvent]:
        """解析单个TXT格式的事件"""
        lines = text.split("\n")
        
        title = "未命名事件"
        date = "未知时间"
        location = "未知地点"
        content_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith("事件标题："):
                title = line[5:].strip()
            elif line.startswith("时间："):
                date = line[3:].strip()
            elif line.startswith("地点："):
                location = line[3:].strip()
            else:
                content_lines.append(line)
        
        content = "\n".join(content_lines).strip()
        
        if not content and title == "未命名事件":
            return None
        
        return HistoricalEvent(
            title=title,
            date=date,
            location=location,
            content=content or text,
        )
    
    def load_all(self) -> List[HistoricalEvent]:
        """
        加载数据目录中的所有历史事件文件
        
        Returns:
            所有加载的历史事件列表
        """
        all_events = []
        
        if not self.data_dir.exists():
            return all_events
        
        # 加载JSON文件
        for json_file in self.data_dir.glob("*.json"):
            try:
                events = self.load_json(str(json_file))
                all_events.extend(events)
            except Exception as e:
                print(f"加载 {json_file} 失败: {e}")
        
        # 加载CSV文件
        for csv_file in self.data_dir.glob("*.csv"):
            try:
                events = self.load_csv(str(csv_file))
                all_events.extend(events)
            except Exception as e:
                print(f"加载 {csv_file} 失败: {e}")
        
        # 加载TXT文件
        for txt_file in self.data_dir.glob("*.txt"):
            try:
                events = self.load_txt(str(txt_file))
                all_events.extend(events)
            except Exception as e:
                print(f"加载 {txt_file} 失败: {e}")
        
        return all_events
    
    def save_as_graphrag_input(
        self,
        events: List[HistoricalEvent],
        output_dir: Optional[str] = None
    ) -> int:
        """
        将历史事件保存为GraphRAG输入格式
        
        Args:
            events: 历史事件列表
            output_dir: 输出目录（默认为数据目录）
            
        Returns:
            保存的文件数量
        """
        out_dir = Path(output_dir) if output_dir else self.data_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for i, event in enumerate(events):
            filename = f"event_{i:04d}.txt"
            filepath = out_dir / filename
            filepath.write_text(event.to_text(), encoding="utf-8")
        
        return len(events)
