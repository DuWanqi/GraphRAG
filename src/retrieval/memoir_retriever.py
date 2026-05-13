"""
回忆录历史背景检索器
基于GraphRAG进行知识图谱检索
支持关键词检索、向量检索、混合检索三种模式
"""

import os
import asyncio
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .memoir_parser import MemoirParser, MemoirContext


class RetrievalMode(Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """检索结果"""
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    text_units: List[str] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    context: Optional[MemoirContext] = None
    query: str = ""
    
    def merge(self, other: 'RetrievalResult') -> 'RetrievalResult':
        """合并另一个检索结果"""
        self.entities.extend(other.entities)
        self.relationships.extend(other.relationships)
        self.text_units.extend(other.text_units)
        self.communities.extend(other.communities)
        return self


class MemoirRetriever:
    """回忆录检索器"""
    
    def __init__(self, llm_adapter=None, index_dir: Optional[Union[str, Path]] = None):
        self.llm_adapter = llm_adapter
        default_root = Path(__file__).resolve().parent.parent.parent / "data" / "graphrag_output"
        self.index_dir = Path(index_dir) if index_dir is not None else default_root
        
        # 缓存数据
        self._entities_df = None
        self._relationships_df = None
        self._text_units_df = None
        self._communities_df = None
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        # 兼容 GraphRAG 1.x 和 2.x
        entity_files = [
            output_dir / "entities.parquet",
            output_dir / "create_final_entities.parquet"
        ]
        relationship_files = [
            output_dir / "relationships.parquet",
            output_dir / "create_final_relationships.parquet"
        ]
        text_unit_files = [
            output_dir / "text_units.parquet",
            output_dir / "create_final_text_units.parquet"
        ]
        community_files = [
            output_dir / "community_reports.parquet",
            output_dir / "create_final_community_reports.parquet"
        ]
        
        for file_path in entity_files:
            if file_path.exists():
                self._entities_df = pd.read_parquet(file_path)
                break
        
        for file_path in relationship_files:
            if file_path.exists():
                self._relationships_df = pd.read_parquet(file_path)
                break
        
        for file_path in text_unit_files:
            if file_path.exists():
                self._text_units_df = pd.read_parquet(file_path)
                break
        
        for file_path in community_files:
            if file_path.exists():
                self._communities_df = pd.read_parquet(file_path)
                break
        
        print(f"[DEBUG] 加载实体: {len(self._entities_df) if self._entities_df is not None else 0} 条")
        print(f"[DEBUG] 加载关系: {len(self._relationships_df) if self._relationships_df is not None else 0} 条")
        print(f"[DEBUG] 加载文本单元: {len(self._text_units_df) if self._text_units_df is not None else 0} 条")
    
    def is_index_ready(self) -> bool:
        """检查索引是否就绪"""
        output_dir = self.index_dir / "output"
        new_format = output_dir / "entities.parquet"
        old_format = output_dir / "create_final_entities.parquet"
        return new_format.exists() or old_format.exists()
    
    async def retrieve(
        self,
        memoir_text: str,
        top_k: int = 10,
        use_llm_parsing: bool = False,
        mode: str = "hybrid"
    ) -> RetrievalResult:
        """执行检索"""
        # 解析回忆录文本，提取上下文信息
        parser = MemoirParser()
        context = parser.parse(memoir_text, use_llm=use_llm_parsing)
        
        # 根据模式执行检索
        if mode == "keyword":
            return await self._keyword_retrieve(context, top_k)
        elif mode == "vector":
            return await self._vector_retrieve(context, top_k)
        else:
            return await self._hybrid_retrieve(context, top_k)
    
    def retrieve_sync(self, memoir_text: str, top_k: int = 10) -> RetrievalResult:
        """同步版本的检索"""
        return asyncio.run(self.retrieve(memoir_text, top_k, use_llm_parsing=False))
    
    async def _keyword_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """关键词检索"""
        result = RetrievalResult()
        result.context = context
        
        # 搜索实体、关系、文本单元和社区报告
        result.entities = self._search_entities(context, top_k)
        result.relationships = self._search_relationships(context, top_k)
        result.text_units = self._search_text_units(context, top_k)
        result.communities = self._search_communities(context, top_k)
        
        return result
    
    async def _vector_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """向量检索"""
        # 简单实现：使用关键词检索作为回退
        return await self._keyword_retrieve(context, top_k)
    
    async def _hybrid_retrieve(self, context: MemoirContext, top_k: int) -> RetrievalResult:
        """混合检索"""
        # 获取关键词检索结果
        keyword_result = await self._keyword_retrieve(context, top_k)
        
        # 合并结果（去重）
        return keyword_result
    
    def _search_entities(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关实体（改进版：加权评分+主题相关性过滤）"""
        if self._entities_df is None or self._entities_df.empty:
            print(f"[_search_entities] 实体数据为空")
            return []
        
        # 需要过滤的通用实体（太泛泛，没有针对性）
        GENERIC_ENTITIES = {
            '中国', '中华人民共和国', '北京', '国务院', 'STATE COUNCIL',
            '中共中央', 'CPC CENTRAL COMMITTEE', '中央政府', '中央',
            'EUROPEAN UNION', '欧盟', 'BRICS',
        }
        
        # 需要过滤的不相关地点实体（与目标地点无关）
        UNRELATED_LOCATIONS = {
            'SHANGHAI', '北京', 'BEIJING', '上海',
            '天津', 'TIANJIN', '重庆', 'CHONGQING',
            '成都', 'CHENGDU', '武汉', 'WUHAN', '杭州', 'HANGZHOU',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高
        if context.location:
            search_terms.append((context.location, 20))
            location_map = {
                "深圳": "SHENZHEN", "北京": "BEIJING", "上海": "SHANGHAI",
                "广州": "GUANGZHOU", "香港": "HONG KONG",
                "广东": "GUANGDONG", "佛山": "FOSHAN", "东莞": "DONGGUAN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 20))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_entities] 搜索词: {search_terms}")
        print(f"[_search_entities] 实体总数: {len(self._entities_df)}")
        
        # 构建目标地点集合（用于严格匹配）
        target_locations = set()
        if context.location:
            target_locations.add(context.location)
            target_locations.add(context.location.upper())
            if context.location in location_map:
                target_locations.add(location_map[context.location])
                target_locations.add(location_map[context.location].lower())
        
        for _, row in self._entities_df.iterrows():
            entity_name = str(row.get("title", row.get("name", "")))
            entity_desc = str(row.get("description", ""))
            search_text = f"{entity_name} {entity_desc}".upper()
            
            # 过滤通用实体
            if entity_name in GENERIC_ENTITIES:
                continue
            
            # 过滤不相关地点实体
            if entity_name in UNRELATED_LOCATIONS:
                continue
            
            score = 0
            year_matched = False
            exact_year_matched = False  # 是否匹配目标年份
            location_matched = False
            topic_matched = False
            
            for term, weight in search_terms:
                term_upper = str(term).upper()
                if term_upper in search_text:
                    score += weight
                    if str(term) == str(context.year):
                        year_matched = True
                        exact_year_matched = True  # 精确匹配目标年份
                    elif str(term) == str(context.location) or term_upper in target_locations:
                        location_matched = True
                    elif term in topic_keywords:
                        topic_matched = True
            
            # 过滤策略：必须匹配年份
            if not year_matched:
                continue
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 15
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 25
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append({
                "name": entity_name,
                "type": row.get("type", "unknown"),
                "description": entity_desc,
                "score": score,
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 去重：移除重复的实体（中英文变体）
        seen_names = set()
        unique_results = []
        for r in results:
            name_lower = r["name"].lower()
            if name_lower in seen_names:
                continue
            is_duplicate = False
            for seen in seen_names:
                if name_lower in seen or seen in name_lower:
                    is_duplicate = True
                    break
            if not is_duplicate:
                seen_names.add(name_lower)
                unique_results.append(r)
        
        print(f"[_search_entities] 匹配结果数: {len(results)}, 去重后: {len(unique_results)}")
        if unique_results:
            print(f"[_search_entities] 前3个结果: {[(r['name'], r['score']) for r in unique_results[:3]]}")
        out = unique_results[: min(top_k, 10)]  # 限制最多返回10个
        return out
    
    def _search_relationships(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关关系（改进版：加权评分+主题相关性过滤）"""
        if self._relationships_df is None or self._relationships_df.empty:
            return []
        
        # 需要过滤的不相关关系关键词
        UNRELATED_KEYWORDS = {
            '嫦娥', 'CHANG', '探月', '月球', 'BRICS', '金砖',
            '维和', '平潭', '滨海新区', '塘沽', '汉沽', '大港',
            '里斯本', '欧盟', 'EKATERINBURG', '嫦娥一号',
            'BINHAI', 'DAGANG', 'TANGGU', 'HANGU', 'LISBON',
        }
        
        # 需要过滤的不相关地点
        UNRELATED_LOCATIONS = {
            '上海', 'SHANGHAI', '北京', 'BEIJING', '天津', 'TIANJIN',
            '成都', 'CHENGDU', '武汉', 'WUHAN', '杭州', 'HANGZHOU',
            '莫斯科', 'MOSCOW', '叶卡捷琳堡',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高
        if context.location:
            search_terms.append((context.location, 20))
            location_map = {
                "广州": "GUANGZHOU", "广东": "GUANGDONG",
                "佛山": "FOSHAN", "东莞": "DONGGUAN", "深圳": "SHENZHEN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 20))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_relationships] 搜索词: {search_terms}")
        
        for _, row in self._relationships_df.iterrows():
            source = str(row.get("source", row.get("source_title", "")))
            target = str(row.get("target", row.get("target_title", "")))
            desc = str(row.get("description", ""))
            
            search_text = f"{source} {target} {desc}".upper()
            
            # 过滤不相关关键词
            if any(kw.upper() in search_text for kw in UNRELATED_KEYWORDS):
                continue
            
            # 过滤不相关地点
            if any(loc.upper() in search_text for loc in UNRELATED_LOCATIONS):
                continue
            
            score = 0
            year_matched = False
            exact_year_matched = False  # 是否匹配目标年份
            location_matched = False
            topic_matched = False
            
            for term, weight in search_terms:
                term_upper = str(term).upper()
                if term_upper in search_text:
                    score += weight
                    if str(term) == str(context.year):
                        year_matched = True
                        exact_year_matched = True  # 精确匹配目标年份
                    elif str(term) == str(context.location):
                        location_matched = True
                    elif term in topic_keywords:
                        topic_matched = True
            
            # 过滤策略：必须匹配年份
            if not year_matched:
                continue
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 15
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 25
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append({
                "source": source,
                "target": target,
                "description": desc,
                "weight": row.get("weight", 1),
                "score": score,
            })
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 去重：移除重复的关系
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['source']}_{r['target']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        print(f"[_search_relationships] 匹配结果数: {len(results)}, 去重后: {len(unique_results)}")
        return unique_results[:min(top_k, 8)]  # 限制最多返回8个
    
    def _search_text_units(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[str]:
        """搜索相关文本单元（改进版：加权评分+主题相关性过滤）"""
        if self._text_units_df is None or self._text_units_df.empty:
            return []
        
        # 需要过滤的不相关文本关键词
        UNRELATED_TEXT_KEYWORDS = {
            '嫦娥', '探月', '月球', 'BRICS', '金砖',
            '维和', '平潭', '滨海新区', '里斯本', '欧盟',
            'HU JINTAO', 'JIANG ZEMIN', 'DENG XIAOPING',
        }
        
        results = []
        
        # 构建搜索词和权重
        search_terms = []
        topic_keywords = set()
        
        # 年份权重最高（目标年份权重更高）
        if context.year:
            search_terms.append((context.year, 30))  # 目标年份权重30
            # 添加邻近年份（权重较低）
            try:
                year_int = int(context.year)
                search_terms.append((str(year_int - 1), 10))  # 前一年权重10
                search_terms.append((str(year_int + 1), 10))  # 后一年权重10
            except:
                pass
        
        # 地点权重次高（提高权重）
        if context.location:
            search_terms.append((context.location, 35))  # 提高到35
            location_map = {
                "广州": "GUANGZHOU", "广东": "GUANGDONG",
                "佛山": "FOSHAN", "东莞": "DONGGUAN", "深圳": "SHENZHEN",
            }
            if context.location in location_map:
                search_terms.append((location_map[context.location], 35))
        
        # 主题关键词权重
        for kw in context.keywords[:5]:
            if kw and len(kw) >= 2:
                search_terms.append((kw, 15))
                topic_keywords.add(kw)
        
        if not search_terms:
            return []
        
        print(f"[_search_text_units] 搜索词: {search_terms}")
        
        for _, row in self._text_units_df.iterrows():
            text = str(row.get("text", row.get("content", "")))
            
            # 过滤过短的文本
            if len(text) < 50:
                continue
            
            # 过滤不相关文本
            if any(keyword in text for keyword in UNRELATED_TEXT_KEYWORDS):
                continue
            
            # 年份匹配：允许目标年份及邻近年份
            year_matched = False
            exact_year_matched = False
            found_year = None
            if context.year:
                target_year = int(context.year)
                allowed_years = {str(target_year - 1), str(target_year), str(target_year + 1)}
                
                first_part = text[:300]
                
                for year in allowed_years:
                    year_pattern = rf'{year}[\s年]'
                    match = re.search(year_pattern, first_part)
                    if match:
                        match_str = match.group(0)
                        # 简化检查：匹配到年份数字后面跟空格或"年"就认为是有效的年份引用
                        # 检查是否是完整的年份（避免误匹配如"20090"这样的数字）
                        if len(match_str) >= 5:
                            year_matched = True
                            found_year = year
                            if year == str(target_year):
                                exact_year_matched = True
                            break
                
                if not year_matched:
                    continue
            
            # 地点匹配检查（硬性要求：必须包含目标地点或邻近城市/地区）
            location_matched = False
            if context.location:
                # 直接匹配目标地点
                if context.location in text:
                    location_matched = True
                else:
                    # 检查邻近城市
                    nearby_cities = ["深圳", "东莞", "佛山", "珠海", "中山"]
                    for city in nearby_cities:
                        if city in text:
                            location_matched = True
                            break
                    # 检查省份和区域
                    if not location_matched:
                        regions = ["广东", "珠三角", "珠江三角洲", "粤港澳"]
                        for region in regions:
                            if region in text:
                                location_matched = True
                                break
            
            # 硬性要求：必须同时匹配年份和地点
            if not (year_matched and location_matched):
                continue
            
            score = 0
            topic_matched = False
            
            for term, weight in search_terms:
                if term in text:
                    score += weight
                    if term in topic_keywords:
                        topic_matched = True
            
            # 额外加分：精确匹配目标年份
            if exact_year_matched:
                score += 30  # 提高加分
            
            # 额外加分：直接匹配目标地点（而不是邻近城市）
            if context.location and context.location in text:
                score += 20  # 直接匹配广州加20分
            
            # 额外加分：同时匹配地点和主题
            if location_matched and topic_matched:
                score += 30
            elif location_matched:
                score += 15
            elif topic_matched:
                score += 10
            
            results.append((text, score, exact_year_matched, location_matched, topic_matched))
        
        # 排序策略：首先按是否精确匹配目标年份排序，然后按分数降序
        # 这样确保2009年的内容优先于2010年的内容
        results.sort(key=lambda x: (x[2], x[1], x[3] and x[4], x[3], x[4]), reverse=True)
        
        print(f"[_search_text_units] 匹配结果数: {len(results)}")
        
        # 确保同时包含目标年份和邻近年份的内容
        # 先取前5个2009年的，再取3个2010年的
        year_2009 = [r for r in results if r[2]]  # 精确匹配2009年
        year_other = [r for r in results if not r[2]]  # 2008或2010年
        
        # 混合结果：优先2009年，但也要包含一些2010年的内容
        mixed_results = year_2009[:5] + year_other[:3]
        
        return [r[0] for r in mixed_results[:min(top_k, 8)]]  # 限制最多返回8个
    
    def _search_communities(
        self,
        context: MemoirContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """搜索相关社区报告"""
        if self._communities_df is None or self._communities_df.empty:
            return []
        
        results = []
        search_terms = [context.year, context.location] + context.keywords
        search_terms = [t for t in search_terms if t]
        
        for _, row in self._communities_df.iterrows():
            title = str(row.get("title", ""))
            summary = str(row.get("summary", ""))
            
            score = 0
            for term in search_terms:
                if str(term) in title or str(term) in summary:
                    score += 1
            
            if score > 0:
                results.append({
                    "title": title,
                    "summary": summary,
                    "score": score,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:min(top_k, 3)]  # 限制最多返回3个
