"""
检索策略对比评测模块
用于对比不同检索策略的效果，生成实验报告

不修改原有代码，独立运行评测实验
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime

import pandas as pd

from ..config import get_settings
from ..retrieval import MemoirRetriever, RetrievalResult, MemoirContext, VectorRetriever
from ..llm import create_llm_adapter


@dataclass
class TestCase:
    """测试用例"""
    query_id: str
    memoir_text: str
    ground_truth_entities: List[str] = field(default_factory=list)
    ground_truth_time: Optional[str] = None
    ground_truth_location: Optional[str] = None
    query_type: str = "simple"  # simple, multi_hop, temporal


@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    hit_at_k: Dict[int, bool] = field(default_factory=dict)
    mrr: float = 0.0
    latency_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """评测结果"""
    strategy_name: str
    total_queries: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_latency_ms: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_queries": self.total_queries,
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "hit_at_3": round(self.hit_at_3, 4),
            "hit_at_5": round(self.hit_at_5, 4),
            "hit_at_10": round(self.hit_at_10, 4),
        }


class RetrievalBenchmark:
    """
    检索策略对比评测器
    
    支持对比的检索策略：
    1. keyword_only: 仅关键词匹配（当前实现）
    2. vector_only: 仅向量检索
    3. hybrid: 混合策略（关键词+向量融合）
    4. entity_first: 实体优先检索
    5. community_first: 社区报告优先
    """
    
    CN_EN_MAP = {
        "深圳": "SHENZHEN",
        "上海": "SHANGHAI",
        "北京": "BEIJING",
        "广州": "GUANGZHOU",
        "香港": "HONG KONG",
        "中国": "CHINA",
        "邓小平": "DENG XIAOPING",
        "蛇口": "SHEKOU",
        "经济特区": "SPECIAL ECONOMIC ZONE",
        "改革开放": "REFORM AND OPENING",
    }
    
    EN_CN_MAP = {v: k for k, v in CN_EN_MAP.items()}
    
    ENTITY_NAME_MAP = {
        "深圳": ["SHENZHEN", "深圳", "shenzhen"],
        "邓小平": ["DENG XIAOPING", "邓小平", "deng xiaoping"],
        "上海": ["SHANGHAI", "上海", "shanghai"],
        "北京": ["BEIJING", "北京", "beijing"],
        "香港": ["HONG KONG", "香港", "hong kong"],
        "浦东新区": ["PUDONG", "浦东", "pudong"],
        "经济特区": ["SPECIAL ECONOMIC ZONE", "经济特区", "SEZ"],
        "改革开放": ["REFORM AND OPENING", "改革开放", "reform"],
        "奥运会": ["OLYMPICS", "奥运会", "olympic"],
        "世博会": ["WORLD EXPO", "世博会", "expo"],
        "WTO": ["WTO", "世界贸易组织", "WORLD TRADE ORGANIZATION"],
        "南巡": ["SOUTHERN TOUR", "南巡", "southern tour"],
        "回归": ["RETURN", "回归", "handover"],
    }
    
    DEFAULT_TEST_CASES = [
        TestCase(
            query_id="001",
            memoir_text="1988年夏天，我从大学毕业，怀揣着梦想来到了深圳。",
            ground_truth_entities=["深圳", "SHENZHEN"],
            ground_truth_time="1988",
            ground_truth_location="深圳",
            query_type="simple"
        ),
        TestCase(
            query_id="002",
            memoir_text="1992年，邓小平南巡讲话后，深圳迎来了新的发展机遇。",
            ground_truth_entities=["邓小平", "DENG XIAOPING", "深圳"],
            ground_truth_time="1992",
            ground_truth_location="深圳",
            query_type="simple"
        ),
        TestCase(
            query_id="003",
            memoir_text="深圳蛇口工业区是中国改革开放的试验田。",
            ground_truth_entities=["SHENZHEN SHEKOU INDUSTRIAL ZONE", "蛇口"],
            ground_truth_time=None,
            ground_truth_location="深圳",
            query_type="simple"
        ),
        TestCase(
            query_id="004",
            memoir_text="全国人大常委会通过了关于经济特区的决议。",
            ground_truth_entities=["NATIONAL PEOPLE'S CONGRESS STANDING COMMITTEE"],
            ground_truth_time=None,
            ground_truth_location=None,
            query_type="simple"
        ),
        TestCase(
            query_id="005",
            memoir_text="亚洲四小龙的经济腾飞给中国带来了启示。",
            ground_truth_entities=["ASIA", "KOREA", "THAILAND"],
            ground_truth_time=None,
            ground_truth_location=None,
            query_type="multi_hop"
        ),
    ]
    
    def __init__(
        self,
        index_dir: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        llm_provider: str = "gemini",
    ):
        """
        初始化评测器
        
        Args:
            index_dir: GraphRAG索引目录
            test_cases: 测试用例列表（默认使用内置用例）
            llm_provider: 用于向量检索的LLM提供商
        """
        settings = get_settings()
        self.index_dir = Path(index_dir or settings.graphrag_output_dir)
        self.test_cases = test_cases or self.DEFAULT_TEST_CASES
        self.retriever = MemoirRetriever(index_dir=str(self.index_dir))
        
        try:
            llm_adapter = create_llm_adapter(provider=llm_provider)
            self.vector_retriever = VectorRetriever(
                index_dir=str(self.index_dir),
                llm_adapter=llm_adapter
            )
        except Exception as e:
            print(f"[Warning] 无法初始化向量检索器: {e}")
            self.vector_retriever = None
        
        self._load_index_data()
    
    def _load_index_data(self):
        """加载索引数据"""
        output_dir = self.index_dir / "output"
        
        entities_file = output_dir / "entities.parquet"
        if not entities_file.exists():
            entities_file = output_dir / "create_final_entities.parquet"
        if entities_file.exists():
            self._entities_df = pd.read_parquet(entities_file)
        else:
            self._entities_df = None
        
        communities_file = output_dir / "community_reports.parquet"
        if not communities_file.exists():
            communities_file = output_dir / "create_final_community_reports.parquet"
        if communities_file.exists():
            self._communities_df = pd.read_parquet(communities_file)
        else:
            self._communities_df = None
        
        text_file = output_dir / "text_units.parquet"
        if not text_file.exists():
            text_file = output_dir / "create_final_text_units.parquet"
        if text_file.exists():
            self._text_units_df = pd.read_parquet(text_file)
        else:
            self._text_units_df = None
        
        self._generate_test_cases_from_index()
    
    def _generate_test_cases_from_index(self):
        """基于索引中实际存在的实体生成测试用例"""
        if self._entities_df is None or len(self._entities_df) == 0:
            return
        
        generated_cases = []
        
        geo_entities = self._entities_df[self._entities_df['type'] == 'GEO']['title'].tolist()
        person_entities = self._entities_df[self._entities_df['type'] == 'PERSON']['title'].tolist()
        event_entities = self._entities_df[self._entities_df['type'] == 'EVENT']['title'].tolist()
        
        entity_name_map = {
            "SHENZHEN": {"cn": "深圳", "time": "1988"},
            "DENG XIAOPING": {"cn": "邓小平", "time": "1992"},
            "BEIJING": {"cn": "北京", "time": "2008"},
            "SHANGHAI": {"cn": "上海", "time": "2010"},
            "HUAWEI": {"cn": "华为", "time": "1987"},
        }
        
        for entity in geo_entities[:3]:
            info = entity_name_map.get(entity, {"cn": entity, "time": None})
            generated_cases.append(TestCase(
                query_id=f"geo_{entity[:10]}",
                memoir_text=f"{info['cn']}是我人生中重要的地方。",
                ground_truth_entities=[entity],
                ground_truth_time=info['time'],
                ground_truth_location=info['cn'],
                query_type="simple"
            ))
        
        for entity in person_entities[:2]:
            info = entity_name_map.get(entity, {"cn": entity, "time": None})
            generated_cases.append(TestCase(
                query_id=f"person_{entity[:10]}",
                memoir_text=f"{info['cn']}的事迹深深影响了我。",
                ground_truth_entities=[entity],
                ground_truth_time=info['time'],
                ground_truth_location=None,
                query_type="simple"
            ))
        
        for entity in event_entities[:2]:
            generated_cases.append(TestCase(
                query_id=f"event_{entity[:10]}",
                memoir_text=f"{entity}是那个时代的重要事件。",
                ground_truth_entities=[entity],
                ground_truth_time=None,
                ground_truth_location=None,
                query_type="simple"
            ))
        
        if generated_cases:
            self.test_cases = generated_cases
    
    def is_ready(self) -> bool:
        """检查评测是否就绪"""
        return self._entities_df is not None
    
    def _normalize_entity_name(self, name: str) -> List[str]:
        """获取实体的所有可能名称变体"""
        name_lower = name.lower()
        variants = [name, name_lower, name.upper()]
        
        for cn_name, en_variants in self.ENTITY_NAME_MAP.items():
            if name in en_variants or name_lower in [v.lower() for v in en_variants]:
                variants.extend(en_variants)
                break
        
        return list(set(variants))

    def _calculate_metrics(
        self,
        retrieved_entities: List[Dict[str, Any]],
        ground_truth: List[str],
        latency_ms: float,
    ) -> RetrievalMetrics:
        """计算检索指标"""
        metrics = RetrievalMetrics(latency_ms=latency_ms)
        
        if not ground_truth:
            return metrics
        
        retrieved_names = []
        for e in retrieved_entities:
            name = e.get("name", "")
            retrieved_names.extend(self._normalize_entity_name(name))
        retrieved_names = list(set(retrieved_names))
        
        ground_truth_variants = []
        for gt in ground_truth:
            ground_truth_variants.extend(self._normalize_entity_name(gt))
        ground_truth_variants = list(set(v.lower() for v in ground_truth_variants))
        
        retrieved_set = set(n.lower() for n in retrieved_names)
        ground_truth_set = set(ground_truth_variants)
        
        if retrieved_set:
            metrics.precision = len(retrieved_set & ground_truth_set) / len(retrieved_set)
        
        if ground_truth_set:
            metrics.recall = len(retrieved_set & ground_truth_set) / len(ground_truth_set)
        
        if metrics.precision + metrics.recall > 0:
            metrics.f1 = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
        
        for k in [3, 5, 10]:
            top_k = retrieved_names[:k]
            top_k_lower = [n.lower() for n in top_k]
            metrics.hit_at_k[k] = any(gt.lower() in top_k_lower for gt in ground_truth)
        
        for i, name in enumerate(retrieved_names):
            if name.lower() in ground_truth_variants:
                metrics.mrr = 1.0 / (i + 1)
                break
        
        return metrics
    
    def _strategy_keyword_only(self, context: MemoirContext, top_k: int) -> List[Dict[str, Any]]:
        """策略1：仅关键词匹配（当前实现）"""
        if self._entities_df is None:
            return []
        
        results = []
        keywords = context.keywords if context.keywords else []
        location = context.location or ""
        
        # 扩展关键词：添加中英文映射
        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.append(kw)
            if kw in self.CN_EN_MAP:
                expanded_keywords.append(self.CN_EN_MAP[kw])
            if kw.upper() in self.EN_CN_MAP:
                expanded_keywords.append(self.EN_CN_MAP[kw.upper()])
        
        # 扩展location
        expanded_location = location
        if location in self.CN_EN_MAP:
            expanded_location = self.CN_EN_MAP[location]
        
        for _, row in self._entities_df.iterrows():
            entity_name = str(row.get("title", ""))
            entity_desc = str(row.get("description", ""))
            
            score = 0
            for kw in expanded_keywords:
                if kw.lower() in entity_name.lower() or kw.lower() in entity_desc.lower():
                    score += 1
            
            if location and (location.lower() in entity_name.lower() or expanded_location.lower() in entity_name.lower()):
                score += 2
            
            if score > 0:
                results.append({
                    "name": entity_name,
                    "type": row.get("type", "unknown"),
                    "description": entity_desc[:100],
                    "score": score,
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def _strategy_entity_first(self, context: MemoirContext, top_k: int) -> List[Dict[str, Any]]:
        """策略2：实体优先，优先返回人物和组织类型实体"""
        base_results = self._strategy_keyword_only(context, top_k * 2)
        
        persons = [e for e in base_results if e.get("type") in ["PERSON", "人物"]]
        orgs = [e for e in base_results if e.get("type") in ["ORGANIZATION", "组织"]]
        others = [e for e in base_results if e not in persons and e not in orgs]
        
        return (persons + orgs + others)[:top_k]
    
    def _strategy_community_first(self, context: MemoirContext, top_k: int) -> List[Dict[str, Any]]:
        """策略3：社区报告优先，从社区中提取实体"""
        community_entities = []
        
        if self._communities_df is not None and len(self._communities_df) > 0:
            keywords = context.keywords if context.keywords else []
            
            for _, row in self._communities_df.iterrows():
                content = str(row.get("full_content", row.get("summary", "")))
                
                for kw in keywords:
                    if kw.lower() in content.lower():
                        if self._entities_df is not None:
                            for _, entity_row in self._entities_df.iterrows():
                                entity_name = str(entity_row.get("title", ""))
                                if entity_name in content:
                                    if entity_name not in [e.get("name") for e in community_entities]:
                                        community_entities.append({
                                            "name": entity_name,
                                            "type": entity_row.get("type", "unknown"),
                                            "description": str(entity_row.get("description", ""))[:100],
                                            "score": 1.0,
                                            "source": "community"
                                        })
                        break
        
        entity_results = self._strategy_keyword_only(context, top_k)
        
        seen = set(e.get("name") for e in community_entities)
        for e in entity_results:
            if e.get("name") not in seen:
                community_entities.append(e)
        
        return community_entities[:top_k]
    
    def _strategy_hybrid(self, context: MemoirContext, top_k: int) -> List[Dict[str, Any]]:
        """策略4：混合策略，融合多源检索结果"""
        entity_results = self._strategy_keyword_only(context, top_k)
        
        entity_scores = {}
        for i, e in enumerate(entity_results):
            name = e.get("name", "")
            entity_scores[name] = e.get("score", 0) + (top_k - i) * 0.1
        
        if self._communities_df is not None and len(self._communities_df) > 0:
            keywords = context.keywords if context.keywords else []
            
            for _, row in self._communities_df.head(3).iterrows():
                content = str(row.get("full_content", row.get("summary", "")))
                community_score = 0
                for kw in keywords:
                    if kw.lower() in content.lower():
                        community_score += 0.5
                
                if self._entities_df is not None and community_score > 0:
                    for _, entity_row in self._entities_df.iterrows():
                        entity_name = str(entity_row.get("title", ""))
                        if entity_name in content:
                            if entity_name in entity_scores:
                                entity_scores[entity_name] += community_score
                            else:
                                entity_scores[entity_name] = community_score
        
        all_entities = {e.get("name"): e for e in entity_results}
        if self._entities_df is not None:
            for _, row in self._entities_df.iterrows():
                name = str(row.get("title", ""))
                if name in entity_scores and name not in all_entities:
                    all_entities[name] = {
                        "name": name,
                        "type": row.get("type", "unknown"),
                        "description": str(row.get("description", ""))[:100],
                        "score": entity_scores[name],
                    }
        
        sorted_entities = sorted(
            all_entities.values(),
            key=lambda x: entity_scores.get(x.get("name", ""), 0),
            reverse=True
        )
        
        return sorted_entities[:top_k]
    
    async def _strategy_vector_only(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """策略：仅向量检索"""
        if self.vector_retriever is None or not self.vector_retriever.is_ready():
            return []
        
        entities = await self.vector_retriever.search_entities(query, top_k)
        return entities
    
    async def _strategy_hybrid_async(self, context: MemoirContext, query: str, top_k: int) -> List[Dict[str, Any]]:
        """策略：混合检索（关键词+向量融合）"""
        keyword_results = self._strategy_keyword_only(context, top_k)
        
        if self.vector_retriever and self.vector_retriever.is_ready():
            vector_results = await self.vector_retriever.search_entities(query, top_k)
            return self._merge_keyword_vector(keyword_results, vector_results, top_k)
        
        return keyword_results
    
    def _merge_keyword_vector(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """融合关键词和向量检索结果"""
        merged = {}
        
        for item in keyword_results:
            name = item.get("name", "")
            if name:
                merged[name] = item.copy()
                merged[name]["score"] = item.get("score", 1) * 0.5
                merged[name]["source"] = "keyword"
        
        for item in vector_results:
            name = item.get("name", "")
            if name:
                if name in merged:
                    merged[name]["score"] += (1 - item.get("score", 0)) * 0.5
                    merged[name]["source"] = "hybrid"
                else:
                    merged[name] = item.copy()
                    merged[name]["score"] = (1 - item.get("score", 0)) * 0.5
                    merged[name]["source"] = "vector"
        
        sorted_results = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        return sorted_results[:top_k]
    
    def run_strategy(
        self,
        strategy_name: str,
        strategy_fn: Callable,
        top_k: int = 10,
    ) -> BenchmarkResult:
        """
        运行单个策略的评测
        
        Args:
            strategy_name: 策略名称
            strategy_fn: 策略函数
            top_k: 返回结果数量
            
        Returns:
            BenchmarkResult: 评测结果
        """
        all_metrics = []
        details = []
        
        for test_case in self.test_cases:
            context = MemoirContext(
                original_text=test_case.memoir_text,
                year=test_case.ground_truth_time,
                location=test_case.ground_truth_location,
            )
            
            start_time = time.time()
            results = strategy_fn(context, top_k)
            latency_ms = (time.time() - start_time) * 1000
            
            metrics = self._calculate_metrics(
                retrieved_entities=results,
                ground_truth=test_case.ground_truth_entities,
                latency_ms=latency_ms,
            )
            all_metrics.append(metrics)
            
            details.append({
                "query_id": test_case.query_id,
                "query_type": test_case.query_type,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "hit_at_3": metrics.hit_at_k.get(3, False),
                "hit_at_5": metrics.hit_at_k.get(5, False),
                "latency_ms": metrics.latency_ms,
            })
        
        avg_precision = sum(m.precision for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m.recall for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m.f1 for m in all_metrics) / len(all_metrics)
        avg_latency = sum(m.latency_ms for m in all_metrics) / len(all_metrics)
        hit_at_3 = sum(1 for m in all_metrics if m.hit_at_k.get(3, False)) / len(all_metrics)
        hit_at_5 = sum(1 for m in all_metrics if m.hit_at_k.get(5, False)) / len(all_metrics)
        hit_at_10 = sum(1 for m in all_metrics if m.hit_at_k.get(10, False)) / len(all_metrics)
        
        return BenchmarkResult(
            strategy_name=strategy_name,
            total_queries=len(self.test_cases),
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_latency_ms=avg_latency,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            details=details,
        )
    
    def run_all_strategies(self, top_k: int = 10) -> Dict[str, BenchmarkResult]:
        """
        运行所有策略的对比评测
        
        Args:
            top_k: 返回结果数量
            
        Returns:
            Dict[str, BenchmarkResult]: 各策略的评测结果
        """
        strategies = {
            "keyword_only": self._strategy_keyword_only,
            "entity_first": self._strategy_entity_first,
            "community_first": self._strategy_community_first,
            "hybrid_keyword": self._strategy_hybrid,
        }
        
        results = {}
        for name, fn in strategies.items():
            print(f"Running strategy: {name}...")
            results[name] = self.run_strategy(name, fn, top_k)
        
        if self.vector_retriever and self.vector_retriever.is_ready():
            print("Running strategy: vector_only...")
            results["vector_only"] = self.run_async_strategy("vector_only", top_k)
            
            print("Running strategy: hybrid_vector...")
            results["hybrid_vector"] = self.run_async_strategy("hybrid_vector", top_k)
        
        return results
    
    def run_async_strategy(self, strategy_name: str, top_k: int = 10) -> BenchmarkResult:
        """运行异步策略"""
        all_metrics = []
        details = []
        
        for test_case in self.test_cases:
            context = MemoirContext(
                original_text=test_case.memoir_text,
                year=test_case.ground_truth_time,
                location=test_case.ground_truth_location,
            )
            query = context.to_query()
            
            start_time = time.time()
            
            if strategy_name == "vector_only":
                results = asyncio.run(self._strategy_vector_only(query, top_k))
            elif strategy_name == "hybrid_vector":
                results = asyncio.run(self._strategy_hybrid_async(context, query, top_k))
            else:
                results = []
            
            latency_ms = (time.time() - start_time) * 1000
            
            metrics = self._calculate_metrics(
                retrieved_entities=results,
                ground_truth=test_case.ground_truth_entities,
                latency_ms=latency_ms,
            )
            all_metrics.append(metrics)
            
            details.append({
                "query_id": test_case.query_id,
                "query_type": test_case.query_type,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "hit_at_3": metrics.hit_at_k.get(3, False),
                "hit_at_5": metrics.hit_at_k.get(5, False),
                "latency_ms": metrics.latency_ms,
            })
        
        avg_precision = sum(m.precision for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m.recall for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m.f1 for m in all_metrics) / len(all_metrics)
        avg_latency = sum(m.latency_ms for m in all_metrics) / len(all_metrics)
        hit_at_3 = sum(1 for m in all_metrics if m.hit_at_k.get(3, False)) / len(all_metrics)
        hit_at_5 = sum(1 for m in all_metrics if m.hit_at_k.get(5, False)) / len(all_metrics)
        hit_at_10 = sum(1 for m in all_metrics if m.hit_at_k.get(10, False)) / len(all_metrics)
        
        return BenchmarkResult(
            strategy_name=strategy_name,
            total_queries=len(self.test_cases),
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_latency_ms=avg_latency,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            details=details,
        )
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Optional[str] = None,
    ) -> str:
        """
       生成评测报告
        
        Args:
            results: 评测结果
            output_path: 报告输出路径
            
        Returns:
            str: 报告内容
        """
        lines = []
        lines.append("# 检索策略对比评测报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"测试用例数: {len(self.test_cases)}")
        
        lines.append("\n## 汇总结果\n")
        lines.append("| 策略 | Precision | Recall | F1 | Hit@3 | Hit@5 | Hit@10 | 延迟(ms) |")
        lines.append("|------|-----------|--------|-----|-------|-------|--------|----------|")
        
        for name, result in results.items():
            lines.append(
                f"| {name} | {result.avg_precision:.4f} | {result.avg_recall:.4f} | "
                f"{result.avg_f1:.4f} | {result.hit_at_3:.2%} | {result.hit_at_5:.2%} | "
                f"{result.hit_at_10:.2%} | {result.avg_latency_ms:.2f} |"
            )
        
        best_f1 = max(results.items(), key=lambda x: x[1].avg_f1)
        lines.append(f"\n**最佳策略（F1）**: {best_f1[0]} (F1={best_f1[1].avg_f1:.4f})")
        
        lines.append("\n## 各策略详细分析\n")
        for name, result in results.items():
            lines.append(f"### {name}\n")
            lines.append(f"- 平均Precision: {result.avg_precision:.4f}")
            lines.append(f"- 平均Recall: {result.avg_recall:.4f}")
            lines.append(f"- 平均F1: {result.avg_f1:.4f}")
            lines.append(f"- Hit@3: {result.hit_at_3:.2%}")
            lines.append(f"- Hit@5: {result.hit_at_5:.2%}")
            lines.append(f"- Hit@10: {result.hit_at_10:.2%}")
            lines.append(f"- 平均延迟: {result.avg_latency_ms:.2f}ms\n")
        
        lines.append("\n## 测试用例详情\n")
        for test_case in self.test_cases:
            lines.append(f"- **{test_case.query_id}** ({test_case.query_type}): {test_case.memoir_text[:50]}...")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
        
        return report
    
    def export_results(
        self,
        results: Dict[str, BenchmarkResult],
        output_dir: str,
    ):
        """
        导出评测结果到文件
        
        Args:
            results: 评测结果
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary_data = []
        for name, result in results.items():
            summary_data.append(result.to_dict())
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "benchmark_summary.csv", index=False, encoding="utf-8")
        
        for name, result in results.items():
            details_df = pd.DataFrame(result.details)
            details_df.to_csv(output_path / f"benchmark_{name}_details.csv", index=False, encoding="utf-8")
        
        self.generate_report(results, str(output_path / "benchmark_report.md"))
        
        print(f"Results exported to: {output_path}")


def run_benchmark(
    index_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_cases: Optional[List[TestCase]] = None,
) -> Dict[str, BenchmarkResult]:
    """
    运行检索策略对比评测的便捷函数
    
    Args:
        index_dir: GraphRAG索引目录
        output_dir: 结果输出目录
        test_cases: 自定义测试用例
        
    Returns:
        Dict[str, BenchmarkResult]: 评测结果
    """
    benchmark = RetrievalBenchmark(index_dir=index_dir, test_cases=test_cases)
    
    if not benchmark.is_ready():
        print("Error: Index not ready. Please build the index first.")
        return {}
    
    results = benchmark.run_all_strategies()
    
    if output_dir:
        benchmark.export_results(results, output_dir)
    else:
        report = benchmark.generate_report(results)
        print(report)
    
    return results


if __name__ == "__main__":
    run_benchmark()
