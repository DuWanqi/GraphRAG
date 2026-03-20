"""
检索策略对比评测模块
用于对比不同检索策略的效果，生成实验报告

不修改原有代码，独立运行评测实验
"""

import os
import json
import math
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
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
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
    avg_mrr: float = 0.0
    avg_ndcg_at_3: float = 0.0
    avg_ndcg_at_5: float = 0.0
    avg_ndcg_at_10: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "total_queries": self.total_queries,
            "avg_precision": round(self.avg_precision, 4),
            "avg_recall": round(self.avg_recall, 4),
            "avg_f1": round(self.avg_f1, 4),
            "avg_mrr": round(self.avg_mrr, 4),
            "avg_ndcg_at_3": round(self.avg_ndcg_at_3, 4),
            "avg_ndcg_at_5": round(self.avg_ndcg_at_5, 4),
            "avg_ndcg_at_10": round(self.avg_ndcg_at_10, 4),
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

    # LLM-as-a-Judge 相关性评分 prompt
    RELEVANCE_JUDGE_PROMPT = """你是一位检索质量评估专家。请判断以下检索到的实体与用户的回忆录查询之间的相关性。

## 回忆录查询
{query}

## 检索到的实体列表
{entities}

请对每个实体打出相关性分数（0-3分）：
- 3分：高度相关，直接涉及查询提到的时间、地点、人物或事件
- 2分：较为相关，与查询的历史背景或主题有明确联系
- 1分：略有关联，存在间接联系但不够紧密
- 0分：不相关

请以JSON格式返回，例如：
{{"scores": [{{"entity": "实体名", "score": 3, "reason": "直接提到的地点"}}, ...]}}

只返回JSON，不要其他内容。"""

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
        self.llm_provider = llm_provider

        # LLM adapter for LLM-as-a-Judge relevance scoring
        try:
            self.llm_adapter = create_llm_adapter(provider=llm_provider)
        except Exception as e:
            print(f"[Warning] 无法初始化LLM适配器: {e}")
            self.llm_adapter = None

        try:
            self.vector_retriever = VectorRetriever(
                index_dir=str(self.index_dir),
                llm_adapter=self.llm_adapter,
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

        # nDCG 和 MRR 必须由 LLM-as-a-Judge 计算，此处不填充
        # 调用 _calculate_metrics_with_llm 获取完整指标

        return metrics

    async def _calculate_metrics_with_llm(
        self,
        retrieved_entities: List[Dict[str, Any]],
        ground_truth: List[str],
        latency_ms: float,
        query_text: str,
    ) -> RetrievalMetrics:
        """使用 LLM-as-a-Judge 计算检索指标（含 nDCG 和 MRR）"""
        # 先计算基础指标（Precision/Recall/F1/Hit@K）
        metrics = self._calculate_metrics(retrieved_entities, ground_truth, latency_ms)

        if not retrieved_entities:
            return metrics

        # 用 LLM 对检索结果打相关性分
        rel_vector = await self._llm_judge_relevance(query_text, retrieved_entities)

        # 用 LLM 打分重新计算 nDCG
        for k in [3, 5, 10]:
            metrics.ndcg_at_k[k] = self._ndcg(rel_vector, k, len(ground_truth))

        # 用 LLM 打分重新计算 MRR（第一个相关性 >= 2 的结果）
        metrics.mrr = 0.0
        for i, rel in enumerate(rel_vector):
            if rel >= 2:
                metrics.mrr = 1.0 / (i + 1)
                break

        return metrics

    async def _llm_judge_relevance(
        self,
        query_text: str,
        retrieved_entities: List[Dict[str, Any]],
    ) -> List[float]:
        """使用 LLM-as-a-Judge 对检索结果评分（0-3）"""
        if not self.llm_adapter:
            raise RuntimeError("nDCG/MRR 必须使用 LLM-as-a-Judge 评分，但未配置 LLM 适配器。请提供有效的 llm_provider。")

        # 构建实体列表文本
        entity_lines = []
        for i, entity in enumerate(retrieved_entities):
            name = entity.get("name", "")
            etype = entity.get("type", "")
            desc = entity.get("description", "")[:80]
            entity_lines.append(f"{i+1}. {name} (类型: {etype}): {desc}")
        entities_text = "\n".join(entity_lines)

        prompt = self.RELEVANCE_JUDGE_PROMPT.format(
            query=query_text,
            entities=entities_text,
        )

        try:
            response = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            result_text = response.content.strip()

            # 解析 JSON
            import re
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                scores_list = data.get("scores", [])

                # 按实体顺序映射分数
                scores_map = {}
                for item in scores_list:
                    ename = item.get("entity", "")
                    score = float(item.get("score", 0))
                    scores_map[ename] = min(max(score, 0), 3)  # clamp 0-3

                rel_vector = []
                for entity in retrieved_entities:
                    name = entity.get("name", "")
                    # 精确匹配或子串匹配
                    matched_score = scores_map.get(name, None)
                    if matched_score is None:
                        for sname, sscore in scores_map.items():
                            if sname in name or name in sname:
                                matched_score = sscore
                                break
                    rel_vector.append(matched_score if matched_score is not None else 0.0)
                return rel_vector
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM judge 失败: {e}, 降级为规则评分")

        return [0.0] * len(retrieved_entities)

    @staticmethod
    def _ndcg(relevances: List[float], k: int, n_relevant: int) -> float:
        """Compute nDCG@k given a relevance vector."""
        rel_k = relevances[:k]
        dcg = sum(
            rel / math.log2(i + 2) for i, rel in enumerate(rel_k)
        )
        # ideal: sort all relevances descending, take top-k
        ideal_rels = sorted(relevances, reverse=True)[:k]
        if not ideal_rels and n_relevant > 0:
            ideal_rels = [3.0] * min(k, n_relevant)
        elif not ideal_rels:
            return 0.0
        idcg = sum(
            rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels)
        )
        return dcg / idcg if idcg > 0 else 0.0

    
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
        """运行单个策略的评测（使用 LLM-as-a-Judge）"""
        if not self.llm_adapter:
            raise RuntimeError("nDCG/MRR 必须使用 LLM-as-a-Judge 评分，但未配置 LLM 适配器。请提供有效的 llm_provider。")
        return asyncio.run(
            self.run_strategy_async(strategy_name, strategy_fn, top_k, use_llm_judge=True)
        )

    async def run_strategy_async(
        self,
        strategy_name: str,
        strategy_fn: Callable,
        top_k: int = 10,
        use_llm_judge: bool = True,
    ) -> BenchmarkResult:
        """
        运行单个策略的评测

        Args:
            strategy_name: 策略名称
            strategy_fn: 策略函数
            top_k: 返回结果数量
            use_llm_judge: 是否使用 LLM-as-a-Judge 评分 nDCG/MRR

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

            if use_llm_judge and self.llm_adapter:
                metrics = await self._calculate_metrics_with_llm(
                    retrieved_entities=results,
                    ground_truth=test_case.ground_truth_entities,
                    latency_ms=latency_ms,
                    query_text=test_case.memoir_text,
                )
            else:
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
                "mrr": metrics.mrr,
                "ndcg_at_3": metrics.ndcg_at_k.get(3, 0.0),
                "ndcg_at_5": metrics.ndcg_at_k.get(5, 0.0),
                "ndcg_at_10": metrics.ndcg_at_k.get(10, 0.0),
                "hit_at_3": metrics.hit_at_k.get(3, False),
                "hit_at_5": metrics.hit_at_k.get(5, False),
                "latency_ms": metrics.latency_ms,
            })

        n = len(all_metrics)
        return self._aggregate_benchmark_result(strategy_name, all_metrics, details, n)

    @staticmethod
    def _aggregate_benchmark_result(
        strategy_name: str,
        all_metrics: List[RetrievalMetrics],
        details: List[Dict[str, Any]],
        n: int,
    ) -> BenchmarkResult:
        """汇总指标到 BenchmarkResult"""
        avg_precision = sum(m.precision for m in all_metrics) / n
        avg_recall = sum(m.recall for m in all_metrics) / n
        avg_f1 = sum(m.f1 for m in all_metrics) / n
        avg_latency = sum(m.latency_ms for m in all_metrics) / n
        avg_mrr = sum(m.mrr for m in all_metrics) / n
        avg_ndcg_3 = sum(m.ndcg_at_k.get(3, 0.0) for m in all_metrics) / n
        avg_ndcg_5 = sum(m.ndcg_at_k.get(5, 0.0) for m in all_metrics) / n
        avg_ndcg_10 = sum(m.ndcg_at_k.get(10, 0.0) for m in all_metrics) / n
        hit_at_3 = sum(1 for m in all_metrics if m.hit_at_k.get(3, False)) / n
        hit_at_5 = sum(1 for m in all_metrics if m.hit_at_k.get(5, False)) / n
        hit_at_10 = sum(1 for m in all_metrics if m.hit_at_k.get(10, False)) / n

        return BenchmarkResult(
            strategy_name=strategy_name,
            total_queries=n,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            avg_latency_ms=avg_latency,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            hit_at_10=hit_at_10,
            avg_mrr=avg_mrr,
            avg_ndcg_at_3=avg_ndcg_3,
            avg_ndcg_at_5=avg_ndcg_5,
            avg_ndcg_at_10=avg_ndcg_10,
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

    async def run_all_strategies_async(self, top_k: int = 10) -> Dict[str, BenchmarkResult]:
        """
        使用 LLM-as-a-Judge 运行所有策略的对比评测

        nDCG 和 MRR 的相关性分数由 LLM 打分（0-3），而非规则匹配。
        """
        strategies = {
            "keyword_only": self._strategy_keyword_only,
            "entity_first": self._strategy_entity_first,
            "community_first": self._strategy_community_first,
            "hybrid_keyword": self._strategy_hybrid,
        }

        results = {}
        for name, fn in strategies.items():
            print(f"Running strategy (LLM judge): {name}...")
            results[name] = await self.run_strategy_async(name, fn, top_k, use_llm_judge=True)

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
                "mrr": metrics.mrr,
                "ndcg_at_3": metrics.ndcg_at_k.get(3, 0.0),
                "ndcg_at_5": metrics.ndcg_at_k.get(5, 0.0),
                "ndcg_at_10": metrics.ndcg_at_k.get(10, 0.0),
                "hit_at_3": metrics.hit_at_k.get(3, False),
                "hit_at_5": metrics.hit_at_k.get(5, False),
                "latency_ms": metrics.latency_ms,
            })

        return self._aggregate_benchmark_result(strategy_name, all_metrics, details, len(all_metrics))

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
        lines.append("| 策略 | Precision | Recall | F1 | MRR | nDCG@3 | nDCG@5 | nDCG@10 | Hit@3 | Hit@5 | Hit@10 | 延迟(ms) |")
        lines.append("|------|-----------|--------|-----|-----|--------|--------|---------|-------|-------|--------|----------|")

        for name, result in results.items():
            lines.append(
                f"| {name} | {result.avg_precision:.4f} | {result.avg_recall:.4f} | "
                f"{result.avg_f1:.4f} | {result.avg_mrr:.4f} | "
                f"{result.avg_ndcg_at_3:.4f} | {result.avg_ndcg_at_5:.4f} | {result.avg_ndcg_at_10:.4f} | "
                f"{result.hit_at_3:.2%} | {result.hit_at_5:.2%} | "
                f"{result.hit_at_10:.2%} | {result.avg_latency_ms:.2f} |"
            )

        best_f1 = max(results.items(), key=lambda x: x[1].avg_f1)
        best_ndcg = max(results.items(), key=lambda x: x[1].avg_ndcg_at_5)
        lines.append(f"\n**最佳策略（F1）**: {best_f1[0]} (F1={best_f1[1].avg_f1:.4f})")
        lines.append(f"**最佳策略（nDCG@5）**: {best_ndcg[0]} (nDCG@5={best_ndcg[1].avg_ndcg_at_5:.4f})")

        lines.append("\n## 各策略详细分析\n")
        for name, result in results.items():
            lines.append(f"### {name}\n")
            lines.append(f"- 平均Precision: {result.avg_precision:.4f}")
            lines.append(f"- 平均Recall: {result.avg_recall:.4f}")
            lines.append(f"- 平均F1: {result.avg_f1:.4f}")
            lines.append(f"- 平均MRR: {result.avg_mrr:.4f}")
            lines.append(f"- nDCG@3: {result.avg_ndcg_at_3:.4f}")
            lines.append(f"- nDCG@5: {result.avg_ndcg_at_5:.4f}")
            lines.append(f"- nDCG@10: {result.avg_ndcg_at_10:.4f}")
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


_DOC_RELEVANCE_JUDGE_PROMPT = """你是一位检索质量评估专家。请判断以下检索到的文档段落与用户的回忆录查询之间的相关性。

## 回忆录查询
{query}

## 检索到的文档列表
{documents}

请对每个文档打出相关性分数（0-3分）：
- 3分：高度相关，直接涉及查询提到的时间、地点、人物或事件
- 2分：较为相关，与查询的历史背景或主题有明确联系
- 1分：略有关联，存在间接联系但不够紧密
- 0分：不相关

请以JSON格式返回，例如：
{{"scores": [{{"doc_id": 1, "score": 3, "reason": "直接描述了相关历史事件"}}, ...]}}

只返回JSON，不要其他内容。"""


async def evaluate_retrieval_quality(
    query_text: str,
    text_units: List[str],
    llm_adapter,
) -> Dict[str, Any]:
    """
    独立的检索质量评估函数，使用 LLM-as-a-Judge 对检索到的文档段落打分。

    Args:
        query_text: 回忆录查询文本
        text_units: 检索到的文档段落列表
        llm_adapter: LLM 适配器

    Returns:
        Dict 包含 ndcg_at_k, mrr, per_doc_scores
    """
    if not text_units:
        return {
            "ndcg_at_3": 0.0, "ndcg_at_5": 0.0, "ndcg_at_10": 0.0,
            "mrr": 0.0, "per_doc_scores": [],
        }

    # 构建文档列表文本（截取前150字作为摘要）
    doc_lines = []
    for i, text in enumerate(text_units):
        snippet = text.strip().replace("\n", " ")[:150]
        doc_lines.append(f"{i+1}. {snippet}")
    documents_text = "\n".join(doc_lines)

    prompt = _DOC_RELEVANCE_JUDGE_PROMPT.format(
        query=query_text,
        documents=documents_text,
    )

    per_doc_scores = []
    rel_vector = [0.0] * len(text_units)

    try:
        response = await llm_adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        result_text = response.content.strip()

        import re
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            scores_list = data.get("scores", [])

            for item in scores_list:
                doc_id = int(item.get("doc_id", 0)) - 1  # 1-indexed -> 0-indexed
                score = float(item.get("score", 0))
                reason = item.get("reason", "")
                if 0 <= doc_id < len(text_units):
                    rel_vector[doc_id] = min(max(score, 0), 3)
                    per_doc_scores.append({
                        "doc_id": doc_id + 1,
                        "snippet": text_units[doc_id].strip().replace("\n", " ")[:60],
                        "score": rel_vector[doc_id],
                        "reason": reason,
                    })

        # 补齐未被 LLM 返回的文档
        scored_ids = {item["doc_id"] for item in per_doc_scores}
        for i, text in enumerate(text_units):
            if (i + 1) not in scored_ids:
                per_doc_scores.append({
                    "doc_id": i + 1,
                    "snippet": text.strip().replace("\n", " ")[:60],
                    "score": 0.0,
                    "reason": "",
                })
        per_doc_scores.sort(key=lambda x: x["doc_id"])

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"LLM judge 评估失败: {e}")
        for i, text in enumerate(text_units):
            per_doc_scores.append({
                "doc_id": i + 1,
                "snippet": text.strip().replace("\n", " ")[:60],
                "score": 0.0,
                "reason": f"评估失败: {e}",
            })

    # 计算 nDCG
    n_docs = len(text_units)
    ndcg_scores = {}
    for k in [3, 5, 10]:
        ndcg_scores[k] = RetrievalBenchmark._ndcg(rel_vector, k, n_docs)

    # 计算 MRR（第一个 score >= 2 的文档）
    mrr = 0.0
    for i, rel in enumerate(rel_vector):
        if rel >= 2:
            mrr = 1.0 / (i + 1)
            break

    return {
        "ndcg_at_3": ndcg_scores[3],
        "ndcg_at_5": ndcg_scores[5],
        "ndcg_at_10": ndcg_scores[10],
        "mrr": mrr,
        "per_doc_scores": per_doc_scores,
    }


def run_benchmark(
    index_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    test_cases: Optional[List[TestCase]] = None,
    llm_provider: str = "hunyuan",
    use_llm_judge: bool = True,
) -> Dict[str, BenchmarkResult]:
    """
    运行检索策略对比评测的便捷函数

    Args:
        index_dir: GraphRAG索引目录
        output_dir: 结果输出目录
        test_cases: 自定义测试用例
        llm_provider: LLM 提供商（用于 LLM-as-a-Judge）
        use_llm_judge: 是否使用 LLM 打分 nDCG/MRR

    Returns:
        Dict[str, BenchmarkResult]: 评测结果
    """
    benchmark = RetrievalBenchmark(
        index_dir=index_dir, test_cases=test_cases, llm_provider=llm_provider,
    )

    if not benchmark.is_ready():
        print("Error: Index not ready. Please build the index first.")
        return {}

    if use_llm_judge and benchmark.llm_adapter:
        print("使用 LLM-as-a-Judge 评估检索相关性 (nDCG/MRR)...")
        results = asyncio.run(benchmark.run_all_strategies_async())
    else:
        results = benchmark.run_all_strategies()

    if output_dir:
        benchmark.export_results(results, output_dir)
    else:
        report = benchmark.generate_report(results)
        print(report)

    return results


if __name__ == "__main__":
    run_benchmark()
