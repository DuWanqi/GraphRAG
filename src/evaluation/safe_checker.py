"""
独立知识验证器 (SAFE-inspired)

使用 LLM 自身知识或网络搜索独立验证生成文本的事实准确性，
不依赖项目知识库，从而帮助评估知识库的合理性和完整性。

两种验证模式：
1. LLM 独立知识验证（默认）：让 LLM 仅凭自身训练知识判断事实真伪
2. 网络搜索验证（可选）：通过 Google Custom Search 搜索网络证据后由 LLM 判断

参考：
- SAFE (Search-Augmented Factuality Evaluator, 2024)
  Paper: https://arxiv.org/pdf/2403.18802
- FActScore (2023)
  Paper: https://arxiv.org/pdf/2305.14251
"""

import json
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import httpx

from ..llm import LLMAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SAFECheckResult:
    """独立知识验证结果"""
    safe_score: float                          # 独立验证得分 = supported / total
    total_facts: int
    supported_facts: int
    unsupported_facts: int
    irrelevant_facts: int                      # 无法判断的事实
    fact_details: List[Dict[str, Any]] = field(default_factory=list)
    kb_comparison: Optional[Dict[str, Any]] = None   # 与 KB FActScore 的对比
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "safe_score": self.safe_score,
            "total_facts": self.total_facts,
            "supported_facts": self.supported_facts,
            "unsupported_facts": self.unsupported_facts,
            "irrelevant_facts": self.irrelevant_facts,
            "fact_details": self.fact_details,
            "summary": self.summary,
        }
        if self.kb_comparison:
            result["kb_comparison"] = self.kb_comparison
        return result


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """请将以下文本拆解为独立的原子事实。

规则：
1. 每个原子事实是一个简短、独立、可核查的陈述句
2. 每条只包含一个信息点（一个时间、一个地点、一个事件等）
3. 去掉主观评价、文学修辞、模糊表述，只保留客观可验证的事实
4. 保持原文的人名、地名、数字不变
5. 如果没有可验证的事实，返回空列表 []

文本：
{text}

请以JSON数组格式返回，每条是一个字符串。例如：
["十一届三中全会于1978年12月召开", "十一届三中全会在北京举行"]

只返回JSON数组，不要其他内容。"""


_LLM_VERIFY_PROMPT = """你是一位历史事实核查专家。请仅凭你自身的知识判断以下事实是否正确。
不要猜测，如果你不确定请回答 UNSURE。

事实：{fact}

请严格按照以下JSON格式回答，不要输出其他内容：
{{"verdict": "SUPPORTED" 或 "NOT_SUPPORTED" 或 "UNSURE", "explanation": "简要理由"}}"""


_BATCH_LLM_VERIFY_PROMPT = """你是一位历史事实核查专家。请仅凭你自身的知识逐一判断以下事实是否正确。
不要猜测，如果你不确定请回答 UNSURE。

需要验证的事实：
{facts}

请严格按照以下JSON格式返回，不要输出其他内容：
{{
  "results": [
    {{"fact": "事实内容", "verdict": "SUPPORTED 或 NOT_SUPPORTED 或 UNSURE", "explanation": "简要理由"}},
    ...
  ]
}}"""


_SEARCH_VERIFY_PROMPT = """你是一位历史事实核查专家。请根据以下搜索结果判断事实是否正确。

搜索结果：
{search_results}

事实：{fact}

请严格按照以下JSON格式回答，不要输出其他内容：
{{"verdict": "SUPPORTED" 或 "NOT_SUPPORTED" 或 "UNSURE", "explanation": "简要理由"}}"""


_SEARCH_QUERY_PROMPT = """请为以下事实生成一个简短的中文搜索查询词，用于在搜索引擎中查找验证该事实的信息。
只返回搜索查询词本身，不要其他内容。

事实：{fact}"""


# ---------------------------------------------------------------------------
# SAFEFactChecker
# ---------------------------------------------------------------------------

class SAFEFactChecker:
    """
    独立知识验证器

    与 FActScoreChecker 的区别：
    - FActScoreChecker 用项目知识库（检索结果）作为验证依据
    - SAFEFactChecker 用 LLM 自身知识 / 网络搜索作为验证依据
    两者结果互相独立，对比后可评估知识库的覆盖率和准确性。
    """

    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        google_api_key: Optional[str] = None,
        google_cse_id: Optional[str] = None,
    ):
        self.llm_adapter = llm_adapter
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self._cache: Dict[str, SAFECheckResult] = {}
        self._decompose_cache: Dict[str, List[str]] = {}

    @property
    def search_enabled(self) -> bool:
        return bool(self.google_api_key and self.google_cse_id)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    async def check(
        self,
        generated_text: str,
        memoir_text: str = "",
        atomic_facts: Optional[List[str]] = None,
        kb_factscore: Optional[float] = None,
        kb_supported_facts: Optional[int] = None,
        kb_total_facts: Optional[int] = None,
        use_search: bool = False,
    ) -> SAFECheckResult:
        """
        执行独立知识验证

        Args:
            generated_text: 待验证的生成文本
            memoir_text: 原始回忆录（可选，仅用于上下文参考）
            atomic_facts: 预先分解好的原子事实（可选，避免重复分解）
            kb_factscore: KB 验证的 FActScore（用于对比）
            kb_supported_facts: KB 验证中被支持的事实数
            kb_total_facts: KB 验证中的总事实数
            use_search: 是否启用网络搜索模式
        """
        cache_key = hashlib.md5(
            f"{generated_text[:200]}|safe|{use_search}".encode()
        ).hexdigest()
        if cache_key in self._cache:
            logger.info("[SAFE] 使用缓存结果")
            return self._cache[cache_key]

        # 1. 分解原子事实
        if atomic_facts is None:
            atomic_facts = await self._decompose_text(generated_text)
        total = len(atomic_facts)
        if total == 0:
            return SAFECheckResult(
                safe_score=0.0, total_facts=0,
                supported_facts=0, unsupported_facts=0, irrelevant_facts=0,
                summary="未提取到可验证的原子事实",
            )

        # 2. 逐一（批量）验证
        if use_search and self.search_enabled:
            details = await self._verify_batch_search(atomic_facts)
        else:
            details = await self._verify_batch_llm(atomic_facts)

        # 3. 统计
        supported = sum(1 for d in details if d["verdict"] == "SUPPORTED")
        unsupported = sum(1 for d in details if d["verdict"] == "NOT_SUPPORTED")
        irrelevant = sum(1 for d in details if d["verdict"] == "UNSURE")
        safe_score = supported / total if total > 0 else 0.0

        # 4. KB 对比
        kb_comparison = None
        if kb_factscore is not None:
            kb_comparison = self._compare_with_kb(
                safe_score, details,
                kb_factscore, kb_supported_facts, kb_total_facts,
            )

        summary = self._generate_summary(
            safe_score, total, supported, unsupported, irrelevant,
            kb_comparison, use_search,
        )

        result = SAFECheckResult(
            safe_score=safe_score,
            total_facts=total,
            supported_facts=supported,
            unsupported_facts=unsupported,
            irrelevant_facts=irrelevant,
            fact_details=details,
            kb_comparison=kb_comparison,
            summary=summary,
        )
        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # 原子事实分解（复用 FActScoreChecker 相同的 prompt）
    # ------------------------------------------------------------------

    async def _decompose_text(self, text: str) -> List[str]:
        cache_key = hashlib.md5(text[:200].encode()).hexdigest()
        if cache_key in self._decompose_cache:
            return self._decompose_cache[cache_key]

        if not self.llm_adapter:
            facts = self._decompose_rule_based(text)
            self._decompose_cache[cache_key] = facts
            return facts

        prompt = _DECOMPOSE_PROMPT.format(text=text[:1000])
        try:
            resp = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=1024,
            )
            raw = resp.content.strip()
            try:
                facts = json.loads(raw)
                if not isinstance(facts, list):
                    facts = []
            except json.JSONDecodeError:
                facts = self._extract_json_array(raw)
            self._decompose_cache[cache_key] = facts
            return facts
        except Exception as e:
            logger.error(f"[SAFE] 分解文本失败: {e}")
            facts = self._decompose_rule_based(text)
            self._decompose_cache[cache_key] = facts
            return facts

    @staticmethod
    def _decompose_rule_based(text: str) -> List[str]:
        sentences = re.split(r'[。！？.!?\n;；]', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue
            if any(s in f or f in s for f in facts):
                continue
            if any(c.isalnum() for c in s):
                facts.append(s)
        return facts

    @staticmethod
    def _extract_json_array(text: str) -> List[str]:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        lines = text.strip().split('\n')
        facts = []
        for line in lines:
            line = re.sub(r'^[-*•]\s*', '', line.strip())
            line = re.sub(r'^\d+\.\s*', '', line)
            if line and len(line) > 3:
                facts.append(line)
        return facts

    # ------------------------------------------------------------------
    # 批量 LLM 自身知识验证
    # ------------------------------------------------------------------

    async def _verify_batch_llm(
        self, facts: List[str], batch_size: int = 5,
    ) -> List[Dict[str, Any]]:
        """用 LLM 自身知识批量验证原子事实"""
        if not self.llm_adapter:
            return [{"fact": f, "verdict": "UNSURE", "explanation": "无LLM适配器"} for f in facts]

        all_details: List[Dict[str, Any]] = []

        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            facts_str = "\n".join(f"{j+1}. {fact}" for j, fact in enumerate(batch))
            prompt = _BATCH_LLM_VERIFY_PROMPT.format(facts=facts_str)

            try:
                resp = await self.llm_adapter.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=1024, timeout=30,
                )
                batch_details = self._parse_batch_response(resp.content, batch)
                all_details.extend(batch_details)
            except Exception as e:
                logger.error(f"[SAFE] 批量LLM验证失败: {e}")
                all_details.extend(
                    {"fact": f, "verdict": "UNSURE", "explanation": f"验证失败: {e}"}
                    for f in batch
                )

        return all_details

    def _parse_batch_response(
        self, response_text: str, original_facts: List[str],
    ) -> List[Dict[str, Any]]:
        """解析批量验证结果"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                results_list = data.get("results", [])

                # 按序号/内容匹配
                details = []
                for idx, fact in enumerate(original_facts):
                    matched = False
                    for item in results_list:
                        item_fact = item.get("fact", "")
                        if fact in item_fact or item_fact in fact:
                            verdict = self._normalize_verdict(item.get("verdict", "UNSURE"))
                            details.append({
                                "fact": fact,
                                "verdict": verdict,
                                "explanation": item.get("explanation", ""),
                            })
                            matched = True
                            break
                    if not matched and idx < len(results_list):
                        item = results_list[idx]
                        verdict = self._normalize_verdict(item.get("verdict", "UNSURE"))
                        details.append({
                            "fact": fact,
                            "verdict": verdict,
                            "explanation": item.get("explanation", ""),
                        })
                        matched = True
                    if not matched:
                        details.append({
                            "fact": fact,
                            "verdict": "UNSURE",
                            "explanation": "未能从LLM响应中匹配到结果",
                        })
                return details
        except Exception as e:
            logger.error(f"[SAFE] 解析批量响应失败: {e}")

        return [
            {"fact": f, "verdict": "UNSURE", "explanation": "响应解析失败"}
            for f in original_facts
        ]

    @staticmethod
    def _normalize_verdict(v: str) -> str:
        v = v.strip().upper()
        if "SUPPORTED" in v and "NOT" not in v:
            return "SUPPORTED"
        if "NOT" in v:
            return "NOT_SUPPORTED"
        return "UNSURE"

    # ------------------------------------------------------------------
    # 网络搜索验证
    # ------------------------------------------------------------------

    async def _verify_batch_search(
        self, facts: List[str],
    ) -> List[Dict[str, Any]]:
        """通过网络搜索逐一验证原子事实"""
        details: List[Dict[str, Any]] = []
        for fact in facts:
            detail = await self._verify_single_search(fact)
            details.append(detail)
        return details

    async def _verify_single_search(self, fact: str) -> Dict[str, Any]:
        """搜索网络并用 LLM 判断"""
        # 生成搜索查询
        query = await self._generate_search_query(fact)
        # 执行搜索
        search_results = await self._search_google(query)
        if not search_results:
            # 搜索无结果，回退到 LLM 知识
            return await self._verify_single_llm(fact)

        snippets = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in search_results[:5]
        )

        prompt = _SEARCH_VERIFY_PROMPT.format(
            search_results=snippets, fact=fact,
        )
        try:
            resp = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=256, timeout=15,
            )
            parsed = self._parse_single_response(resp.content)
            parsed["fact"] = fact
            parsed["search_query"] = query
            parsed["source"] = "web_search"
            return parsed
        except Exception as e:
            logger.error(f"[SAFE] 搜索验证失败: {e}")
            return {"fact": fact, "verdict": "UNSURE", "explanation": str(e), "source": "error"}

    async def _verify_single_llm(self, fact: str) -> Dict[str, Any]:
        """单条 LLM 知识验证（回退用）"""
        prompt = _LLM_VERIFY_PROMPT.format(fact=fact)
        try:
            resp = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=256, timeout=15,
            )
            parsed = self._parse_single_response(resp.content)
            parsed["fact"] = fact
            parsed["source"] = "llm_knowledge"
            return parsed
        except Exception as e:
            return {"fact": fact, "verdict": "UNSURE", "explanation": str(e), "source": "error"}

    def _parse_single_response(self, text: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "verdict": self._normalize_verdict(data.get("verdict", "UNSURE")),
                    "explanation": data.get("explanation", ""),
                }
        except Exception:
            pass
        # 简单文本解析
        upper = text.upper()
        if "SUPPORTED" in upper and "NOT" not in upper:
            return {"verdict": "SUPPORTED", "explanation": text.strip()[:100]}
        if "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper:
            return {"verdict": "NOT_SUPPORTED", "explanation": text.strip()[:100]}
        return {"verdict": "UNSURE", "explanation": text.strip()[:100]}

    async def _generate_search_query(self, fact: str) -> str:
        """用 LLM 为事实生成搜索查询"""
        if not self.llm_adapter:
            return fact[:80]
        prompt = _SEARCH_QUERY_PROMPT.format(fact=fact)
        try:
            resp = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=64, timeout=10,
            )
            return resp.content.strip()[:120]
        except Exception:
            return fact[:80]

    async def _search_google(self, query: str) -> List[Dict[str, str]]:
        """调用 Google Custom Search JSON API"""
        if not self.google_api_key or not self.google_cse_id:
            return []
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": 5,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
            items = data.get("items", [])
            return [
                {
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                }
                for item in items
            ]
        except Exception as e:
            logger.warning(f"[SAFE] Google搜索失败: {e}")
            return []

    # ------------------------------------------------------------------
    # KB 对比
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_with_kb(
        safe_score: float,
        safe_details: List[Dict[str, Any]],
        kb_factscore: float,
        kb_supported: Optional[int],
        kb_total: Optional[int],
    ) -> Dict[str, Any]:
        """对比独立验证与 KB 验证的结果，评估知识库质量"""
        diff = safe_score - kb_factscore

        if diff > 0.15:
            assessment = "知识库覆盖不足：独立验证得分明显高于知识库验证，建议补充知识库内容"
        elif diff < -0.15:
            assessment = "知识库质量良好：知识库验证得分高于独立验证，说明知识库提供了有效的事实支撑"
        else:
            assessment = "知识库与独立知识基本一致：两者评分接近，知识库覆盖较为合理"

        return {
            "safe_score": safe_score,
            "kb_factscore": kb_factscore,
            "score_diff": round(diff, 4),
            "kb_supported": kb_supported,
            "kb_total": kb_total,
            "assessment": assessment,
        }

    # ------------------------------------------------------------------
    # 摘要
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_summary(
        safe_score: float, total: int, supported: int,
        unsupported: int, irrelevant: int,
        kb_comparison: Optional[Dict], use_search: bool,
    ) -> str:
        mode = "网络搜索" if use_search else "LLM独立知识"
        parts = [
            f"独立验证模式: {mode}",
            f"SAFE Score: {safe_score:.1%} ({supported}/{total} 事实被支持)",
        ]
        if unsupported:
            parts.append(f"不支持: {unsupported} 条")
        if irrelevant:
            parts.append(f"无法判断: {irrelevant} 条")
        if kb_comparison:
            parts.append(f"\n知识库对比: {kb_comparison['assessment']}")
            parts.append(
                f"  KB FActScore={kb_comparison['kb_factscore']:.1%} vs "
                f"SAFE Score={kb_comparison['safe_score']:.1%} "
                f"(差值={kb_comparison['score_diff']:+.1%})"
            )
        return "\n".join(parts)

    def clear_cache(self):
        self._cache.clear()
        self._decompose_cache.clear()
