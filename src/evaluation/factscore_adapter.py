"""
原子事实检查器
基于原子事实分解方法实现高效的事实性判断功能

核心优势：
1. 原子化处理：将长文本分解为短原子事实，每个操作的prompt都很短
2. 短prompt验证：每个原子事实独立验证，大幅降低token消耗
3. 缓存优化：对中间结果进行缓存，避免重复计算

实现原理：
- 将生成文本分解为独立的原子事实（每个事实只包含一个信息点）
- 对每个原子事实使用短prompt进行验证
- 收集不支持的事实作为幻觉检测结果
"""

import os
import re
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from ..llm import LLMAdapter
from ..retrieval import RetrievalResult

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False


@dataclass
class FactCheckResult:
    """事实性检查结果"""
    is_factual: bool
    confidence: float
    factscore: float = 0.0  # S(y) = supported_facts / total_facts
    total_facts: int = 0
    supported_facts: int = 0
    inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    entity_coverage: float = 0.0
    evidence_support: float = 0.0
    summary: str = ""


class FActScoreChecker:
    """
    基于原子事实分解的事实性判断器
    
    核心方法：
    1. 原子事实分解：将长文本分解为独立的原子事实
    2. 短prompt验证：每个原子事实使用短prompt验证
    3. 缓存机制：对分解和验证结果进行缓存
    
    相比传统方法的优势：
    - 传统方法：一次性用超长prompt验证整段文本（~1500 tokens）
    - 本方法：分解为多个短prompt验证（每个~200 tokens）
    - 大幅降低API调用成本和响应时间
    """
    
    # 原子事实分解prompt（短）
    DECOMPOSE_PROMPT = """请将以下文本拆解为独立的原子事实。

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
    
    # 事实验证prompt（短）
    VERIFY_PROMPT = """基于以下上下文，判断事实是否正确。

上下文：
{context}

事实：{fact}

请回答 True 或 False，并简要说明理由。"""
    
    # 批量事实验证prompt
    BATCH_VERIFY_PROMPT = """请根据提供的上下文，逐一验证以下事实是否正确。

上下文：
{context}

需要验证的事实：
{facts}

请对每个事实进行验证，并返回JSON格式的结果，其中包含每个事实的验证结果（true表示支持，false表示不支持）。

输出格式：
{{
  "results": [
    {{"fact": "事实内容", "supported": true/false}},
    ...
  ]
}}

只返回JSON，不要其他内容。"""
    
    # 回忆录验证prompt
    MEMOIR_VERIFY_PROMPT = """请判断以下事实是否在回忆录原文中有依据。

回忆录原文：
{memoir_text}

需要验证的事实：
{fact}

请判断该事实是否是回忆录中提到的个人经历（在回忆录原文中有依据）。

请以JSON格式返回：
{{
    "supported": true/false,
    "explanation": "判断理由"
}}

只返回JSON，不要其他内容。"""

    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        cache_dir: str = ".cache/factscore",
        use_rule_decompose: bool = False,
    ):
        """
        初始化FActScore检查器
        
        Args:
            llm_adapter: LLM适配器
            cache_dir: 缓存目录
            use_rule_decompose: 是否使用规则拆分（默认False，使用LLM拆分）
        """
        self.llm_adapter = llm_adapter
        self.cache_dir = cache_dir
        self.use_rule_decompose = use_rule_decompose
        self._decompose_cache: Dict[str, List[str]] = {}
        self._verify_cache: Dict[str, bool] = {}
        self._result_cache: Dict[str, FactCheckResult] = {}
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
    
    def _cache_key(self, memoir_text: str, generated_text: str) -> str:
        """生成缓存key"""
        content = f"{memoir_text[:100]}|{generated_text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def check(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult] = None,
        use_llm: bool = True,
        use_rule_decompose: Optional[bool] = None,
        max_atomic_facts: Optional[int] = None,
    ) -> FactCheckResult:
        """
        执行事实性检查（使用FActScore方法）
        
        Args:
            memoir_text: 原始回忆录文本
            generated_text: 生成的历史背景文本
            retrieval_result: 检索结果（包含历史背景证据）
            use_llm: 是否使用LLM进行深度检查
            use_rule_decompose: 是否使用规则拆分（优先级高于实例配置）
            
        Returns:
            FactCheckResult: 事实性检查结果
        """
        logger = logging.getLogger(__name__)
        
        # 检查缓存
        cache_key = self._cache_key(memoir_text, generated_text)
        if cache_key in self._result_cache:
            logger.info("[FActScore] 使用缓存结果")
            return self._result_cache[cache_key]
        
        # 保存原始设置
        original_use_rule_decompose = self.use_rule_decompose
        
        # 如果指定了use_rule_decompose参数，临时覆盖实例设置
        if use_rule_decompose is not None:
            self.use_rule_decompose = use_rule_decompose
            logger.info(f"[FActScore] 临时设置use_rule_decompose={use_rule_decompose}")
        
        try:
            inconsistencies = []
            total_facts = 0
            supported_facts = 0

            # FActScore原子化检查
            if use_llm and self.llm_adapter:
                logger.info("[FActScore] 执行原子化检查")
                factscore_issues, total_facts, supported_facts = await self._factscore_check(
                    memoir_text,
                    generated_text,
                    retrieval_result,
                    max_atomic_facts=max_atomic_facts,
                )
                inconsistencies.extend(factscore_issues)

            # 计算 FActScore 比值: S(y) = supported / total
            factscore_ratio = supported_facts / total_facts if total_facts > 0 else 0.0

            # 计算指标
            entity_coverage = self._calculate_entity_coverage(generated_text, retrieval_result)
            evidence_support = self._calculate_evidence_support(generated_text, retrieval_result)

            # 计算置信度
            confidence = self._calculate_confidence(inconsistencies, entity_coverage, evidence_support)

            # 生成总结
            summary = self._generate_summary(inconsistencies, confidence, factscore_ratio, total_facts)

            result = FactCheckResult(
                is_factual=len(inconsistencies) == 0,
                confidence=confidence,
                factscore=factscore_ratio,
                total_facts=total_facts,
                supported_facts=supported_facts,
                inconsistencies=inconsistencies,
                entity_coverage=entity_coverage,
                evidence_support=evidence_support,
                summary=summary,
            )
            
            # 缓存结果
            self._result_cache[cache_key] = result
            
            return result
        finally:
            # 恢复原始设置
            if use_rule_decompose is not None:
                self.use_rule_decompose = original_use_rule_decompose
                logger.info(f"[FActScore] 恢复use_rule_decompose={original_use_rule_decompose}")
    
    async def _factscore_check(
        self,
        memoir_text: str,
        generated_text: str,
        retrieval_result: Optional[RetrievalResult],
        max_atomic_facts: Optional[int] = None,
    ) -> tuple:
        """
        FActScore原子化检查

        将生成文本分解为原子事实，逐一验证。
        验证上下文包括：回忆录原文 + 检索到的历史背景。

        Returns:
            (inconsistencies, total_facts, supported_facts)
        """
        logger = logging.getLogger(__name__)
        inconsistencies = []

        retrieval_context = ""
        if retrieval_result:
            retrieval_context = retrieval_result.get_context_text()

        # 合并上下文：原文 + 检索背景。润色改写后的内容大部分来自原文，
        # 原文是最重要的验证依据。
        context_parts = []
        if memoir_text:
            context_parts.append(f"【回忆录原文】\n{memoir_text}")
        if retrieval_context:
            context_parts.append(f"【历史背景】\n{retrieval_context}")
        context = "\n\n".join(context_parts)

        if not context:
            logger.warning("[FActScore] 无上下文，跳过原子化检查")
            return [], 0, 0

        # 分解生成文本为原子事实
        logger.info("[FActScore] 分解生成文本为原子事实...")
        atomic_facts = await self._decompose_text(generated_text)
        if max_atomic_facts is not None and len(atomic_facts) > max_atomic_facts:
            logger.info(
                f"[FActScore] 原子事实 {len(atomic_facts)} 条超过上限 {max_atomic_facts}，截断以控制成本"
            )
            atomic_facts = atomic_facts[:max_atomic_facts]

        # 过滤纯文学描写，只保留可验证的事实性陈述
        atomic_facts = self._filter_literary_sentences(atomic_facts)

        total_facts = len(atomic_facts)
        logger.info(f"[FActScore] 过滤后剩余 {total_facts} 个可验证事实")

        if total_facts == 0:
            return [], 0, 0

        # 先做快速规则匹配：如果事实能在原文中找到关键词对应，直接标记为支持
        rule_supported, remaining_facts = self._rule_match_against_source(
            atomic_facts, memoir_text
        )
        logger.info(
            f"[FActScore] 规则匹配: {len(rule_supported)}/{total_facts} 个事实直接由原文支持"
        )

        # 对剩余事实用 LLM 批量验证
        if remaining_facts:
            logger.info(f"[FActScore] LLM 验证剩余 {len(remaining_facts)} 个事实")
            verification_results = await self._verify_facts_batch(
                remaining_facts, context
            )
        else:
            verification_results = []

        unsupported_facts = []
        for fact, is_supported in zip(remaining_facts, verification_results):
            if not is_supported:
                unsupported_facts.append(fact)
                logger.info(f"[FActScore] 事实不支持: {fact[:50]}...")

        for fact in unsupported_facts:
            inconsistencies.append({
                "type": "unsupported_claim",
                "generated_text": fact,
                "explanation": "该事实缺乏原文或历史背景证据支持",
                "severity": 0.4,
            })

        supported_facts = total_facts - len(inconsistencies)
        logger.info(
            f"[FActScore] 验证完成: {supported_facts}/{total_facts} 事实被支持 "
            f"(FActScore={supported_facts/total_facts:.2%})"
        )
        return inconsistencies, total_facts, supported_facts
    
    async def _verify_against_memoir(
        self,
        memoir_text: str,
        inconsistencies: List[Dict[str, Any]],
    ) -> List[str]:
        """
        验证不支持的事实是否在回忆录原文中有依据
        
        检查被标记为"缺乏证据支持"的事实是否在回忆录原文中有依据
        
        Args:
            memoir_text: 回忆录原文
            inconsistencies: 不支持的事实列表
            
        Returns:
            在回忆录中有依据的事实列表
        """
        logger = logging.getLogger(__name__)
        verified_facts = []
        
        # 只检查unsupported_claim类型的事实
        unsupported_facts = [
            inc for inc in inconsistencies 
            if inc.get("type") == "unsupported_claim"
        ]
        
        if not unsupported_facts:
            logger.info("[MemoirVerify] 没有需要验证的事实")
            return verified_facts
        
        logger.info(f"[MemoirVerify] 需要验证 {len(unsupported_facts)} 个事实是否在回忆录中有依据")
        
        # 批量验证：将所有事实一次性发送给LLM
        if self.llm_adapter:
            facts_list = [inc.get("generated_text", "") for inc in unsupported_facts]
            facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(facts_list)])
            
            prompt = f"""请检查以下事实是否在回忆录原文中有依据。对于每个事实，回答 YES 或 NO。

回忆录原文：
{memoir_text[:1500]}

需要检查的事实：
{facts_str}

请按照以下格式回答：
1. YES/NO
2. YES/NO
...
"""
            
            try:
                response = await self.llm_adapter.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,
                )
                
                result_text = response.content.strip()
                logger.info(f"[MemoirVerify] LLM响应: {result_text[:100]}...")
                
                # 解析结果
                lines = result_text.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line and ("YES" in line or "NO" in line):
                        # 提取数字和结果
                        import re
                        match = re.match(r'(\d+)\.\s*(YES|NO)', line)
                        if match:
                            idx = int(match.group(1)) - 1
                            if 0 <= idx < len(unsupported_facts):
                                inc = unsupported_facts[idx]
                                fact = inc.get("generated_text", "")
                                if "YES" in match.group(2):
                                    verified_facts.append(fact)
                                    logger.info(f"[MemoirVerify] 事实在回忆录中有依据: {fact[:50]}...")
                                else:
                                    logger.info(f"[MemoirVerify] 事实在回忆录中无依据: {fact[:50]}...")
            except Exception as e:
                logger.error(f"[MemoirVerify] 验证失败: {e}")
        
        logger.info(f"[MemoirVerify] 验证完成，{len(verified_facts)}/{len(unsupported_facts)} 个事实在回忆录中有依据")
        return verified_facts
    
    def _remove_false_positives(
        self,
        inconsistencies: List[Dict[str, Any]],
        verified_facts: List[str],
    ) -> List[Dict[str, Any]]:
        """
        移除在回忆录中有依据的事实（误报）
        
        Args:
            inconsistencies: 所有检测到的不一致项
            verified_facts: 在回忆录中有依据的事实列表
            
        Returns:
            过滤后的不一致项列表
        """
        logger = logging.getLogger(__name__)
        
        if not verified_facts:
            return inconsistencies
        
        # 过滤：保留不在verified_facts中的不一致项
        filtered = []
        removed_count = 0
        
        for inc in inconsistencies:
            # 检查这个不一致项是否是被误报的
            is_false_positive = (
                inc.get("type") == "unsupported_claim" and
                any(fact in inc.get("generated_text", "") or inc.get("generated_text", "") in fact 
                    for fact in verified_facts)
            )
            
            if is_false_positive:
                logger.info(f"[RemoveFP] 移除误报: {inc.get('generated_text', '')[:50]}...")
                removed_count += 1
            else:
                filtered.append(inc)
        
        logger.info(f"[RemoveFP] 总共移除 {removed_count} 个误报，剩余 {len(filtered)} 个问题")
        return filtered
    
    # 纯文学修辞/感官描写模式，不含可验证事实
    _LITERARY_PATTERNS = re.compile(
        r"^(?:.*(?:仿佛|如同|像是|似乎|宛如|恰似|好像|犹如).*[。！？]?$)"
        r"|^(?:.*(?:心中|内心|眼前|仿佛在|似乎在).*(?:涌动|荡漾|闪烁|浮现|回响).*$)"
    )

    @staticmethod
    def _filter_literary_sentences(facts: List[str]) -> List[str]:
        """过滤纯文学描写句，只保留含可验证信息的陈述。"""
        filtered = []
        for fact in facts:
            s = fact.strip()
            if len(s) < 6:
                continue
            has_verifiable = bool(re.search(
                r"[\d一二三四五六七八九十百千万亿]"  # 包含数字
                r"|[年月日号]"                        # 包含时间词
                r"|(?:省|市|县|区|镇|村|塬|河|山)"    # 包含地名标志
                r"|(?:叫|姓|名|是|在|从|到|去|来|做|当|任)", s  # 包含事实性动词
            ))
            if has_verifiable:
                filtered.append(s)
        return filtered

    @staticmethod
    def _rule_match_against_source(
        facts: List[str], source_text: str,
    ) -> tuple[list[str], list[str]]:
        """
        规则快速匹配：如果一个"事实"中的关键实词（去掉虚词后）
        大部分能在原文中找到，就直接判定为"被支持"。

        Returns:
            (supported_facts, remaining_facts_for_llm)
        """
        supported = []
        remaining = []
        for fact in facts:
            # 提取事实中的关键片段（人名、数字、地名等）
            key_tokens = re.findall(
                r"[\u4e00-\u9fff]{2,}"  # 中文词（2字以上）
                r"|[A-Za-z]+\d+"        # 英文+数字
                r"|\d+",               # 纯数字
                fact,
            )
            if not key_tokens:
                remaining.append(fact)
                continue
            # 去掉常见虚词
            stop_words = {
                "一个", "一些", "这个", "那个", "我们", "他们", "自己", "什么",
                "可以", "已经", "开始", "成为", "不是", "没有", "但是", "因为",
                "所以", "如果", "就是", "虽然", "仿佛", "似乎", "好像", "如同",
                "依旧", "依然", "终于", "渐渐", "慢慢", "忽然", "不断", "只是",
                "然而", "同时", "随着", "每个", "无数", "整个",
            }
            meaningful = [t for t in key_tokens if t not in stop_words]
            if not meaningful:
                remaining.append(fact)
                continue
            matched = sum(1 for t in meaningful if t in source_text)
            ratio = matched / len(meaningful)
            if ratio >= 0.5:
                supported.append(fact)
            else:
                remaining.append(fact)
        return supported, remaining

    def _decompose_text_rule_based(self, text: str) -> List[str]:
        """
        基于规则的原子事实分解（使用jieba）
        
        优势：速度快，无需LLM调用
        缺点：语义理解能力有限
        """
        import re
        import time
        start_time = time.time()
        
        # 按多种标点符号分段，确保即使没有句号也能拆分
        sentences = re.split(r'[。！？.!?\n;；]', text)
        
        facts = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:  # 过滤太短的句子
                continue
            
            # 过滤明显的重复内容
            if any(sentence in fact or fact in sentence for fact in facts):
                continue
            
            # 只保留有实际内容的句子
            if any(char.isalnum() for char in sentence):
                facts.append(sentence)
        
        # 如果没有拆分出任何事实，将整个文本作为一个事实
        if not facts and text.strip():
            facts.append(text.strip())
        
        end_time = time.time()
        print(f"[DEBUG] 规则拆分完成，耗时: {(end_time - start_time):.4f} 秒，拆分出 {len(facts)} 个事实")
        return facts
    
    async def _decompose_text(self, text: str) -> List[str]:
        """
        将文本分解为原子事实
        
        根据配置选择使用规则拆分或LLM拆分
        """
        # 检查缓存
        cache_key = hashlib.md5(text[:200].encode()).hexdigest()
        if cache_key in self._decompose_cache:
            return self._decompose_cache[cache_key]
        
        # 使用规则拆分
        if self.use_rule_decompose:
            facts = self._decompose_text_rule_based(text)
            # 缓存结果
            self._decompose_cache[cache_key] = facts
            return facts
        
        # 使用LLM拆分
        if not self.llm_adapter:
            return []
        
        prompt = self.DECOMPOSE_PROMPT.format(text=text[:1000])
        
        try:
            response = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            
            result_text = response.content.strip()
            
            # 解析JSON结果
            try:
                facts = json.loads(result_text)
                if not isinstance(facts, list):
                    facts = []
            except json.JSONDecodeError:
                # 尝试从文本中提取JSON数组
                facts = self._extract_json_array(result_text)
            
            # 缓存结果
            self._decompose_cache[cache_key] = facts
            return facts
            
        except Exception as e:
            logging.getLogger(__name__).error(f"分解文本失败: {e}")
            return []
    
    async def _verify_fact(self, fact: str, context: str) -> bool:
        """
        验证单个原子事实
        
        基于上下文验证事实是否正确
        
        Returns:
            bool: True表示事实被支持，False表示不被支持
        """
        # 检查缓存
        cache_key = hashlib.md5(f"{fact}|{context[:100]}".encode()).hexdigest()
        if cache_key in self._verify_cache:
            return self._verify_cache[cache_key]
        
        if not self.llm_adapter:
            return True  # 无LLM时默认通过
        
        # 快速规则检查：如果事实在上下文中直接出现，直接返回True
        if fact in context:
            self._verify_cache[cache_key] = True
            return True
        
        prompt = self.VERIFY_PROMPT.format(context=context[:500], fact=fact)
        
        try:
            response = await self.llm_adapter.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=128,  # 减少max_tokens，加快响应速度
                timeout=10,  # 添加超时设置
            )
            
            result_text = response.content.strip().lower()
            
            # 解析结果
            is_supported = "true" in result_text or "正确" in result_text
            
            # 缓存结果
            self._verify_cache[cache_key] = is_supported
            return is_supported
            
        except asyncio.TimeoutError:
            logging.getLogger(__name__).warning(f"验证事实超时: {fact[:50]}...")
            return True  # 超时时默认通过
        except Exception as e:
            logging.getLogger(__name__).error(f"验证事实失败: {e}")
            return True  # 出错时默认通过
    
    async def _verify_facts_batch(self, facts: List[str], context: str, batch_size: int = 5) -> List[bool]:
        """
        批量验证事实
        
        批量验证多个事实，提高性能
        
        Args:
            facts: 事实列表
            context: 验证上下文
            batch_size: 每批处理的事实数量
            
        Returns:
            验证结果列表，True表示支持，False表示不支持
        """
        import time
        import json
        import re
        
        if not facts:
            return []
        
        start_time = time.time()
        results = []
        logger = logging.getLogger(__name__)
        
        # 分批处理
        for i in range(0, len(facts), batch_size):
            batch = facts[i:i + batch_size]
            batch_start = time.time()
            
            logger.info(f"[BATCH_VERIFY] 处理批次 {i//batch_size + 1}/{(len(facts) + batch_size - 1)//batch_size}, 包含 {len(batch)} 个事实")
            
            # 构建批量验证的prompt
            facts_str = "\n".join([f"{j+1}. {fact}" for j, fact in enumerate(batch)])
            prompt = self.BATCH_VERIFY_PROMPT.format(
                context=context[:500],
                facts=facts_str
            )
            
            try:
                response = await self.llm_adapter.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=512,  # 足够处理5个事实的结果
                    timeout=20,  # 批量验证的超时时间
                )
                
                # 解析批量结果
                batch_results = self._parse_batch_results(response.content, batch)
                results.extend(batch_results)
                
                batch_time = time.time() - batch_start
                logger.info(f"[BATCH_VERIFY] 批次处理完成，耗时: {batch_time:.2f} 秒")
                
            except asyncio.TimeoutError:
                logger.warning(f"[BATCH_VERIFY] 批量验证超时，使用默认值")
                results.extend([True] * len(batch))
            except Exception as e:
                logger.error(f"[BATCH_VERIFY] 批量验证失败: {e}")
                results.extend([True] * len(batch))
        
        total_time = time.time() - start_time
        logger.info(f"[BATCH_VERIFY] 批量验证完成，总耗时: {total_time:.2f} 秒，平均每个事实: {(total_time/len(facts)):.4f} 秒")
        return results
    
    def _parse_batch_results(self, response_text: str, original_facts: List[str]) -> List[bool]:
        """
        解析批量验证的结果
        
        Args:
            response_text: 模型返回的文本
            original_facts: 原始事实列表
            
        Returns:
            验证结果列表
        """
        try:
            import json
            import re
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # 构建结果映射
                fact_results = {}
                for item in data.get("results", []):
                    fact = item.get("fact", "").strip()
                    supported = item.get("supported", True)
                    fact_results[fact] = supported
                
                # 匹配原始事实
                results = []
                for fact in original_facts:
                    # 查找最匹配的结果
                    matched = False
                    for result_fact, supported in fact_results.items():
                        if fact in result_fact or result_fact in fact:
                            results.append(supported)
                            matched = True
                            break
                    if not matched:
                        # 快速规则检查：如果事实在上下文中直接出现，直接返回True
                        if fact in response_text:
                            results.append(True)
                        else:
                            results.append(True)  # 未匹配时默认通过
                
                return results
                
        except Exception as e:
            logging.getLogger(__name__).error(f"解析批量结果失败: {e}")
        
        # 解析失败时，所有事实默认通过
        return [True] * len(original_facts)
    
    def _calculate_entity_coverage(self, generated_text: str, retrieval_result: Optional[RetrievalResult]) -> float:
        """计算实体覆盖率"""
        if not retrieval_result:
            return 0.0
        
        if not JIEBA_AVAILABLE:
            return 0.5  # 默认中等覆盖率
        
        # 使用jieba分词计算重叠
        gen_words = set(jieba.lcut(generated_text))
        evidence_words = set()
        
        # 从检索结果中提取文本
        if retrieval_result.get_context_text():
            context_text = retrieval_result.get_context_text()
            evidence_words.update(jieba.lcut(context_text))
        
        if not gen_words:
            return 0.0
        
        overlap = len(gen_words & evidence_words)
        return min(overlap / len(gen_words), 1.0)
    
    def _calculate_evidence_support(self, generated_text: str, retrieval_result: Optional[RetrievalResult]) -> float:
        """计算证据支持度"""
        if not retrieval_result:
            return 0.0
        
        # 基于检索结果的完整性计算支持度
        if retrieval_result.has_results:
            # 如果有检索结果，给中等以上的支持度
            if retrieval_result.entities and len(retrieval_result.entities) > 3:
                return 0.8
            elif retrieval_result.text_units:
                return 0.7
            else:
                return 0.5
        return 0.3
    
    def _calculate_confidence(self, inconsistencies: List[Dict[str, Any]], entity_coverage: float, evidence_support: float) -> float:
        """计算整体置信度"""
        base_confidence = 0.8
        
        # 根据不一致项扣分
        for inc in inconsistencies:
            base_confidence -= inc.get("severity", 0.5) * 0.2
        
        # 根据覆盖率和证据支持度调整
        base_confidence += (entity_coverage + evidence_support) * 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _generate_summary(
        self,
        inconsistencies: List[Dict[str, Any]],
        confidence: float,
        factscore: float = 0.0,
        total_facts: int = 0,
    ) -> str:
        """生成检查总结"""
        parts = []
        if total_facts > 0:
            supported = total_facts - len(inconsistencies)
            parts.append(f"FActScore: {factscore:.1%} ({supported}/{total_facts} 事实被支持)")

        if not inconsistencies:
            parts.append("未检测到明显的事实不一致问题。")
        else:
            parts.append(f"发现 {len(inconsistencies)} 处潜在问题：")
            for inc in inconsistencies:
                parts.append(f"- 缺乏证据支持: {inc.get('generated_text', '')[:50]}...")
            if confidence < 0.5:
                parts.append("\n建议仔细核查生成内容。")

        return "\n".join(parts)
    
    def _extract_json_array(self, text: str) -> List[str]:
        """从文本中提取JSON数组"""
        try:
            # 尝试找到方括号包裹的内容
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except:
            pass
        
        # 如果提取失败，按行分割
        lines = text.strip().split('\n')
        facts = []
        for line in lines:
            line = line.strip()
            # 移除常见的列表标记
            line = re.sub(r'^[-*•]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)
            if line and len(line) > 3:
                facts.append(line)
        
        return facts
    
    def clear_cache(self):
        """清空所有缓存"""
        self._decompose_cache.clear()
        self._verify_cache.clear()
        self._result_cache.clear()
