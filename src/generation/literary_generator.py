"""
文学润色生成器
将历史背景信息转化为具有文学性的描述文本
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, AsyncIterator, TYPE_CHECKING

from ..llm import LLMAdapter, LLMRouter, create_llm_adapter
from ..retrieval import RetrievalResult, MemoirContext

if TYPE_CHECKING:
    from ..retrieval import MemoirRetriever
    from .long_form_orchestrator import LongFormGenerationResult

from .prompts import PromptTemplates, get_system_prompt


@dataclass
class GenerationResult:
    """生成结果"""
    content: str
    provider: str
    model: str
    memoir_context: Optional[MemoirContext] = None
    retrieval_info: Optional[Dict[str, Any]] = None
    novel_content_brief: Optional[Any] = None  # NovelContentBrief from novel_content_extractor

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "memoir_context": self.memoir_context.to_dict() if self.memoir_context else None,
            "retrieval_info": self.retrieval_info,
        }
        if self.novel_content_brief:
            result["novel_content_brief"] = {
                "has_novel_content": self.novel_content_brief.has_novel_content,
                "novel_entity_count": len(self.novel_content_brief.novel_entities),
                "novel_relationship_count": len(self.novel_content_brief.novel_relationships),
                "summary": self.novel_content_brief.summary,
            }
        return result


@dataclass
class MultiGenerationResult:
    """多LLM生成结果（用于对比）"""
    results: Dict[str, GenerationResult]
    errors: Dict[str, str]
    memoir_context: Optional[MemoirContext] = None
    
    def get_best(self) -> Optional[GenerationResult]:
        """获取第一个成功的结果"""
        for result in self.results.values():
            return result
        return None


class LiteraryGenerator:
    """
    文学润色生成器
    
    功能：
    1. 接收检索到的历史背景信息
    2. 使用LLM生成具有文学性的描述
    3. 确保生成内容与回忆录风格融合
    4. 支持多LLM并行生成对比
    """
    
    DEFAULT_SYSTEM_PROMPT_KEY = "default"
    
    def __init__(
        self,
        llm_adapter: Optional[LLMAdapter] = None,
        llm_router: Optional[LLMRouter] = None,
    ):
        """
        初始化生成器
        
        Args:
            llm_adapter: 单一LLM适配器
            llm_router: 多LLM路由器（用于对比生成）
        """
        self.llm_adapter = llm_adapter
        self.llm_router = llm_router

    def _build_retrieval_info(self, retrieval_result: RetrievalResult) -> Dict[str, Any]:
        """构建检索信息元数据（避免重复代码）"""
        return {
            "entities_count": len(retrieval_result.entities),
            "communities_count": len(retrieval_result.communities),
            "query": retrieval_result.query,
        }

    async def generate(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        style: str = "standard",
        length_hint: str = "200-500字",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        chapter_context: str = "",
    ) -> GenerationResult:
        """
        生成历史背景描述
        
        Args:
            memoir_text: 回忆录原文
            retrieval_result: 检索结果
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            GenerationResult: 生成结果
        """
        if not self.llm_adapter and not self.llm_router:
            raise ValueError("需要提供llm_adapter或llm_router")
        
        timing = os.getenv("TEMP_TIMING") == "1"
        t0 = time.perf_counter()

        # 构建提示词
        t_prompt0 = time.perf_counter()
        prompt = self._build_prompt(memoir_text, retrieval_result, style=style, length_hint=length_hint, chapter_context=chapter_context)
        t_prompt = time.perf_counter() - t_prompt0
        if timing:
            print(f"[TEMP_TIMING] generator.build_prompt={t_prompt:.3f}s prompt_chars={len(prompt)}")
        
        # 调用LLM生成
        adapter = self.llm_adapter
        if adapter is None and self.llm_router:
            # 使用路由器中的第一个适配器
            adapters = self.llm_router.list_adapters()
            if adapters:
                adapter = self.llm_router.adapters[adapters[0]]
        
        if adapter is None:
            raise ValueError("没有可用的LLM适配器")
        
        t_llm0 = time.perf_counter()
        response = await adapter.generate(
            prompt=prompt,
            system_prompt=get_system_prompt(self.DEFAULT_SYSTEM_PROMPT_KEY),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        t_llm = time.perf_counter() - t_llm0
        if timing:
            total = time.perf_counter() - t0
            print(
                f"[TEMP_TIMING] generator.llm_call={t_llm:.3f}s total={total:.3f}s "
                f"provider={response.provider.value} model={response.model}"
            )
        
        return GenerationResult(
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            memoir_context=retrieval_result.context,
            retrieval_info=self._build_retrieval_info(retrieval_result),
            novel_content_brief=getattr(retrieval_result, '_novel_content_brief', None),
        )

    async def generate_stream(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        style: str = "standard",
        length_hint: str = "200-500字",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        chapter_context: str = "",
    ) -> AsyncIterator[str]:
        """
        流式生成：逐步产出新增文本片段（delta）。
        """
        if not self.llm_adapter and not self.llm_router:
            raise ValueError("需要提供llm_adapter或llm_router")

        prompt = self._build_prompt(memoir_text, retrieval_result, style=style, length_hint=length_hint, chapter_context=chapter_context)

        adapter = self.llm_adapter
        if adapter is None and self.llm_router:
            adapters = self.llm_router.list_adapters()
            if adapters:
                adapter = self.llm_router.adapters[adapters[0]]
        if adapter is None:
            raise ValueError("没有可用的LLM适配器")

        if not hasattr(adapter, "generate_stream"):
            raise ValueError("当前适配器不支持流式生成")

        async for delta in adapter.generate_stream(
            prompt=prompt,
            system_prompt=get_system_prompt(self.DEFAULT_SYSTEM_PROMPT_KEY),
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if delta:
                yield delta
    
    async def generate_parallel(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        providers: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> MultiGenerationResult:
        """
        使用多个LLM并行生成（用于对比评估）
        
        Args:
            memoir_text: 回忆录原文
            retrieval_result: 检索结果
            providers: 要使用的提供商列表
            temperature: 生成温度
            max_tokens: 最大token数
            
        Returns:
            MultiGenerationResult: 多LLM生成结果
        """
        if not self.llm_router:
            raise ValueError("并行生成需要提供llm_router")
        
        prompt = self._build_prompt(memoir_text, retrieval_result)
        
        # 并行调用多个LLM
        multi_response = await self.llm_router.generate_parallel(
            prompt=prompt,
            providers=providers,
            system_prompt=get_system_prompt(self.DEFAULT_SYSTEM_PROMPT_KEY),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # 转换结果
        results = {}
        for provider_name, response in multi_response.responses.items():
            results[provider_name] = GenerationResult(
                content=response.content,
                provider=response.provider.value,
                model=response.model,
                memoir_context=retrieval_result.context,
                retrieval_info=self._build_retrieval_info(retrieval_result)
            )
        
        return MultiGenerationResult(
            results=results,
            errors=multi_response.errors,
            memoir_context=retrieval_result.context,
        )

    async def generate_long_form(
        self,
        memoir_text: str,
        retriever: "MemoirRetriever",
        **kwargs: Any,
    ) -> "LongFormGenerationResult":
        """分章检索与生成；参数见 run_long_form_generation。"""
        from .long_form_orchestrator import run_long_form_generation

        return await run_long_form_generation(
            memoir_text,
            retriever,
            self,
            **kwargs,
        )

    def generate_sync(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """同步版本的生成"""
        return asyncio.run(self.generate(memoir_text, retrieval_result, temperature))
    
    def _build_prompt(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        style: str = "standard",
        length_hint: str = "200-500字",
        chapter_context: str = "",
    ) -> str:
        """构建生成提示词"""
        from .novel_content_extractor import extract_novel_content

        context = retrieval_result.context

        # 提取并分类 RAG 内容
        novel_brief = extract_novel_content(memoir_text, retrieval_result)

        # 将 novel_brief 附加到 retrieval_result（供后续评估使用）
        retrieval_result._novel_content_brief = novel_brief

        # 格式化为 prompt 注入用的两个区块
        formatted = novel_brief.format_for_prompt()

        template = PromptTemplates.get_template(style=style)
        return template.format(
            memoir_text=memoir_text,
            year=context.year or "未知",
            location=context.location or "未知",
            aligned_context=formatted["aligned_context"],
            novel_context=formatted["novel_context"],
            length_hint=length_hint,
            chapter_context=chapter_context,
        )
    
    async def enhance_memoir(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        position: str = "after",
        temperature: float = 0.7,
    ) -> str:
        """
        增强回忆录文本
        
        将生成的历史背景插入回忆录中
        
        Args:
            memoir_text: 回忆录原文
            retrieval_result: 检索结果
            position: 插入位置 ("before", "after", "interspersed")
            temperature: 生成温度
            
        Returns:
            str: 增强后的完整文本
        """
        result = await self.generate(memoir_text, retrieval_result, temperature)
        
        if position == "before":
            return f"{result.content}\n\n{memoir_text}"
        elif position == "after":
            return f"{memoir_text}\n\n{result.content}"
        else:  # interspersed - 需要更复杂的处理
            # 简单实现：在原文后追加
            return f"{memoir_text}\n\n{result.content}"
