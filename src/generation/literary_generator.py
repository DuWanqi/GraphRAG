"""
文学润色生成器
将历史背景信息转化为具有文学性的描述文本
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from ..llm import LLMAdapter, LLMRouter, MultiLLMResponse, create_llm_adapter
from ..retrieval import RetrievalResult, MemoirContext


@dataclass
class GenerationResult:
    """生成结果"""
    content: str
    provider: str
    model: str
    memoir_context: Optional[MemoirContext] = None
    retrieval_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "memoir_context": self.memoir_context.to_dict() if self.memoir_context else None,
            "retrieval_info": self.retrieval_info,
        }


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
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一位优秀的回忆录作家和历史学者。你的任务是将历史背景信息融入个人回忆录中，使其既准确又富有文学性。

你的写作风格应该：
1. 使用温暖、怀旧的笔调，如同在回忆往事
2. 将宏大的历史事件与个人经历自然衔接
3. 使用生动的细节描写，让读者感同身受
4. 保持历史事实的准确性，不添加虚构的具体事件
5. 语言优美流畅，适合中国读者的阅读习惯

注意事项：
- 不要使用"据记载"、"历史上"等生硬的过渡词
- 要让历史背景与个人叙事融为一体
- 控制篇幅在1-3段，约200-500字"""
    
    # 生成提示词模板
    GENERATION_PROMPT = """请根据以下信息，为回忆录片段生成1-3段历史背景描述，使其能自然融入个人叙事中。

## 回忆录原文
{memoir_text}

## 时间背景
年份：{year}
地点：{location}

## 检索到的历史信息
{context}

## 要求
1. 生成的内容应该能够自然地衔接在回忆录原文之后或穿插其中
2. 使用文学性的语言，避免教科书式的叙述
3. 突出这段历史与个人经历的关联
4. 总字数控制在200-500字之间
5. 可以分成1-3个自然段

请直接输出生成的历史背景描述文本，不要包含任何解释或元信息。"""
    
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
    
    async def generate(
        self,
        memoir_text: str,
        retrieval_result: RetrievalResult,
        temperature: float = 0.7,
        max_tokens: int = 1024,
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
        
        # 构建提示词
        prompt = self._build_prompt(memoir_text, retrieval_result)
        
        # 调用LLM生成
        adapter = self.llm_adapter
        if adapter is None and self.llm_router:
            # 使用路由器中的第一个适配器
            adapters = self.llm_router.list_adapters()
            if adapters:
                adapter = self.llm_router.adapters[adapters[0]]
        
        if adapter is None:
            raise ValueError("没有可用的LLM适配器")
        
        response = await adapter.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return GenerationResult(
            content=response.content,
            provider=response.provider.value,
            model=response.model,
            memoir_context=retrieval_result.context,
            retrieval_info={
                "entities_count": len(retrieval_result.entities),
                "communities_count": len(retrieval_result.communities),
                "query": retrieval_result.query,
            }
        )
    
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
            system_prompt=self.SYSTEM_PROMPT,
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
                retrieval_info={
                    "entities_count": len(retrieval_result.entities),
                    "communities_count": len(retrieval_result.communities),
                    "query": retrieval_result.query,
                }
            )
        
        return MultiGenerationResult(
            results=results,
            errors=multi_response.errors,
            memoir_context=retrieval_result.context,
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
    ) -> str:
        """构建生成提示词"""
        context = retrieval_result.context
        
        return self.GENERATION_PROMPT.format(
            memoir_text=memoir_text,
            year=context.year or "未知",
            location=context.location or "未知",
            context=retrieval_result.get_context_text() or "暂无相关历史信息",
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
