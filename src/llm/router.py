"""
多LLM路由器
支持同时调用多个LLM并行生成，用于对比评估
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from .adapter import LLMAdapter, LLMResponse, LLMProvider
from .factory import get_all_adapters, create_llm_adapter


@dataclass
class MultiLLMResponse:
    """多LLM响应结果"""
    responses: Dict[str, LLMResponse]  # 提供商名称 -> 响应
    errors: Dict[str, str]  # 提供商名称 -> 错误信息
    
    def get_best_response(self) -> Optional[LLMResponse]:
        """获取第一个成功的响应"""
        for response in self.responses.values():
            return response
        return None
    
    def get_response(self, provider: str) -> Optional[LLMResponse]:
        """获取指定提供商的响应"""
        return self.responses.get(provider)
    
    @property
    def success_count(self) -> int:
        """成功响应数量"""
        return len(self.responses)
    
    @property
    def error_count(self) -> int:
        """错误数量"""
        return len(self.errors)


class LLMRouter:
    """
    多LLM路由器
    
    用于管理多个LLM适配器，支持：
    - 单一LLM调用
    - 多LLM并行调用（用于对比评估）
    - 故障转移
    """
    
    def __init__(self, adapters: Optional[Dict[str, LLMAdapter]] = None):
        """
        初始化路由器
        
        Args:
            adapters: LLM适配器字典，如果不提供则自动加载所有可用适配器
        """
        self.adapters = adapters or get_all_adapters()
    
    def add_adapter(self, name: str, adapter: LLMAdapter):
        """添加适配器"""
        self.adapters[name] = adapter
    
    def remove_adapter(self, name: str):
        """移除适配器"""
        self.adapters.pop(name, None)
    
    def list_adapters(self) -> List[str]:
        """列出所有可用适配器"""
        return list(self.adapters.keys())
    
    async def generate(
        self,
        prompt: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        使用单一LLM生成
        
        Args:
            prompt: 用户提示
            provider: 指定提供商，如果不指定则使用第一个可用的
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLMResponse: 响应结果
        """
        if provider:
            if provider not in self.adapters:
                raise ValueError(f"提供商 {provider} 不可用。可用: {self.list_adapters()}")
            adapter = self.adapters[provider]
        else:
            if not self.adapters:
                raise ValueError("没有可用的LLM适配器")
            adapter = next(iter(self.adapters.values()))
        
        return await adapter.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def generate_parallel(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> MultiLLMResponse:
        """
        使用多个LLM并行生成
        
        Args:
            prompt: 用户提示
            providers: 要使用的提供商列表，如果不指定则使用所有可用的
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            MultiLLMResponse: 多LLM响应结果
        """
        if providers is None:
            providers = self.list_adapters()
        
        # 创建异步任务
        tasks = {}
        for provider_name in providers:
            if provider_name in self.adapters:
                adapter = self.adapters[provider_name]
                tasks[provider_name] = adapter.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
        
        # 并行执行
        responses = {}
        errors = {}
        
        results = await asyncio.gather(
            *[tasks[name] for name in tasks.keys()],
            return_exceptions=True
        )
        
        for provider_name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                errors[provider_name] = str(result)
            else:
                responses[provider_name] = result
        
        return MultiLLMResponse(responses=responses, errors=errors)
    
    async def generate_with_fallback(
        self,
        prompt: str,
        providers: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        使用故障转移策略生成
        
        按顺序尝试多个LLM，直到成功为止
        
        Args:
            prompt: 用户提示
            providers: 按优先级排序的提供商列表
            
        Returns:
            LLMResponse: 第一个成功的响应
            
        Raises:
            Exception: 如果所有LLM都失败
        """
        if providers is None:
            providers = self.list_adapters()
        
        last_error = None
        
        for provider_name in providers:
            if provider_name not in self.adapters:
                continue
            
            try:
                return await self.adapters[provider_name].generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                last_error = e
                continue
        
        raise last_error or ValueError("没有可用的LLM适配器")
