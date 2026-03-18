"""
统一的多LLM适配器
支持 Deepseek, Qwen3, 混元, Google Gemini, 智谱GLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator
import asyncio

import litellm
from litellm import acompletion, completion, aembedding, embedding


class LLMProvider(Enum):
    """支持的LLM提供商"""
    DEEPSEEK = "deepseek"
    QWEN = "qwen"
    HUNYUAN = "hunyuan"
    GEMINI = "gemini"
    GLM = "glm"  # 智谱AI
    OPENAI = "openai"
    OLLAMA = "ollama"


@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        """获取总token数"""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return 0


class LLMAdapter(ABC):
    """LLM适配器抽象基类"""
    
    def __init__(
        self,
        api_key: str,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.extra_config = kwargs
    
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """返回提供商类型"""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """返回默认模型名称"""
        pass
    
    @abstractmethod
    def _get_litellm_model_name(self) -> str:
        """获取LiteLLM格式的模型名称"""
        pass
    
    def _get_model(self) -> str:
        """获取当前使用的模型"""
        return self.model or self.default_model
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        异步聊天接口
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成token数
            
        Returns:
            LLMResponse: 响应结果
        """
        litellm_model = self._get_litellm_model_name()
        
        # 配置API密钥和基础URL
        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base
        
        response = await acompletion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._get_model(),
            provider=self.provider,
            usage=dict(response.usage) if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        同步聊天接口
        """
        litellm_model = self._get_litellm_model_name()
        
        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base
        
        response = completion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._get_model(),
            provider=self.provider,
            usage=dict(response.usage) if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        简化的生成接口
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示（可选）
            
        Returns:
            LLMResponse: 响应结果
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return await self.chat(messages, temperature, max_tokens, **kwargs)
    
    async def get_embedding(
        self,
        text: str,
    ) -> List[float]:
        """
        获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            List[float]: 向量表示
        """
        try:
            litellm_model = self._get_litellm_model_name()
            
            embedding_model = self._get_embedding_model()
            
            response = await aembedding(
                model=embedding_model,
                input=[text],
                api_key=self.api_key,
                api_base=self.api_base,
            )
            
            return response.data[0]["embedding"]
            
        except Exception as e:
            raise RuntimeError(f"获取embedding失败: {e}")
    
    def _get_embedding_model(self) -> str:
        """获取embedding模型名称"""
        return self.extra_config.get("embedding_model", "text-embedding-ada-001")


class DeepseekAdapter(LLMAdapter):
    """Deepseek适配器"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.DEEPSEEK
    
    @property
    def default_model(self) -> str:
        return "deepseek-chat"
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        return f"deepseek/{model}"


class QwenAdapter(LLMAdapter):
    """阿里云通义千问适配器"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.QWEN
    
    @property
    def default_model(self) -> str:
        return "qwen-plus"
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        # Qwen通过OpenAI兼容接口调用
        return f"openai/{model}"


class HunyuanAdapter(LLMAdapter):
    """腾讯混元适配器"""
    
    def __init__(
        self,
        api_key: str,
        secret_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(api_key, api_base, model, **kwargs)
        self.secret_id = secret_id
        self.secret_key = secret_key
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.HUNYUAN
    
    @property
    def default_model(self) -> str:
        return "hunyuan-lite"
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        # 混元通过OpenAI兼容接口或专用接口
        return f"hunyuan/{model}"


class GeminiAdapter(LLMAdapter):
    """Google Gemini适配器"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GEMINI
    
    @property
    def default_model(self) -> str:
        return "gemini-2.5-flash"
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        # LiteLLM 使用 gemini/ 前缀调用 Google Generative AI
        return f"gemini/{model}"
    
    def _get_embedding_model(self) -> str:
        """Gemini使用text-embedding-004"""
        return self.extra_config.get("embedding_model", "text-embedding-004")
    
    def _get_safety_settings(self) -> List[Dict[str, str]]:
        """
        获取Gemini安全设置
        
        将所有安全类别的阈值设置为OFF，避免内容被截断或过滤。
        这对于生成长文本和处理各种主题的内容非常重要。
        """
        return [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "OFF"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "OFF"
            }
        ]
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        异步聊天接口（Gemini专用，包含安全设置）
        """
        litellm_model = self._get_litellm_model_name()
        
        # 配置API密钥和基础URL
        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base
        
        # 添加安全设置，避免内容被截断
        extra_params["safety_settings"] = self._get_safety_settings()
        
        response = await acompletion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._get_model(),
            provider=self.provider,
            usage=dict(response.usage) if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )
    
    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        同步聊天接口（Gemini专用，包含安全设置）
        """
        litellm_model = self._get_litellm_model_name()
        
        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base
        
        # 添加安全设置，避免内容被截断
        extra_params["safety_settings"] = self._get_safety_settings()
        
        response = completion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._get_model(),
            provider=self.provider,
            usage=dict(response.usage) if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )


class GLMAdapter(LLMAdapter):
    """智谱AI GLM适配器"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.GLM
    
    @property
    def default_model(self) -> str:
        return "glm-4.7-flash"  # 智谱GLM-4.7-FLASH，普惠模型，免费调用
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        # LiteLLM 使用 openai/ 前缀调用智谱API
        return f"openai/{model}"
    
    @property
    def default_api_base(self) -> str:
        """智谱API默认地址"""
        return "https://open.bigmodel.cn/api/paas/v4"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> LLMResponse:
        """
        异步聊天接口（GLM专用，关闭thinking模式）
        """
        litellm_model = self._get_litellm_model_name()
        
        # 配置API密钥和基础URL
        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base
        
        # 关闭GLM的thinking模式
        # 智谱API通过extra_body参数传递额外配置
        extra_params["extra_body"] = {
            "thinking": {
                "type": "disabled"  # 关闭深度思考模式
            }
        }
        
        response = await acompletion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_params,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self._get_model(),
            provider=self.provider,
            usage=dict(response.usage) if response.usage else None,
            finish_reason=response.choices[0].finish_reason
        )


class OpenAIAdapter(LLMAdapter):
    """OpenAI适配器（可选，用于兼容性）"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI
    
    @property
    def default_model(self) -> str:
        return "gpt-4o-mini"
    
    def _get_litellm_model_name(self) -> str:
        return self._get_model()


class OllamaAdapter(LLMAdapter):
    """Ollama本地模型适配器"""
    
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OLLAMA
    
    @property
    def default_model(self) -> str:
        return "qwen3:32b"
    
    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        return f"ollama/{model}"
