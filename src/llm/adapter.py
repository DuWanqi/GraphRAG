"""
统一的多LLM适配器
支持 Deepseek, Qwen3, 混元, Google Gemini, 智谱GLM
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator, Union
import asyncio

import litellm
from litellm import acompletion, completion, aembedding, embedding
import aiohttp

def _prettify_ollama_model_name(model: str) -> str:
    """
    将 Ollama 模型名转为更友好的展示名。
    例如: qwen3:32b -> Qwen3-32b, deepseek-r1:70b -> Deepseek-r1-70b
    """
    raw = (model or "").strip()
    if not raw:
        return raw
    # 有些模型可能形如 "library/qwen3:32b"
    raw = raw.split("/")[-1]
    raw = raw.replace(":", "-")
    return raw[:1].upper() + raw[1:]


async def fetch_ollama_model_names(api_base: str) -> List[str]:
    """
    从 Ollama /api/tags 拉取模型名称列表（name 字段）。
    """
    base = (api_base or "http://localhost:11434").rstrip("/")
    url = f"{base}/api/tags"
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Ollama /api/tags 返回 {resp.status}: {text}")
            payload = await resp.json()

    names: List[str] = []
    for m in (payload.get("models") or []):
        n = m.get("name")
        if isinstance(n, str) and n.strip():
            names.append(n.strip())
    return names


def list_ollama_models_sync(api_base: Optional[str] = None) -> List[str]:
    """
    同步获取 Ollama 模型列表（供 Gradio UI 构建阶段使用）。
    """
    base = (api_base or "http://localhost:11434").rstrip("/")
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        # 在已有事件循环中无法阻塞等待；这里退化为“空列表”，前端可通过刷新按钮重试。
        return []

    return asyncio.run(fetch_ollama_model_names(base))


def build_ollama_model_choices(api_base: Optional[str] = None) -> List[tuple]:
    """
    构建 Gradio Dropdown 友好的 choices: [(label, value), ...]
    label 形如: "Qwen3-32b (Ollama)"，value 为原始模型名: "qwen3:32b"
    """
    try:
        names = list_ollama_models_sync(api_base=api_base)
    except Exception:
        # Ollama 未运行时不阻塞 UI 启动
        names = []
    choices: List[tuple] = []
    for name in names:
        label = f"{_prettify_ollama_model_name(name)} (Ollama)"
        choices.append((label, name))
    return choices


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
    
    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        max_parse_retries: int = 2,
        json_pattern: str = r'\{.*\}',
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Chat 包装：要求 LLM 返回 JSON，仅对“模型输出格式问题”重试。

        - 解析失败 = 找不到 JSON 或 json.loads 抛异常 → 立即重试（无退避，
          因为这是模型采样非确定性，不是限流，等待无意义）
        - Transport 异常（429/5xx/超时/网络）由底层 chat() 通过 litellm
          num_retries 处理，**不**在本层叠加重试

        Args:
            max_parse_retries: 解析失败后额外尝试次数（默认 2 → 总共 ≤3 次调用）
            json_pattern: 提取 JSON 的正则。默认匹配对象 `{...}`，
                需要数组时传 `r'\\[.*\\]'`
        """
        import json as _json
        import re as _re

        last_err: Optional[Exception] = None
        for _ in range(max_parse_retries + 1):
            response = await self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            raw = (response.content or "").strip()
            match = _re.search(json_pattern, raw, _re.DOTALL)
            if match:
                try:
                    return _json.loads(match.group(0))
                except _json.JSONDecodeError as e:
                    last_err = e
            else:
                last_err = ValueError(
                    f"未找到匹配 {json_pattern} 的 JSON: {raw[:160]}"
                )
        raise RuntimeError(
            f"LLM JSON 解析连续 {max_parse_retries + 1} 次失败: {last_err}"
        )

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

        # 默认 transport 层重试 2 次（限流/5xx/超时由 LiteLLM 指数退避处理）
        kwargs.setdefault("num_retries", 2)

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

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式聊天接口：逐步 yield 新增文本片段（delta）。
        """
        litellm_model = self._get_litellm_model_name()

        extra_params = {}
        if self.api_key:
            extra_params["api_key"] = self.api_key
        if self.api_base:
            extra_params["api_base"] = self.api_base

        kwargs.setdefault("num_retries", 2)
        stream = await acompletion(
            model=litellm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **extra_params,
            **kwargs
        )

        async for chunk in stream:
            # 兼容不同 provider 的 chunk 结构
            delta = None
            try:
                # litellm 可能返回对象，也可能返回 dict（不同 provider/版本）
                if isinstance(chunk, dict):
                    # OpenAI 兼容结构：{"choices":[{"delta":{"content":...}}]}
                    choices = chunk.get("choices") or []
                    choice = choices[0] if choices else {}
                    if isinstance(choice, dict):
                        d = choice.get("delta")
                        if isinstance(d, dict):
                            delta = d.get("content")
                        if delta is None:
                            m = choice.get("message")
                            if isinstance(m, dict):
                                delta = m.get("content")
                    # Ollama 兼容结构（常见）：{"message":{"content":...}}
                    if delta is None:
                        m = chunk.get("message")
                        if isinstance(m, dict):
                            delta = m.get("content")
                else:
                    choice = chunk.choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        delta = getattr(delta, "content", None)
                    if delta is None:
                        msg = getattr(choice, "message", None)
                        if msg is not None:
                            delta = getattr(msg, "content", None)
            except Exception:
                delta = None

            if isinstance(delta, str) and delta:
                yield delta
    
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

        kwargs.setdefault("num_retries", 2)
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

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成接口：逐步 yield 新增文本片段（delta）。
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async for delta in self.chat_stream(messages, temperature, max_tokens, **kwargs):
            yield delta
    
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
    
    @property
    def default_api_base(self) -> str:
        return "https://api.hunyuan.cloud.tencent.com/v1"

    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        # 混元通过OpenAI兼容接口调用
        if not self.api_base:
            self.api_base = self.default_api_base
        return f"openai/{model}"


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

        kwargs.setdefault("num_retries", 2)
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

        kwargs.setdefault("num_retries", 2)
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

        kwargs.setdefault("num_retries", 2)
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
    
    async def _assert_model_available(self) -> None:
        """
        调用前预检：从 /api/tags 获取可用模型，避免盲猜模型名。
        """
        base = (self.api_base or "http://localhost:11434").rstrip("/")
        url = f"{base}/api/tags"
        model = self._get_model()
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Ollama /api/tags 返回 {resp.status}: {text}")
                    payload = await resp.json()
        except Exception as e:
            # 网络/服务不可用时，沿用原有 LiteLLM 报错路径
            return

        names: List[str] = []
        for m in (payload.get("models") or []):
            n = m.get("name")
            if isinstance(n, str):
                names.append(n)

        if names and model not in names:
            preview = ", ".join(names[:20])
            more = "" if len(names) <= 20 else f" ... (+{len(names) - 20})"
            raise ValueError(
                f"Ollama 未找到模型 '{model}'。"
                f"当前 {url} 可用模型包括: {preview}{more}。"
                f"请把 DEFAULT_LLM_MODEL 改成其中一个，或确认 OLLAMA_MODELS/OLLAMA_API_BASE 指向同一服务。"
            )

    def _get_litellm_model_name(self) -> str:
        model = self._get_model()
        return f"ollama/{model}"

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        """
        使用 Ollama 原生 /api/chat。

        说明：
        - Ollama 的 OpenAI 兼容端点（/v1/chat/completions）对部分“推理模型”会把文本放到 reasoning 字段，
          导致 content 为空，从而前端看起来“无法流式/无输出”。
        - 这里直接调用 /api/chat，确保 content 字段有可见输出，并支持真正的 token 流式。
        """
        await self._assert_model_available()
        base = (self.api_base or "http://localhost:11434").rstrip("/")
        url = f"{base}/api/chat"
        payload = {
            "model": self._get_model(),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        timeout = aiohttp.ClientTimeout(total=kwargs.pop("timeout", None) or 120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama /api/chat 返回 {resp.status}: {text}")
                data = await resp.json()

        msg = (data.get("message") or {}) if isinstance(data, dict) else {}
        content = msg.get("content") if isinstance(msg, dict) else ""
        return LLMResponse(
            content=content or "",
            model=self._get_model(),
            provider=self.provider,
            usage=None,
            finish_reason=None,
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Ollama 原生流式：读取 NDJSON，每条 JSON 里 message.content 是增量片段。
        """
        await self._assert_model_available()
        base = (self.api_base or "http://localhost:11434").rstrip("/")
        url = f"{base}/api/chat"
        payload = {
            "model": self._get_model(),
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        timeout = aiohttp.ClientTimeout(total=kwargs.pop("timeout", None) or 120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama /api/chat 返回 {resp.status}: {text}")

                import json
                buffer = ""
                async for raw in resp.content.iter_chunked(1024):
                    if not raw:
                        continue
                    buffer += raw.decode("utf-8", errors="ignore")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        msg = obj.get("message")
                        if isinstance(msg, dict):
                            delta = msg.get("content")
                            if isinstance(delta, str) and delta:
                                yield delta
                        if obj.get("done") is True:
                            return
