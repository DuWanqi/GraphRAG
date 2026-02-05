"""
LLM适配器工厂
提供统一的适配器创建和管理接口
"""

from typing import Optional, Dict, Type
from .adapter import (
    LLMAdapter,
    LLMProvider,
    DeepseekAdapter,
    QwenAdapter,
    HunyuanAdapter,
    GeminiAdapter,
    OpenAIAdapter
)
from ..config import get_settings


# 提供商到适配器类的映射
ADAPTER_REGISTRY: Dict[LLMProvider, Type[LLMAdapter]] = {
    LLMProvider.DEEPSEEK: DeepseekAdapter,
    LLMProvider.QWEN: QwenAdapter,
    LLMProvider.HUNYUAN: HunyuanAdapter,
    LLMProvider.GEMINI: GeminiAdapter,
    LLMProvider.OPENAI: OpenAIAdapter,
}

# 提供商到默认模型的映射
DEFAULT_MODELS: Dict[LLMProvider, str] = {
    LLMProvider.DEEPSEEK: "deepseek-chat",
    LLMProvider.QWEN: "qwen-plus",
    LLMProvider.HUNYUAN: "hunyuan-lite",
    LLMProvider.GEMINI: "gemini-1.5-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
}


def create_llm_adapter(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> LLMAdapter:
    """
    创建LLM适配器
    
    Args:
        provider: 提供商名称（deepseek/qwen/hunyuan/gemini/openai）
        model: 模型名称
        api_key: API密钥（如果不提供，从环境变量读取）
        api_base: API基础URL
        **kwargs: 其他参数
        
    Returns:
        LLMAdapter: 对应的适配器实例
        
    Raises:
        ValueError: 如果提供商不支持或缺少必要配置
    """
    settings = get_settings()
    
    # 确定提供商
    if provider is None:
        provider = settings.default_llm_provider
    
    try:
        provider_enum = LLMProvider(provider.lower())
    except ValueError:
        raise ValueError(
            f"不支持的LLM提供商: {provider}. "
            f"支持的提供商: {[p.value for p in LLMProvider]}"
        )
    
    # 获取适配器类
    adapter_class = ADAPTER_REGISTRY.get(provider_enum)
    if adapter_class is None:
        raise ValueError(f"未找到提供商 {provider} 的适配器")
    
    # 从配置获取API密钥和基础URL
    if api_key is None:
        api_key = _get_api_key_from_settings(provider_enum, settings)
    
    if api_base is None:
        api_base = _get_api_base_from_settings(provider_enum, settings)
    
    if api_key is None:
        raise ValueError(
            f"缺少 {provider} 的API密钥。"
            f"请在 .env 文件中设置对应的 API_KEY 环境变量。"
        )
    
    # 特殊处理混元适配器
    if provider_enum == LLMProvider.HUNYUAN:
        return adapter_class(
            api_key=api_key,
            secret_id=settings.hunyuan_secret_id,
            secret_key=settings.hunyuan_secret_key,
            api_base=api_base,
            model=model,
            **kwargs
        )
    
    return adapter_class(
        api_key=api_key,
        api_base=api_base,
        model=model,
        **kwargs
    )


def _get_api_key_from_settings(provider: LLMProvider, settings) -> Optional[str]:
    """从配置获取API密钥"""
    key_mapping = {
        LLMProvider.DEEPSEEK: settings.deepseek_api_key,
        LLMProvider.QWEN: settings.qwen_api_key,
        LLMProvider.HUNYUAN: settings.hunyuan_api_key,
        LLMProvider.GEMINI: settings.google_api_key,
        LLMProvider.OPENAI: settings.openai_api_key,
    }
    return key_mapping.get(provider)


def _get_api_base_from_settings(provider: LLMProvider, settings) -> Optional[str]:
    """从配置获取API基础URL"""
    base_mapping = {
        LLMProvider.DEEPSEEK: settings.deepseek_api_base,
        LLMProvider.QWEN: settings.qwen_api_base,
        LLMProvider.HUNYUAN: None,  # 混元使用默认
        LLMProvider.GEMINI: None,   # Gemini使用默认
        LLMProvider.OPENAI: None,   # OpenAI使用默认
    }
    return base_mapping.get(provider)


def get_available_providers() -> Dict[str, bool]:
    """
    获取可用的LLM提供商列表
    
    Returns:
        Dict[str, bool]: 提供商名称到是否可用的映射
    """
    settings = get_settings()
    
    return {
        "deepseek": settings.deepseek_api_key is not None,
        "qwen": settings.qwen_api_key is not None,
        "hunyuan": settings.hunyuan_api_key is not None,
        "gemini": settings.google_api_key is not None,
        "openai": settings.openai_api_key is not None,
    }


def get_all_adapters() -> Dict[str, LLMAdapter]:
    """
    获取所有已配置的LLM适配器
    
    Returns:
        Dict[str, LLMAdapter]: 提供商名称到适配器的映射
    """
    adapters = {}
    available = get_available_providers()
    
    for provider_name, is_available in available.items():
        if is_available:
            try:
                adapters[provider_name] = create_llm_adapter(provider=provider_name)
            except ValueError:
                pass  # 跳过配置不完整的提供商
    
    return adapters
