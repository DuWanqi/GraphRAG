"""
项目配置管理
使用 pydantic-settings 从环境变量加载配置
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# 计算项目根目录（src/config/settings.py -> 项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class Settings(BaseSettings):
    """项目全局配置"""
    
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Deepseek配置
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_api_base: str = Field(
        default="https://api.deepseek.com/v1", 
        alias="DEEPSEEK_API_BASE"
    )
    
    # Qwen配置
    qwen_api_key: Optional[str] = Field(default=None, alias="QWEN_API_KEY")
    qwen_api_base: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="QWEN_API_BASE"
    )
    
    # 腾讯混元配置（OpenAI 兼容：https://api.hunyuan.cloud.tencent.com/v1 ，仅需控制台 API Key）
    hunyuan_api_key: Optional[str] = Field(default=None, alias="HUNYUAN_API_KEY")
    hunyuan_api_base: Optional[str] = Field(
        default=None,
        alias="HUNYUAN_API_BASE",
    )
    
    # Google Gemini配置
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    # Google Custom Search（用于 SAFE 独立事实验证的网络搜索，可选）
    google_cse_id: Optional[str] = Field(default=None, alias="GOOGLE_CSE_ID")
    
    # 智谱GLM配置
    glm_api_key: Optional[str] = Field(default=None, alias="GLM_API_KEY")
    
    # OpenAI配置 (可选)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    
    # 数标标配置（test分支）
    shubiaobiao_api_key: Optional[str] = Field(default=None, alias="SHUBIAOBIAO_API_KEY")
    shubiaobiao_api_base: str = Field(
        default="https://hk.n1n.ai/v1", 
        alias="SHUBIAOBIAO_API_BASE"
    )
    
    # 默认LLM配置
    default_llm_provider: str = Field(default="hunyuan", alias="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="hunyuan-lite", alias="DEFAULT_LLM_MODEL")

    # Ollama（本地）配置
    # LiteLLM 连接 Ollama 时使用的 base URL（例如 http://localhost:11434）
    ollama_api_base: str = Field(default="http://localhost:11434", alias="OLLAMA_API_BASE")
    
    # GraphRAG路径配置（使用项目根目录的绝对路径）
    graphrag_input_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "input"), 
        alias="GRAPHRAG_INPUT_DIR"
    )
    graphrag_output_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "graphrag_output"),
        alias="GRAPHRAG_OUTPUT_DIR"
    )

    # Plain vector RAG baseline configuration.
    plain_rag_input_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "graphrag_output" / "input"),
        alias="PLAIN_RAG_INPUT_DIR",
    )
    plain_rag_embedding_backend: str = Field(
        default="ollama",
        alias="PLAIN_RAG_EMBEDDING_BACKEND",
    )
    plain_rag_embedding_model: str = Field(
        default="nomic-embed-text",
        alias="PLAIN_RAG_EMBEDDING_MODEL",
    )
    plain_rag_cache_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "plain_vector_rag_cache"),
        alias="PLAIN_RAG_CACHE_DIR",
    )
    
    # 应用配置
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="APP_DEBUG")  # 使用APP_DEBUG避免与系统环境变量冲突


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
