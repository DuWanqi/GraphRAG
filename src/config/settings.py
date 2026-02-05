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
    
    # 腾讯混元配置
    hunyuan_api_key: Optional[str] = Field(default=None, alias="HUNYUAN_API_KEY")
    hunyuan_secret_id: Optional[str] = Field(default=None, alias="HUNYUAN_SECRET_ID")
    hunyuan_secret_key: Optional[str] = Field(default=None, alias="HUNYUAN_SECRET_KEY")
    
    # Google Gemini配置
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    
    # OpenAI配置 (可选)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    
    # 默认LLM配置
    default_llm_provider: str = Field(default="deepseek", alias="DEFAULT_LLM_PROVIDER")
    default_llm_model: str = Field(default="deepseek-chat", alias="DEFAULT_LLM_MODEL")
    
    # GraphRAG路径配置（使用项目根目录的绝对路径）
    graphrag_input_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "input"), 
        alias="GRAPHRAG_INPUT_DIR"
    )
    graphrag_output_dir: str = Field(
        default=str(PROJECT_ROOT / "data" / "output"), 
        alias="GRAPHRAG_OUTPUT_DIR"
    )
    
    # 应用配置
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="APP_DEBUG")  # 使用APP_DEBUG避免与系统环境变量冲突


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
