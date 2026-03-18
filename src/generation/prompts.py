"""
提示词模板库
针对不同场景的提示词模板
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any


class PromptTemplates:
    """提示词模板管理"""

    @classmethod
    @lru_cache(maxsize=1)
    def _load_prompts(cls) -> Dict[str, Any]:
        here = Path(__file__).resolve().parent
        path = here / "prompts.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def get_template(cls, style: str = "standard") -> str:
        """获取指定风格的模板"""
        data = cls._load_prompts()
        styles = data.get("styles") or {}
        style_obj = styles.get(style) or styles.get("standard") or {}
        template = style_obj.get("template")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"prompts.json 中缺少风格模板: {style}")
        return template
    
    @classmethod
    def list_styles(cls) -> Dict[str, str]:
        """列出所有可用风格"""
        data = cls._load_prompts()
        styles = data.get("styles") or {}
        out: Dict[str, str] = {}
        for key, obj in styles.items():
            if isinstance(obj, dict) and isinstance(obj.get("label"), str):
                out[key] = obj["label"]
        return out


# 系统提示词集合
def get_system_prompt(key: str = "default") -> str:
    data = PromptTemplates._load_prompts()
    sp = (data.get("system_prompts") or {}).get(key)
    if isinstance(sp, str) and sp.strip():
        return sp
    # 兜底：使用 default
    sp2 = (data.get("system_prompts") or {}).get("default")
    if isinstance(sp2, str) and sp2.strip():
        return sp2
    raise ValueError("prompts.json 中缺少 system_prompts.default")
