"""
FastAPI 后端服务
提供 REST API 接口
"""

from typing import Optional, List
import os
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.config.workspace_logging import setup_workspace_logging
from src.llm import create_llm_adapter, get_available_providers, LLMRouter
from src.retrieval import MemoirRetriever
from src.generation import (
    LiteraryGenerator,
    PromptTemplates,
    run_long_form_generation,
    single_segment_generation_config,
    single_segment_generation_config_from_range,
    estimate_long_form_generation_timeout,
    estimate_long_form_evaluation_timeout,
    build_long_form_eval_options,
)
from src.evaluation import evaluate_long_form, long_form_eval_to_json
from src.indexing import GraphBuilder


setup_workspace_logging()

# 全局检索器实例（避免重复加载索引）
_retriever: Optional[MemoirRetriever] = None


def _get_retriever() -> MemoirRetriever:
    """获取全局检索器实例"""
    global _retriever
    if _retriever is None:
        _retriever = MemoirRetriever()
    return _retriever


# 创建 FastAPI 应用
app = FastAPI(
    title="记忆图谱 API",
    description="基于RAG与知识图谱的个人回忆录历史背景自动注入系统",
    version="0.1.0",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求/响应模型
class GenerateRequest(BaseModel):
    """生成请求"""
    memoir_text: str = Field(..., description="回忆录文本")
    provider: str = Field(default="hunyuan", description="LLM提供商")
    style: str = Field(default="standard", description="写作风格")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="创意度")
    length_bucket: str = Field(
        default="400-800",
        description="兼容旧客户端：在未提供 generation_min/max 时用于单段模式档位映射",
    )
    generation_min_chars: Optional[int] = Field(
        default=None,
        ge=80,
        description="单段模式：目标生成字数下限；与 generation_max_chars 同时给出时优先生效",
    )
    generation_max_chars: Optional[int] = Field(
        default=None,
        ge=100,
        description="单段模式：目标生成字数上限",
    )
    chapter_generation_min_chars: Optional[int] = Field(
        default=None,
        ge=80,
        description="分章模式：每章目标下限；若缺省则尝试用 generation_*，否则默认 400",
    )
    chapter_generation_max_chars: Optional[int] = Field(
        default=None,
        ge=100,
        description="分章模式：每章目标上限；若缺省则尝试用 generation_*，否则默认 800",
    )
    retrieval_mode: str = Field(default="vector", description="keyword / vector / hybrid")
    chapter_mode: bool = Field(default=False, description="分章/长文：按段检索与生成后合并")


class GenerateResponse(BaseModel):
    """生成响应"""
    content: str
    provider: str
    model: str
    extracted_info: dict
    retrieval_stats: dict
    chapter_mode: bool = False
    eval_summary: Optional[str] = None


class CompareRequest(BaseModel):
    """对比请求"""
    memoir_text: str
    providers: List[str] = Field(default=["deepseek", "qwen"])
    temperature: float = Field(default=0.7)


class CompareResponse(BaseModel):
    """对比响应"""
    results: dict
    errors: dict


class IndexStatusResponse(BaseModel):
    """索引状态响应"""
    indexed: bool
    input_files: int
    entities: int
    relationships: int
    communities: int


# API 路由
@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "记忆图谱 API",
        "version": "0.1.0",
        "description": "基于RAG与知识图谱的个人回忆录历史背景自动注入系统",
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/providers")
async def list_providers():
    """列出可用的LLM提供商"""
    available = get_available_providers()
    return {
        "available": [k for k, v in available.items() if v],
        "all": list(available.keys()),
    }


@app.get("/styles")
async def list_styles():
    """列出可用的写作风格"""
    return PromptTemplates.list_styles()


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    生成历史背景描述
    
    输入回忆录文本，返回生成的历史背景描述
    """
    try:
        timing = os.getenv("TEMP_TIMING") == "1"
        t0 = time.perf_counter()
        # 创建组件
        llm_adapter = create_llm_adapter(provider=request.provider)
        # 使用全局检索器，避免重复加载索引
        global _retriever
        if _retriever is None:
            _retriever = MemoirRetriever(llm_adapter=llm_adapter)
        retriever = _retriever
        generator = LiteraryGenerator(llm_adapter=llm_adapter)

        if request.chapter_mode:
            from src.generation.memoir_segmenter import segment_memoir

            seg_lo, seg_hi = 300, 800
            cg_lo: int
            cg_hi: int
            if (
                request.chapter_generation_min_chars is not None
                and request.chapter_generation_max_chars is not None
            ):
                cg_lo = max(50, int(request.chapter_generation_min_chars))
                cg_hi = max(cg_lo + 1, int(request.chapter_generation_max_chars))
            elif (
                request.generation_min_chars is not None
                and request.generation_max_chars is not None
            ):
                cg_lo = max(80, int(request.generation_min_chars))
                cg_hi = max(cg_lo + 50, int(request.generation_max_chars))
            else:
                cg_lo, cg_hi = 400, 800

            n_ch = max(
                1,
                len(
                    segment_memoir(
                        request.memoir_text,
                        target_min_chars=seg_lo,
                        target_max_chars=seg_hi,
                    )
                ),
            )
            budget_timeout = estimate_long_form_generation_timeout(n_ch)
            t_gen0 = time.perf_counter()
            lf = await asyncio.wait_for(
                run_long_form_generation(
                    request.memoir_text,
                    retriever,
                    generator,
                    style=request.style,
                    temperature=request.temperature,
                    retrieval_mode=request.retrieval_mode,
                    use_llm_parsing=False,
                    target_min_chars=seg_lo,
                    target_max_chars=seg_hi,
                    chapter_gen_min_chars=cg_lo,
                    chapter_gen_max_chars=cg_hi,
                ),
                timeout=budget_timeout,
            )
            if timing:
                print(f"[TEMP_TIMING] api.generate.long_form={time.perf_counter()-t_gen0:.3f}s")
            last = lf.chapters[-1].generation if lf.chapters else None
            ev_text: Optional[str] = None
            if lf.chapters:
                eval_kwargs = build_long_form_eval_options(
                    llm_adapter=llm_adapter,
                    use_llm_eval=False,
                    enable_fact_check=True,
                    max_atomic_facts_per_segment=12,
                    fact_check_timeout_per_segment=45.0,
                    use_rule_decompose=True,
                )
                ev = await asyncio.wait_for(
                    evaluate_long_form(lf, **eval_kwargs),
                    timeout=estimate_long_form_evaluation_timeout(len(lf.chapters)),
                )
                ev_text = ev.summary_text + "\n\n" + long_form_eval_to_json(ev)[:6000]
            if timing:
                print(f"[TEMP_TIMING] api.generate.total={time.perf_counter()-t0:.3f}s")
            return GenerateResponse(
                content=lf.merged_content,
                provider=last.provider if last else request.provider,
                model=last.model if last else "",
                extracted_info={
                    "chapter_mode": True,
                    "chapters": len(lf.chapters),
                    "queries": [ch.retrieval_result.query for ch in lf.chapters],
                },
                retrieval_stats={
                    "entities_per_chapter": [len(ch.retrieval_result.entities) for ch in lf.chapters],
                    "relationships_total": sum(len(ch.retrieval_result.relationships) for ch in lf.chapters),
                },
                chapter_mode=True,
                eval_summary=ev_text,
            )

        # 单段：检索 + 生成
        t_ret0 = time.perf_counter()
        retrieval_result = await asyncio.wait_for(
            retriever.retrieve(
                request.memoir_text,
                top_k=10,
                use_llm_parsing=True,
                mode=request.retrieval_mode,
            ),
            timeout=45.0,
        )
        if timing:
            print(f"[TEMP_TIMING] api.generate.retrieve={time.perf_counter()-t_ret0:.3f}s")

        if (
            request.generation_min_chars is not None
            and request.generation_max_chars is not None
        ):
            g_lo = max(80, int(request.generation_min_chars))
            g_hi = max(g_lo + 50, int(request.generation_max_chars))
            single_cfg = single_segment_generation_config_from_range(g_lo, g_hi)
        else:
            single_cfg = single_segment_generation_config(request.length_bucket)
        t_gen0 = time.perf_counter()
        gen_result = await asyncio.wait_for(
            generator.generate(
                memoir_text=request.memoir_text,
                retrieval_result=retrieval_result,
                temperature=request.temperature,
                style=request.style,
                length_hint=single_cfg["length_hint"],
                max_tokens=single_cfg["max_tokens"],
            ),
            timeout=90.0,
        )
        if timing:
            print(f"[TEMP_TIMING] api.generate.generate={time.perf_counter()-t_gen0:.3f}s")

        context = retrieval_result.context
        if timing:
            print(f"[TEMP_TIMING] api.generate.total={time.perf_counter()-t0:.3f}s")
        return GenerateResponse(
            content=gen_result.content,
            provider=gen_result.provider,
            model=gen_result.model,
            extracted_info={
                "year": context.year,
                "location": context.location,
                "keywords": context.keywords,
                "query": retrieval_result.query,
            },
            retrieval_stats={
                "entities": len(retrieval_result.entities),
                "relationships": len(retrieval_result.relationships),
                "communities": len(retrieval_result.communities),
                "text_units": len(retrieval_result.text_units),
            },
            chapter_mode=False,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except asyncio.TimeoutError as e:
        raise HTTPException(status_code=504, detail="请求超时（检索45s / 生成90s）") from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """
    多模型对比生成
    
    使用多个LLM同时生成，对比输出效果
    """
    try:
        # 创建路由器
        router = LLMRouter()
        generator = LiteraryGenerator(llm_router=router)
        
        # 使用全局检索器，避免重复加载索引
        retrieval_result = _get_retriever().retrieve_sync(request.memoir_text, top_k=10)
        
        # 并行生成
        multi_result = await generator.generate_parallel(
            memoir_text=request.memoir_text,
            retrieval_result=retrieval_result,
            providers=request.providers,
            temperature=request.temperature,
        )
        
        # 构建响应
        results = {}
        for provider_name, result in multi_result.results.items():
            results[provider_name] = {
                "content": result.content,
                "model": result.model,
            }
        
        return CompareResponse(
            results=results,
            errors=multi_result.errors,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比生成失败: {str(e)}")


@app.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """获取索引状态"""
    try:
        builder = GraphBuilder()
        stats = builder.get_index_stats()
        
        return IndexStatusResponse(
            indexed=stats["indexed"],
            input_files=stats["input_files"],
            entities=stats["entities"],
            relationships=stats["relationships"],
            communities=stats["communities"],
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@app.post("/index/build")
async def build_index(provider: str = "deepseek"):
    """
    构建知识图谱索引
    
    使用GraphRAG构建历史事件知识图谱
    """
    try:
        builder = GraphBuilder(llm_provider=provider)
        result = builder.build_index_sync()
        
        if result.success:
            stats = builder.get_index_stats()
            return {
                "success": True,
                "message": result.message,
                "stats": stats,
            }
        else:
            return {
                "success": False,
                "message": result.message,
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引构建失败: {str(e)}")


# 启动服务器
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
