"""
FastAPI 后端服务
提供 REST API 接口
"""

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.llm import create_llm_adapter, get_available_providers, LLMRouter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator, PromptTemplates
from src.indexing import GraphBuilder


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
    provider: str = Field(default="deepseek", description="LLM提供商")
    style: str = Field(default="standard", description="写作风格")
    temperature: float = Field(default=0.7, ge=0.1, le=1.0, description="创意度")


class GenerateResponse(BaseModel):
    """生成响应"""
    content: str
    provider: str
    model: str
    extracted_info: dict
    retrieval_stats: dict


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
        # 创建组件
        llm_adapter = create_llm_adapter(provider=request.provider)
        retriever = MemoirRetriever(llm_adapter=llm_adapter)
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        
        # 检索
        retrieval_result = await retriever.retrieve(
            request.memoir_text,
            top_k=10,
            use_llm_parsing=True,
        )
        
        # 生成
        gen_result = await generator.generate(
            memoir_text=request.memoir_text,
            retrieval_result=retrieval_result,
            temperature=request.temperature,
        )
        
        # 构建响应
        context = retrieval_result.context
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
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        retriever = MemoirRetriever()
        
        # 检索
        retrieval_result = retriever.retrieve_sync(request.memoir_text, top_k=10)
        
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
