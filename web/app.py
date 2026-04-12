"""
记忆图谱 Web 应用
基于 Gradio 构建的用户界面
"""

import asyncio
from typing import Optional, List, Tuple
import gradio as gr

from src.config import get_settings
from src.llm import create_llm_adapter, get_available_providers, LLMRouter
from src.indexing import GraphBuilder, DataLoader
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator, PromptTemplates
from src.evaluation import Evaluator, FActScoreChecker


# 全局变量
settings = get_settings()
retriever: Optional[MemoirRetriever] = None
generator: Optional[LiteraryGenerator] = None
current_provider: Optional[str] = None  # 跟踪当前使用的 provider


def init_components(provider: str = "gemini"):
    """初始化组件"""
    global retriever, generator, current_provider
    
    # 如果 provider 相同且组件已初始化，则跳过
    if provider == current_provider and retriever is not None and generator is not None:
        return f"✅ 已使用 {provider} 模型"
    
    try:
        llm_adapter = create_llm_adapter(provider=provider)
        retriever = MemoirRetriever(llm_adapter=llm_adapter)
        generator = LiteraryGenerator(llm_adapter=llm_adapter)
        current_provider = provider
        return f"✅ 初始化成功！使用 {provider} 模型"
    except Exception as e:
        return f"❌ 初始化失败: {str(e)}"


async def process_memoir_async(
    memoir_text: str,
    provider: str,
    style: str,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "keyword",
    use_rule_decompose: bool = False,
) -> Tuple[str, str, str, str]:
    """
    异步处理回忆录
    
    Returns:
        (生成的历史背景, 提取的信息, 检索结果, 事实性检查结果)
    """
    global retriever, generator, current_provider
    
    if not memoir_text.strip():
        return "请输入回忆录文本", "", "", ""
    
    if retriever is None or generator is None or current_provider != provider:
        init_result = init_components(provider)
        if "失败" in init_result:
            return init_result, "", "", ""
    
    try:
        import time
        start_time = time.time()
        
        print(f"[DEBUG] 开始处理回忆录，长度: {len(memoir_text)} 字符")
        
        # 检索相关内容（包含实体提取）
        retrieve_start = time.time()
        print("[DEBUG] 开始检索相关内容...")
        retrieval_result = await retriever.retrieve(
            memoir_text, 
            top_k=10,
            use_llm_parsing=False,
            mode=retrieval_mode,
        )
        retrieve_time = time.time() - retrieve_start
        print(f"[DEBUG] 检索完成，耗时: {retrieve_time:.2f} 秒")
        
        context = retrieval_result.context
        extracted_info = f"""**提取的时间**: {context.year or '未识别'}
**提取的地点**: {context.location or '未识别'}
**关键词**: {', '.join(context.keywords) if context.keywords else '无'}
**生成的查询**: {retrieval_result.query}
**提取+检索耗时**: {retrieve_time:.2f} 秒"""
        
        retrieval_info = f"""**找到实体**: {len(retrieval_result.entities)} 个
**找到关系**: {len(retrieval_result.relationships)} 个
**社区报告**: {len(retrieval_result.communities)} 个
**相关文本**: {len(retrieval_result.text_units)} 段"""
        
        if retrieval_result.entities:
            retrieval_info += "\n\n**主要实体**:\n"
            for entity in retrieval_result.entities[:5]:
                retrieval_info += f"- {entity.get('name', '未知')}: {entity.get('description', '')[:100]}...\n"
        
        # 生成文本
        gen_start = time.time()
        print("[DEBUG] 开始生成文本...")
        gen_result = await generator.generate(
            memoir_text=memoir_text,
            retrieval_result=retrieval_result,
            temperature=temperature,
            max_tokens=2048,  # 增加最大token数，避免输出被截断
        )
        gen_time = time.time() - gen_start
        print(f"[DEBUG] 生成完成，耗时: {gen_time:.2f} 秒")
        
        # 先生成基本的事实性检查信息
        fact_check_info = f"**生成耗时**: {gen_time:.2f} 秒\n"
        
        # 如果不需要事实性检查，直接返回
        if not enable_fact_check:
            total_time = time.time() - start_time
            fact_check_info += f"\n**总耗时**: {total_time:.2f} 秒"
            print(f"[DEBUG] 处理完成，总耗时: {total_time:.2f} 秒")
            return gen_result.content, extracted_info, retrieval_info, fact_check_info
        
        # 事实性检查（带超时处理）
        print("[DEBUG] 开始事实性检查...")
        
        async def run_fact_check():
            """运行事实性检查"""
            check_start = time.time()
            try:
                llm_adapter = create_llm_adapter(provider=provider)
                # 使用FActScoreChecker替代原来的FactChecker
                fact_checker = FActScoreChecker(llm_adapter=llm_adapter)
                
                # 带超时的事实性检查
                async def check_with_timeout():
                    return await fact_checker.check(
                        memoir_text=memoir_text,
                        generated_text=gen_result.content,
                        retrieval_result=retrieval_result,
                        use_llm=True,
                        use_rule_decompose=use_rule_decompose,
                    )
                
                # 设置120秒超时（延长超时时间）
                fact_result = await asyncio.wait_for(check_with_timeout(), timeout=120.0)
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查完成，耗时: {check_time:.2f} 秒")
                
                status_icon = "✅" if fact_result.is_factual else "⚠️"
                return f"""**生成耗时**: {gen_time:.2f} 秒

### {status_icon} 事实性检查结果

**一致性判定**: {'事实一致' if fact_result.is_factual else '存在潜在问题'}
**置信度**: {fact_result.confidence:.2%}
**实体覆盖率**: {fact_result.entity_coverage:.2%}
**证据支持度**: {fact_result.evidence_support:.2%}
**检查耗时**: {check_time:.2f} 秒

**总结**: {fact_result.summary}

"""
                
            except asyncio.TimeoutError:
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查超时，耗时: {check_time:.2f} 秒")
                return f"**生成耗时**: {gen_time:.2f} 秒\n**检查耗时**: {check_time:.2f} 秒\n事实性检查超时，请稍后重试"
            except Exception as e:
                check_time = time.time() - check_start
                print(f"[DEBUG] 事实性检查失败: {str(e)}")
                return f"**生成耗时**: {gen_time:.2f} 秒\n**检查耗时**: {check_time:.2f} 秒\n事实性检查失败: {str(e)}"
        
        # 优化：不再使用后台任务，而是直接同步执行事实性检查
        # 这样可以确保结果能够正确返回给Gradio
        print("[DEBUG] 开始事实性检查...")
        
        # 运行事实性检查（带超时）
        try:
            fact_check_info = await run_fact_check()
            print("[DEBUG] 事实性检查完成")
        except Exception as e:
            print(f"[DEBUG] 事实性检查失败: {e}")
            fact_check_info = fact_check_info + f"\n**事实性检查失败**: {str(e)}"
        
        total_time = time.time() - start_time
        fact_check_info += f"\n**总耗时**: {total_time:.2f} 秒"
        print(f"[DEBUG] 处理完成，总耗时: {total_time:.2f} 秒")
        
        return gen_result.content, extracted_info, retrieval_info, fact_check_info
        
    except Exception as e:
        print(f"[DEBUG] 处理失败: {str(e)}")
        return f"处理失败: {str(e)}", "", "", ""


def process_memoir(
    memoir_text: str,
    provider: str,
    style: str,
    temperature: float,
    enable_fact_check: bool = True,
    retrieval_mode: str = "keyword",
    use_rule_decompose: bool = False,
) -> Tuple[str, str, str, str]:
    """处理回忆录（同步包装）"""
    return asyncio.run(process_memoir_async(
        memoir_text, provider, style, temperature, enable_fact_check, retrieval_mode, use_rule_decompose
    ))


async def compare_providers_async(
    memoir_text: str,
    selected_providers: List[str],
    temperature: float,
) -> str:
    """
    使用多个LLM对比生成
    """
    if not memoir_text.strip():
        return "请输入回忆录文本"
    
    if not selected_providers:
        return "请至少选择一个LLM提供商"
    
    try:
        # 创建多LLM路由器
        router = LLMRouter()
        
        # 创建生成器
        gen = LiteraryGenerator(llm_router=router)
        
        # 创建检索器并检索
        ret = MemoirRetriever()
        retrieval_result = ret.retrieve_sync(memoir_text, top_k=10)
        
        # 并行生成
        multi_result = await gen.generate_parallel(
            memoir_text=memoir_text,
            retrieval_result=retrieval_result,
            providers=selected_providers,
            temperature=temperature,
        )
        
        # 格式化结果
        output = []
        for provider_name, result in multi_result.results.items():
            output.append(f"## {provider_name.upper()} ({result.model})\n")
            output.append(result.content)
            output.append("\n---\n")
        
        for provider_name, error in multi_result.errors.items():
            output.append(f"## {provider_name.upper()} - 错误\n")
            output.append(f"❌ {error}\n---\n")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"对比生成失败: {str(e)}"


def compare_providers(
    memoir_text: str,
    selected_providers: List[str],
    temperature: float,
) -> str:
    """对比生成（同步包装）"""
    return asyncio.run(compare_providers_async(memoir_text, selected_providers, temperature))


def build_index(llm_provider: str) -> str:
    """构建知识图谱索引"""
    try:
        builder = GraphBuilder(llm_provider=llm_provider)
        result = builder.build_index_sync()
        
        if result.success:
            stats = builder.get_index_stats()
            return f"""✅ 索引构建成功！

**统计信息**:
- 输入文件: {stats['input_files']} 个
- 实体数量: {stats['entities']} 个
- 关系数量: {stats['relationships']} 个
- 社区数量: {stats['communities']} 个

输出目录: {result.output_dir}"""
        else:
            return f"❌ 索引构建失败: {result.message}"
            
    except Exception as e:
        return f"❌ 索引构建异常: {str(e)}"


def get_index_status() -> str:
    """获取索引状态"""
    try:
        builder = GraphBuilder()
        stats = builder.get_index_stats()
        
        if stats["indexed"]:
            return f"""✅ 索引已就绪

**统计信息**:
- 输入文件: {stats['input_files']} 个
- 实体数量: {stats['entities']} 个
- 关系数量: {stats['relationships']} 个
- 社区数量: {stats['communities']} 个"""
        else:
            return f"""⚠️ 索引未构建

输入目录中有 {stats['input_files']} 个文件等待处理。
请点击"构建索引"按钮开始构建。"""
            
    except Exception as e:
        return f"❌ 获取状态失败: {str(e)}"


def create_ui():
    """创建Gradio界面"""
    
    # 获取可用的LLM提供商
    available = get_available_providers()
    available_providers = [k for k, v in available.items() if v]
    if not available_providers:
        available_providers = ["deepseek"]  # 默认
    
    with gr.Blocks(
        title="记忆图谱 - 历史背景注入系统",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="gray",
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .output-box { min-height: 200px; }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🎭 记忆图谱
            ### 基于RAG与知识图谱的个人回忆录历史背景自动注入系统
            
            输入您的回忆录片段，系统将自动检索相关历史背景，并生成具有文学性的描述文本。
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # 主功能标签页
            with gr.TabItem("📝 生成历史背景"):
                with gr.Row():
                    with gr.Column(scale=1):
                        memoir_input = gr.Textbox(
                            label="回忆录片段",
                            placeholder="输入您的回忆录文本，例如：\n1988年夏天，我从大学毕业，怀揣着梦想来到了深圳...",
                            lines=8,
                        )
                        
                        with gr.Row():
                            provider_select = gr.Dropdown(
                                choices=available_providers,
                                value=available_providers[0] if available_providers else None,
                                label="LLM 模型",
                            )
                            style_select = gr.Dropdown(
                                choices=list(PromptTemplates.list_styles().keys()),
                                value="standard",
                                label="写作风格",
                            )
                        
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="创意度 (Temperature)",
                        )
                        
                        retrieval_mode_select = gr.Radio(
                            choices=[
                                ("关键词检索 (快速)", "keyword"),
                                ("向量检索 (精准)", "vector"),
                                ("混合检索 (推荐)", "hybrid"),
                            ],
                            value="keyword",
                            label="检索模式",
                            info="选择知识图谱检索策略",
                        )
                        
                        fact_check_checkbox = gr.Checkbox(
                            value=True,
                            label="🔍 启用事实性检查",
                            info="检测生成内容是否存在幻觉或事实不一致",
                        )
                        
                        rule_decompose_checkbox = gr.Checkbox(
                            value=False,
                            label="⚡ 使用规则拆分",
                            info="使用规则而非LLM进行原子事实拆分，大幅提高速度",
                        )
                        
                        generate_btn = gr.Button("🚀 生成融合后的回忆录", variant="primary")
                    
                    with gr.Column(scale=1):
                        output_text = gr.Textbox(
                            label="融合后的回忆录文本",
                            lines=15,
                            elem_classes=["output-box"],
                        )
                        
                        with gr.Accordion("📊 详细信息", open=False):
                            extracted_info = gr.Markdown(label="提取的信息")
                            retrieval_info = gr.Markdown(label="检索结果")
                        
                        with gr.Accordion("🔍 事实性检查", open=True):
                            fact_check_output = gr.Markdown(label="事实性检查结果")
                
                generate_btn.click(
                    fn=process_memoir,
                    inputs=[memoir_input, provider_select, style_select, temperature_slider, fact_check_checkbox, retrieval_mode_select, rule_decompose_checkbox],
                    outputs=[output_text, extracted_info, retrieval_info, fact_check_output],
                )
            
            # 多模型对比标签页
            with gr.TabItem("🔄 多模型对比"):
                gr.Markdown("使用多个LLM同时生成，对比不同模型的输出效果。")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_input = gr.Textbox(
                            label="回忆录片段",
                            placeholder="输入回忆录文本...",
                            lines=6,
                        )
                        
                        providers_checkbox = gr.CheckboxGroup(
                            choices=available_providers,
                            value=available_providers[:2] if len(available_providers) >= 2 else available_providers,
                            label="选择要对比的模型",
                        )
                        
                        compare_temp = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            label="创意度",
                        )
                        
                        compare_btn = gr.Button("🔄 开始对比", variant="primary")
                    
                    with gr.Column(scale=1):
                        compare_output = gr.Markdown(label="对比结果")
                
                compare_btn.click(
                    fn=compare_providers,
                    inputs=[compare_input, providers_checkbox, compare_temp],
                    outputs=[compare_output],
                )
            
            # 索引管理标签页
            with gr.TabItem("⚙️ 索引管理"):
                gr.Markdown("""
                ## 📊 知识图谱索引管理
                
                索引构建会使用 LLM 从历史事件数据中提取实体和关系，构建知识图谱。
                
                **支持的 LLM 模型：**
                | 提供商 | 环境变量 | 默认模型 |
                |-------|----------|---------|
                | Gemini | `GOOGLE_API_KEY` | gemini-2.5-flash |
                | DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
                | Qwen | `QWEN_API_KEY` | qwen-plus |
                | 智谱GLM | `GLM_API_KEY` | glm-4.7-flash |
                | OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
                | 混元 | `HUNYUAN_API_KEY` | hunyuan-lite |
                
                **注意**：首次构建索引需要几分钟时间，会产生一定的 API 费用。
                """)
                
                with gr.Row():
                    with gr.Column():
                        index_status = gr.Markdown(value=get_index_status())
                        
                        refresh_btn = gr.Button("🔄 刷新状态")
                        refresh_btn.click(fn=get_index_status, outputs=[index_status])
                    
                    with gr.Column():
                        # 完整的模型选择列表
                        all_providers = ["gemini", "deepseek", "qwen", "openai", "hunyuan", "glm"]
                        default_provider = available_providers[0] if available_providers else "gemini"
                        
                        index_provider = gr.Dropdown(
                            choices=all_providers,
                            value=default_provider,
                            label="构建索引使用的 LLM",
                            info="选择用于构建知识图谱的语言模型",
                        )
                        
                        build_btn = gr.Button("🔨 构建索引", variant="primary")
                        build_output = gr.Markdown()
                        
                        build_btn.click(
                            fn=build_index,
                            inputs=[index_provider],
                            outputs=[build_output],
                        )
            
            # 使用说明标签页
            with gr.TabItem("📖 使用说明"):
                gr.Markdown("""
                ## 使用指南
                
                ### 1. 准备工作
                
                1. 在 `.env` 文件中配置您的 LLM API 密钥
                2. 将历史事件数据放入 `data/input/` 目录
                3. 构建知识图谱索引
                
                ### 2. 生成历史背景
                
                1. 在"生成历史背景"标签页输入回忆录片段
                2. 选择 LLM 模型和写作风格
                3. 调整创意度参数
                4. 点击"生成历史背景"按钮
                
                ### 3. 多模型对比
                
                1. 在"多模型对比"标签页输入回忆录片段
                2. 选择多个 LLM 模型
                3. 点击"开始对比"查看不同模型的输出
                
                ### 4. 写作风格说明
                
                - **standard**: 标准风格，平衡的文学性描述
                - **nostalgic**: 怀旧风格，温暖回忆的笔调
                - **narrative**: 叙事融合，与个人故事深度交织
                - **informative**: 简洁信息，重点突出的背景介绍
                - **conversational**: 对话风格，像讲故事一样亲切
                
                ### 5. 支持的 LLM 模型
                
                - Deepseek
                - Qwen (通义千问)
                - Hunyuan (腾讯混元)
                - Google Gemini
                - 智谱GLM
                - OpenAI GPT
                """)
        
        gr.Markdown(
            """
            ---
            **记忆图谱** - 让历史为您的回忆增添厚度 | Capstone Project 2026
            """
        )
    
    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name=settings.app_host,
        server_port=settings.app_port,
        share=False,
    )
