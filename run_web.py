#!/usr/bin/env python
"""
启动 Web 应用的入口脚本
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Gradio 不需要走代理，避免 proxy 导致启动失败
# 保存原始代理设置供 LLM API 调用使用
_saved_http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
_saved_https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(var, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def run_gradio(host: str, port: int, share: bool = False):
    """启动 Gradio 应用"""
    import gradio as gr
    from web.app import create_ui, settings

    app = create_ui()
    # 启用队列以确保 generator 流式更新可以实时推送到前端
    app.queue()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .output-box { min-height: 200px; }
        """,
    )


def run_api(host: str, port: int, reload: bool = False):
    """启动 FastAPI 服务"""
    import uvicorn
    
    uvicorn.run(
        "web.api:app",
        host=host,
        port=port,
        reload=reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="记忆图谱 - Web 应用启动器"
    )
    
    parser.add_argument(
        "mode",
        choices=["gradio", "api", "both"],
        default="gradio",
        nargs="?",
        help="运行模式: gradio(默认), api, 或 both",
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="服务器地址 (默认: 0.0.0.0)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口 (默认: 8000)",
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建 Gradio 公共链接",
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="启用热重载 (仅API模式)",
    )
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                         记忆图谱                              ║
║     基于RAG与知识图谱的个人回忆录历史背景自动注入系统          ║
╚══════════════════════════════════════════════════════════════╝

启动模式: {args.mode}
服务地址: http://{args.host}:{args.port}
""")
    
    if args.mode == "gradio":
        run_gradio(args.host, args.port, args.share)
    elif args.mode == "api":
        run_api(args.host, args.port, args.reload)
    else:  # both
        import threading
        
        # 在后台线程运行API
        api_thread = threading.Thread(
            target=run_api,
            args=(args.host, args.port + 1, False),
            daemon=True,
        )
        api_thread.start()
        print(f"API 服务已启动在: http://{args.host}:{args.port + 1}")
        
        # 主线程运行Gradio
        run_gradio(args.host, args.port, args.share)


if __name__ == "__main__":
    main()
