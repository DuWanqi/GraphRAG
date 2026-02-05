#!/usr/bin/env python
"""
启动 Web 应用的入口脚本
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_gradio(host: str, port: int, share: bool = False):
    """启动 Gradio 应用"""
    from web.app import create_ui, settings
    
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
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
