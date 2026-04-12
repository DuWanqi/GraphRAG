#!/usr/bin/env python3
"""
自定义 GraphRAG 索引构建脚本 - 跳过社区报告阶段
"""

import os
import sys
import asyncio
import pandas as pd

# 设置环境变量
os.environ['SHUBIAOBIAO_API_KEY'] = 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av'
os.environ['SHUBIAOBIAO_API_BASE'] = 'https://hk.n1n.ai/v1'

# 切换到工作目录
os.chdir(r'D:\projects\GraphRAG\data\graphrag_output')

# 添加 graphrag 到路径
sys.path.insert(0, r'D:\sofe\miniconda_itself\Lib\site-packages')

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.workflows.factory import WorkflowFactory
from graphrag.index.typing.context import PipelineRunContext
from graphrag.cache.file_cache import FileCache
from graphrag.storage.file_table_provider import FileTableProvider
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks

async def run_index():
    """运行索引构建（跳过社区报告）"""
    print("Loading configuration...")
    config = GraphRagConfig.from_yaml("settings.yaml")
    
    # 创建工作流
    factory = WorkflowFactory()
    workflows = factory.create_workflows(config)
    
    # 创建上下文
    cache = FileCache("cache")
    table_provider = FileTableProvider("output")
    callbacks = WorkflowCallbacks()
    
    context = PipelineRunContext(
        config=config,
        cache=cache,
        output_table_provider=table_provider,
        callbacks=callbacks,
    )
    
    # 运行工作流（跳过 create_community_reports）
    skip_workflows = ["create_community_reports", "create_community_reports_text"]
    
    for workflow_name, workflow_func in workflows.items():
        if workflow_name in skip_workflows:
            print(f"Skipping workflow: {workflow_name}")
            continue
        
        print(f"Running workflow: {workflow_name}")
        try:
            await workflow_func(config, context)
            print(f"Completed workflow: {workflow_name}")
        except Exception as e:
            print(f"Error in workflow {workflow_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nIndex building completed!")

if __name__ == "__main__":
    asyncio.run(run_index())
