#!/usr/bin/env python3
import os
import sys

# 设置环境变量
os.environ['GRAPHRAG_OUTPUT_DIR'] = './data/graphrag_output/output'
os.environ['GLM_API_KEY'] = '7b5a8b76396b4f0fa8126e7b046ad583.VcowEIPLMsJnOKMp'

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.config import get_settings
from pathlib import Path

settings = get_settings()
print(f"配置的 graphrag_output_dir: {settings.graphrag_output_dir}")

index_dir = Path(settings.graphrag_output_dir)
print(f"index_dir: {index_dir}")
print(f"index_dir 绝对路径: {index_dir.absolute()}")

output_dir = index_dir / "output"
print(f"output_dir (错误的): {output_dir}")
print(f"output_dir 绝对路径: {output_dir.absolute()}")

# 正确的路径应该是
print(f"\n正确的路径应该是: {index_dir}")
print(f"entities.parquet 在正确路径下存在: {(index_dir / 'entities.parquet').exists()}")
print(f"entities.parquet 在错误路径下存在: {(output_dir / 'entities.parquet').exists()}")
