#!/usr/bin/env python3
import os
import subprocess
import sys

# 设置数标标 API 环境变量
os.environ['SHUBIAOBIAO_API_KEY'] = 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av'
os.environ['SHUBIAOBIAO_API_BASE'] = 'https://hk.n1n.ai/v1'

# 获取命令行参数
if len(sys.argv) < 2:
    print("用法: python query_graphrag.py \"你的查询问题\" [local|global|drift]")
    sys.exit(1)

query = sys.argv[1]
method = sys.argv[2] if len(sys.argv) > 2 else 'local'

# 运行查询
result = subprocess.run(
    [
        sys.executable, '-m', 'graphrag', 'query',
        '-r', 'data/graphrag_output',
        '-m', method,
        query
    ],
    cwd=r'D:\projects\GraphRAG',
    env=os.environ
)

sys.exit(result.returncode)
