#!/usr/bin/env python3
import os
import subprocess
import sys

# 使用数标标 API
os.environ['SHUBIAOBIAO_API_KEY'] = 'sk-6IPAckVNGiQCP8Q9O2sAl4TV7KzGmNiGS8YuGsPHF0t406av'
os.environ['SHUBIAOBIAO_API_BASE'] = 'https://hk.n1n.ai/v1'

os.chdir(r'D:\projects\GraphRAG\data\graphrag_output')

result = subprocess.run(
    [sys.executable, '-m', 'graphrag', 'index', '--root', '.', '--verbose'],
    cwd=r'D:\projects\GraphRAG\data\graphrag_output',
    env=os.environ
)

sys.exit(result.returncode)
