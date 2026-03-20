#!/usr/bin/env python3
import os
import subprocess
import sys

os.environ['GLM_API_KEY'] = '9ed068cde7f549efad0fb1affde722d1.SWLbfEK1eni4Tv9d'

os.chdir(r'D:\projects\Capstone\GraphRAG\data\graphrag_output')

result = subprocess.run(
    [sys.executable, '-m', 'graphrag', 'index', '--root', '.', '--verbose'],
    cwd=r'D:\projects\Capstone\GraphRAG\data\graphrag_output',
    env=os.environ
)

sys.exit(result.returncode)
