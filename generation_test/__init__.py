# 生成模块测试目录

这是一个独立的测试目录，用于测试生成模块的功能。测试完成后可以整体删除。

## 快速开始

```bash
# 直接运行测试
conda run -n RAG python generation_test/test_generation_e2e.py

# 或使用pytest
conda run -n RAG pytest generation_test/test_generation_e2e.py -v -s
```

## 删除测试

测试完成后，直接删除整个目录：

```bash
rm -rf generation_test/
```

## 文件说明

- `test_generation_e2e.py` - 端到端测试脚本
- `README.md` - 详细的测试说明
- `__init__.py` - Python包标识文件
