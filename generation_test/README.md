# 生成模块端到端测试

## 测试目标

测试生成模块的完整功能，包括：
1. 回忆录分段（segmentation）
2. 预算分配（budget allocation）
3. RAG检索集成
4. 单段生成
5. 长文生成（多章节）
6. 跨章节上下文管理

## 前置条件

1. **已构建的知识图谱**: 需要 `output/` 目录下有完整的图谱数据
2. **配置好的LLM**: 通过环境变量或配置文件配置LLM访问

## 运行方式

### 方式1: 使用pytest（推荐）

```bash
# 运行所有测试
conda run -n RAG pytest tests/test_generation_e2e.py -v

# 运行特定测试
conda run -n RAG pytest tests/test_generation_e2e.py::TestGenerationModule::test_memoir_segmentation -v

# 显示打印输出
conda run -n RAG pytest tests/test_generation_e2e.py -v -s
```

### 方式2: 直接运行脚本

```bash
conda run -n RAG python tests/test_generation_e2e.py
```

## 测试说明

### 测试1: 回忆录分段
- 验证文本能正确分段
- 检查元数据提取（年份、地点）
- 验证分段报告

### 测试2: 预算分配
- 验证每个段落都有预算
- 检查预算参数合理性

### 测试3: 单段检索
- 使用真实RAG检索
- 验证检索结果包含实体、社区、关系

### 测试4: 单段生成
- 完整的检索+生成流程
- 验证生成内容质量

### 测试5: 长文生成
- 完整的多章节生成流程
- 验证章节合并
- 检查分段报告

### 测试6: 跨章节上下文
- 验证章节上下文记录
- 检查关键短语提取
- 验证上下文传递

## 测试数据

使用内置的简短测试回忆录（约200字），包含：
- 1978年：考上大学
- 1980年：遇到妻子
- 1985年：大学毕业分配工作

## 预期输出

测试会打印详细的中间结果，包括：
- 分段信息
- 预算分配
- 检索结果
- 生成内容
- 章节上下文

## 注意事项

1. 测试需要真实的LLM调用，会产生API费用
2. 测试时间取决于LLM响应速度（通常1-3分钟）
3. 如果没有知识图谱或LLM配置，测试会自动跳过
