# GraphRAG系统交接使用清单

## 1. 环境配置

### 硬件要求：
- CPU：至少4核，建议8核以上
- 内存：至少16GB，建议32GB以上
- 存储：至少50GB可用空间
- 网络：稳定的互联网连接（用于LLM API调用）

### 软件要求：
- Python 3.8+
- Git
- CUDA（可选，用于GPU加速）
- Ollama（可选，用于本地LLM）

### 依赖安装：
```bash
# 克隆仓库
git clone <仓库地址>
cd GraphRAG

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 2. 模型配置

### Nomic模型：
- 系统会自动下载Nomic模型（约1.5GB）
- 首次运行时会自动缓存到本地
- 无需手动下载

### LLM配置：

#### 1. GLM模型（默认）
- 需要在 `.env` 文件中配置 `GLM_API_KEY`
- 示例 `.env` 文件：
  ```
  GLM_API_KEY=your_api_key_here
  ```

#### 2. Ollama本地模型
- **安装Ollama**：
  - 访问 https://ollama.com/download 下载并安装
  - 启动Ollama服务
- **下载模型**：
  ```bash
  ollama pull glm3
  # 或其他模型，如：
  # ollama pull llama3
  # ollama pull gemma2
  ```
- **配置使用Ollama**：
  - 在 `config.py` 中修改LLM配置：
  ```python
  LLM_CONFIG = {
      "provider": "ollama",
      "model": "glm3",  # 或其他下载的模型
      "base_url": "http://localhost:11434",
  }
  ```

## 3. 数据处理流程

### 步骤1：准备历史背景数据
**历史背景数据**（1万-10万条）：
- 格式：支持纯文本、JSON等格式
- 内容：历史事件、人物、地点等背景信息
- 存储位置：建议放在 `./data/background` 目录

### 步骤2：构建历史背景索引
```bash
# 处理历史背景数据
python run_batch.py --input_dir ./data/background --output_dir ./data/background_processed

# 构建历史背景索引
python run_index.py --data_dir ./data/background_processed --index_dir ./index/background
```

### 步骤3：准备回忆录文本
**回忆录文本**（大批量测试数据）：
- 格式：支持纯文本格式
- 内容：个人回忆录、经历等
- 存储位置：建议放在 `./data/memoirs` 目录

### 步骤4：批量测试回忆录
```bash
# 批量处理回忆录文本（可选，用于预处理）
python run_batch.py --input_dir ./data/memoirs --output_dir ./data/memoirs_processed

# 批量测试回忆录（生成历史背景注入）
python run_test.py --input_dir ./data/memoirs --output_dir ./data/test_results --index_dir ./index/background
```

## 4. 命令行工具使用

### 主要命令：
```bash
# 启动Web界面
python run_web.py gradio

# 批量处理数据（历史背景或回忆录）
python run_batch.py --help

# 构建索引（主要用于历史背景数据）
python run_index.py --help

# 批量测试回忆录（使用历史背景索引）
python run_test.py --help
```

### 命令行参数：
- `--input_dir`：输入数据目录（历史背景或回忆录）
- `--output_dir`：输出处理结果目录
- `--data_dir`：构建索引的数据目录（通常是处理后的历史背景数据）
- `--index_dir`：索引存储目录（历史背景索引位置）
- `--batch_size`：批处理大小
- `--num_workers`：并行处理线程数
- `--llm_provider`：LLM提供商（可选：glm, ollama）
- `--llm_model`：LLM模型名称

### 使用Ollama的命令行示例：
```bash
# 使用Ollama处理历史背景数据
python run_batch.py --input_dir ./data/background --output_dir ./data/background_processed --llm_provider ollama --llm_model glm3

# 使用Ollama构建历史背景索引
python run_index.py --data_dir ./data/background_processed --index_dir ./index/background --llm_provider ollama --llm_model glm3

# 使用Ollama批量测试回忆录
python run_test.py --input_dir ./data/memoirs --output_dir ./data/test_results --index_dir ./index/background --llm_provider ollama --llm_model glm3

# 使用Ollama启动Web界面
python run_web.py gradio --llm_provider ollama --llm_model glm3
```

## 5. 系统功能

### 核心功能：
- 历史背景数据处理和索引构建
- 回忆录文本分析和处理
- 实体识别和关系抽取
- 知识图谱构建
- 语义检索（从历史背景索引中）
- 历史背景自动注入
- 事实性检查

### 使用流程：
**流程A：构建历史背景索引**
1. 准备历史背景数据（1万-10万条）
2. 运行批处理命令处理历史背景数据
3. 构建历史背景索引

**流程B：测试回忆录文本**
1. 准备回忆录文本数据（大批量）
2. 启动Web界面或使用命令行工具
3. 上传回忆录或指定回忆录目录
4. 系统自动从历史背景索引中检索相关信息
5. 生成注入历史背景的增强文本
6. 进行事实性检查

**流程C：批量测试**
1. 完成流程A（构建历史背景索引）
2. 准备大批量回忆录文本
3. 使用 `run_test.py` 进行批量处理
4. 查看测试结果

## 6. 常见问题处理

### 问题1：LLM API调用失败
- 检查 `.env` 文件中的API密钥
- 确保网络连接正常
- 检查API调用频率限制

### 问题2：Ollama模型加载失败
- 确保Ollama服务正在运行
- 检查模型是否已正确下载
- 验证模型名称是否正确

### 问题3：索引构建缓慢
- 调整 `--batch_size` 和 `--num_workers` 参数
- 确保硬件资源充足
- 考虑分批次处理数据

### 问题4：内存不足
- 减少批处理大小
- 增加系统内存
- 分批次处理大规模数据

### 问题5：事实性检查超时
- 检查网络连接
- 调整 `factscore_adapter.py` 中的超时设置
- 考虑使用更快的LLM模型

## 7. 性能优化建议

### 数据处理优化：
- 使用SSD存储提高IO速度
- 启用并行处理
- 分批次处理大规模数据

### 系统配置优化：
- 调整 `config.py` 中的参数
- 优化LLM调用参数
- 合理设置批处理大小

### Ollama优化：
- 选择适合硬件的模型大小
- 调整Ollama的内存使用设置
- 考虑使用量化模型减少资源需求

## 8. 交接检查清单

### 环境准备
- [ ] 仓库已成功克隆
- [ ] 虚拟环境已创建并激活
- [ ] 依赖已安装
- [ ] LLM配置已完成（GLM或Ollama）

### 历史背景索引构建
- [ ] 历史背景数据已准备（1万-10万条）
- [ ] 历史背景数据处理命令能够正常运行
- [ ] 历史背景索引能够成功构建
- [ ] 索引文件已正确生成

### 回忆录测试
- [ ] 回忆录文本数据已准备（大批量）
- [ ] 批量测试命令能够正常运行
- [ ] 系统能够从历史背景索引中检索信息
- [ ] 历史背景注入功能正常工作
- [ ] 事实性检查功能正常工作

### 系统验证
- [ ] Web界面能够正常访问
- [ ] 单条回忆录测试能够正常完成
- [ ] 批量回忆录测试能够正常完成
- [ ] 系统性能满足要求

## 9. 联系方式

如果遇到问题，请联系：
- 原作者：[您的名字]
- 技术支持：[联系方式]

---

**注意事项：**
- 首次运行时会下载Nomic模型，可能需要较长时间
- 大规模数据处理可能需要数小时，请耐心等待
- 确保有足够的磁盘空间存储索引和模型
- 定期备份索引数据，避免意外丢失
- 使用Ollama时，确保模型大小与硬件资源匹配

## 10. 完整流程模拟示例

### 场景假设：
- **历史背景数据**：10万条历史事件、人物、地点等信息
- **回忆录文本**：100条个人回忆录
- **Ollama模型**：已部署qwen-32B模型
- **硬件配置**：16核CPU，32GB内存，1TB SSD

### 步骤1：环境准备
```bash
# 1. 克隆仓库
git clone <仓库地址>
cd GraphRAG

# 2. 创建并激活虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置Ollama（已部署qwen-32B模型）
# 确保Ollama服务正在运行
# 检查模型是否可用
ollama list
```

### 步骤2：准备数据
```bash
# 1. 创建数据目录
mkdir -p data\background
mkdir -p data\memoirs

# 2. 放置历史背景数据（10万条）到 data\background 目录
# 3. 放置回忆录文本（100条）到 data\memoirs 目录
```

### 步骤3：处理历史背景数据
```bash
# 1. 批量处理历史背景数据
# 设置较大的批处理大小和并行线程数以提高速度
python run_batch.py \
    --input_dir ./data/background \
    --output_dir ./data/background_processed \
    --batch_size 1000 \
    --num_workers 8 \
    --llm_provider ollama \
    --llm_model qwen-32b

# 预计时间：约1-2小时（取决于硬件性能）
```

### 步骤4：构建历史背景索引
```bash
# 构建历史背景索引
python run_index.py \
    --data_dir ./data/background_processed \
    --index_dir ./index/background \
    --batch_size 1000 \
    --num_workers 8

# 预计时间：约2-3小时（10万条数据）
```

### 步骤5：批量测试回忆录
```bash
# 批量测试100条回忆录
python run_test.py \
    --input_dir ./data/memoirs \
    --output_dir ./data/test_results \
    --index_dir ./index/background \
    --batch_size 10 \
    --num_workers 4 \
    --llm_provider ollama \
    --llm_model qwen-32b

# 预计时间：约30-60分钟（100条回忆录）
```

### 步骤6：查看测试结果
```bash
# 查看测试结果目录
dir ./data/test_results

# 查看生成的增强文本和事实性检查报告
```

### 步骤7：使用Web界面进行交互式测试
```bash
# 启动Web界面
python run_web.py gradio \
    --llm_provider ollama \
    --llm_model qwen-32b

# 访问 http://localhost:8000 进行交互式测试
```

### 性能优化建议：
1. **硬件优化**：使用SSD存储，增加内存到64GB效果更佳
2. **批处理大小**：根据内存大小调整，32GB内存建议batch_size=1000
3. **并行线程**：设置为CPU核心数的一半，避免资源竞争
4. **模型选择**：如果qwen-32B速度太慢，可考虑使用更小的模型如qwen-7B

### 预期结果：
- 成功构建10万条历史背景数据的索引
- 100条回忆录文本都能成功注入历史背景
- 生成的文本包含相关历史背景信息
- 事实性检查能够验证注入的历史背景

祝您交接顺利！