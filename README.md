# 记忆图谱

基于RAG与知识图谱的个人回忆录历史背景自动注入系统

## 项目概述

本项目是一个"历史背景注入"模块，能够：

1. 从回忆录文本中自动提取时间、地点、关键事件
2. 基于知识图谱检索相关的历史背景信息
3. 使用大语言模型将历史事实"润色"成具有文学性的描述
4. 将生成的内容无缝融入个人叙事中

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        记忆图谱系统                          │
├─────────────────────────────────────────────────────────────┤
│  输入层                                                      │
│  ├── 回忆录文稿                                              │
│  └── 历史事件数据                                            │
├─────────────────────────────────────────────────────────────┤
│  核心引擎                                                    │
│  ├── 实体抽取器 (MemoirParser)                              │
│  ├── 知识图谱构建 (GraphBuilder)                            │
│  ├── 向量数据库 (GraphRAG内置)                              │
│  ├── 图谱检索器 (MemoirRetriever)                           │
│  └── 多LLM路由器 (LLMRouter)                                │
├─────────────────────────────────────────────────────────────┤
│  LLM服务层                                                   │
│  ├── Deepseek                                                │
│  ├── Qwen3 (通义千问)                                        │
│  ├── Hunyuan (腾讯混元)                                      │
│  ├── Google Gemini                                           │
│  └── Ollama (本地模型)                                       │
├─────────────────────────────────────────────────────────────┤
│  输出层                                                      │
│  ├── Web界面 (Gradio)                                        │
│  ├── REST API (FastAPI)                                      │
│  └── 评估模块 (Evaluator)                                   │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API密钥

复制 `env.example` 为 `.env` 并填入您的API密钥：

```bash
cp env.example .env
```

编辑 `.env` 文件：

```env
# Deepseek API
DEEPSEEK_API_KEY=your_deepseek_api_key

# Qwen (阿里云通义千问)
QWEN_API_KEY=your_qwen_api_key

# 腾讯混元
HUNYUAN_API_KEY=your_hunyuan_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key

# 默认 LLM 配置（可选 Ollama 本地模型）
DEFAULT_LLM_PROVIDER=gemini
DEFAULT_LLM_MODEL=gemini-2.5-flash

# 使用 Ollama 本地模型时
# DEFAULT_LLM_PROVIDER=ollama
# DEFAULT_LLM_MODEL=qwen3:32b
```

### 3. 准备历史数据

将历史事件数据放入 `data/input/` 目录，支持以下格式：

- **TXT**: 事件之间用 `---` 分隔
- **JSON**: 结构化的事件列表
- **CSV**: 包含 title, date, location, content 列

### 4. 构建知识图谱索引

```bash
python -c "from src.indexing import GraphBuilder; GraphBuilder().build_index_sync()"
```

### 5. 启动Web应用

```bash
# 启动Gradio界面
python run_web.py gradio

# 或启动FastAPI服务
python run_web.py api

# 同时启动两者
python run_web.py both
```

访问 http://localhost:8000 使用系统。

### 6. 使用本地模型 (Ollama)

如果你想使用本地部署的模型，可以配置 Ollama：

```bash
# 确保 Ollama 服务已启动
ollama serve

# 在 .env 中配置
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:32b
```

### 启动流程总结

每次启动项目时，需要执行以下步骤：

```powershell
# 1. 激活虚拟环境
cd D:\projects\Capstone\GraphRAG
.\venv\Scripts\Activate.ps1

# 2. 启动应用
python run_web.py gradio
```

**完整启动命令（一行）**：

```powershell
cd D:\projects\Capstone\GraphRAG; .\venv\Scripts\Activate.ps1; python run_web.py gradio
```

### 界面说明

应用有四个功能标签页：

| 标签 | 功能 |
|------|------|
| 📝 生成历史背景 | 输入回忆录片段，生成相关历史背景 |
| 🔄 多模型对比 | 同时对比多个LLM的生成效果 |
| ⚙️ 索引管理 | 管理历史事件知识图谱索引 |
| 📖 使用说明 | 查看详细使用指南 |

### ⚠️ 注意事项

- **Python版本要求**：GraphRAG 需要 Python 3.10-3.12
- **API密钥配置**：使用前需要配置 `.env` 文件中的 API 密钥
- **首次使用**：需要先在"索引管理"标签页构建知识图谱索引

## 项目结构

```
GraphRAG/
├── src/
│   ├── config/           # 配置管理
│   │   └── settings.py   # 环境变量配置
│   ├── llm/              # 多LLM适配器
│   │   ├── adapter.py    # LLM适配器抽象
│   │   ├── factory.py    # 适配器工厂
│   │   └── router.py     # 多LLM路由器
│   ├── indexing/         # 知识图谱构建
│   │   ├── graph_builder.py  # GraphRAG索引构建
│   │   └── data_loader.py    # 数据加载器
│   ├── retrieval/        # 检索模块
│   │   ├── memoir_parser.py    # 回忆录解析
│   │   └── memoir_retriever.py # 图谱检索
│   ├── generation/       # 文本生成
│   │   ├── literary_generator.py # 文学润色生成
│   │   └── prompts.py    # 提示词模板
│   └── evaluation/       # 评估模块
│       ├── evaluator.py  # 评估器
│       └── metrics.py    # 评估指标
├── data/
│   ├── input/            # 历史数据输入
│   └── output/           # 索引输出
├── web/
│   ├── app.py            # Gradio应用
│   └── api.py            # FastAPI服务
├── tests/                # 测试用例
├── requirements.txt      # 依赖列表
├── run_web.py           # 启动脚本
└── README.md
```

## 功能模块

### 1. 多LLM支持

系统支持多个LLM提供商：

| 提供商 | 模型 | 特点 |
|--------|------|------|
| Deepseek | deepseek-chat | 高性价比，中文优化 |
| Qwen | qwen-plus | 阿里云，中文能力强 |
| Hunyuan | hunyuan-lite | 腾讯，国产优化 |
| Gemini | gemini-2.5-flash | Google，多模态能力 |
| Ollama | qwen3:32b | 本地部署，数据隐私 |

### 2. 写作风格

支持多种写作风格：

- **standard**: 标准风格，平衡的文学性描述
- **nostalgic**: 怀旧风格，温暖回忆的笔调
- **narrative**: 叙事融合，与个人故事深度交织
- **informative**: 简洁信息，重点突出的背景介绍
- **conversational**: 对话风格，像讲故事一样亲切

### 3. 评估体系

三维评估指标：

1. **事实准确性** (Accuracy)
   - 实体覆盖率
   - 时间一致性
   
2. **相关性** (Relevance)
   - 关键词重叠度
   - 语义相似度
   
3. **文学性** (Literary)
   - 长度适当性
   - 段落结构
   - 过渡词使用
   - 描述丰富度

## API接口

### 生成历史背景

```bash
POST /generate
Content-Type: application/json

{
    "memoir_text": "1988年夏天，我从大学毕业，来到了深圳...",
    "provider": "deepseek",
    "style": "standard",
    "temperature": 0.7
}
```

### 多模型对比

```bash
POST /compare
Content-Type: application/json

{
    "memoir_text": "1988年夏天，我从大学毕业...",
    "providers": ["deepseek", "qwen"],
    "temperature": 0.7
}
```

### 索引状态

```bash
GET /index/status
```

## 使用示例

```python
from src.llm import create_llm_adapter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator

# 创建组件
adapter = create_llm_adapter(provider="deepseek")
retriever = MemoirRetriever(llm_adapter=adapter)
generator = LiteraryGenerator(llm_adapter=adapter)

# 输入回忆录
memoir = "1988年夏天，我从大学毕业，怀揣着梦想来到了深圳..."

# 检索历史背景
import asyncio
retrieval_result = asyncio.run(retriever.retrieve(memoir))

# 生成文学化描述
result = asyncio.run(generator.generate(memoir, retrieval_result))
print(result.content)
```

## 详细文档

- 📖 [GraphRAG 部署与实现指南](docs/GRAPHRAG_GUIDE.md) - 详细介绍 GraphRAG 在本项目中的部署和实现
- 🏗️ [系统技术架构文档](docs/TECHNICAL_ARCHITECTURE.md) - 完整的系统架构、设计模式和扩展指南
- 📊 [GraphRAG vs Mem0 对比分析](docs/GRAPHRAG_VS_MEM0_COMPARISON.md) - 两种 RAG 方案的对比

## 参考资源

- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
- **GraphRAG 文档**: https://microsoft.github.io/graphrag/
- **GraphRAG 论文**: https://arxiv.org/abs/2404.16130

## 开发团队

Capstone Project Group2 2026

## License

MIT License
