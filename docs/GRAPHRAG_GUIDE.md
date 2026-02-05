# GraphRAG 部署与实现指南

本文档详细介绍了 Microsoft GraphRAG 在"记忆图谱"项目中的部署和实现方式。

## 目录

1. [GraphRAG 简介](#1-graphrag-简介)
2. [安装配置](#2-安装配置)
3. [核心概念](#3-核心概念)
4. [项目集成架构](#4-项目集成架构)
5. [知识图谱构建流程](#5-知识图谱构建流程)
6. [检索实现](#6-检索实现)
7. [自定义配置](#7-自定义配置)
8. [常见问题](#8-常见问题)

---

## 1. GraphRAG 简介

### 1.1 什么是 GraphRAG？

GraphRAG 是由 Microsoft Research 开发的一种**检索增强生成（RAG）**方法，它结合了知识图谱技术来增强大语言模型的能力。

**官方资源：**
- 📦 **GitHub 仓库**: https://github.com/microsoft/graphrag
- 📖 **官方文档**: https://microsoft.github.io/graphrag/
- 📄 **研究论文**: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130)

### 1.2 GraphRAG vs 传统 RAG

| 特性 | 传统 RAG | GraphRAG |
|------|----------|----------|
| 数据结构 | 向量数据库 | 知识图谱 + 向量 |
| 检索方式 | 语义相似度 | 图遍历 + 语义 |
| 跨文档关联 | ❌ 困难 | ✅ 天然支持 |
| 复杂推理 | ❌ 有限 | ✅ 多跳推理 |
| 全局理解 | ❌ 局部上下文 | ✅ 社区摘要 |

### 1.3 为什么选择 GraphRAG？

对于"历史背景注入"场景，GraphRAG 的优势在于：

1. **连接分散信息**: 能够将不同历史事件之间的关联连接起来
2. **时间线理解**: 通过图结构理解事件的先后顺序和因果关系
3. **实体消歧**: 识别不同文档中的同一实体（如"邓公"="邓小平"）
4. **社区报告**: 生成主题性的历史背景摘要

---

## 2. 安装配置

### 2.1 安装 GraphRAG

```bash
# 使用 pip 安装
pip install graphrag

# 或指定版本
pip install graphrag>=0.3.0
```

**PyPI 地址**: https://pypi.org/project/graphrag/

### 2.2 依赖要求

GraphRAG 需要以下核心依赖：

```txt
# requirements.txt 中的 GraphRAG 相关依赖
graphrag>=0.3.0
openai>=1.30.0          # LLM 调用
tiktoken>=0.7.0         # Token 计算
pandas>=2.0.0           # 数据处理
networkx>=3.0           # 图结构
```

### 2.3 API 密钥配置

GraphRAG 默认使用 OpenAI API，但可以配置为使用其他兼容 API：

```bash
# .env 文件配置
GRAPHRAG_API_KEY=your_api_key_here

# 或使用我们项目的多 LLM 配置
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

---

## 3. 核心概念

### 3.1 GraphRAG 工作流程

```
┌──────────────────────────────────────────────────────────────────┐
│                    GraphRAG 处理流程                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   原始文档 ──┬──▶ 文本分块 ──▶ 实体抽取 ──▶ 关系抽取             │
│              │                                                    │
│              └──▶ 知识图谱 ──▶ 社区检测 ──▶ 社区摘要             │
│                      │                                            │
│                      ▼                                            │
│              ┌─────────────┐                                      │
│              │  索引存储   │                                      │
│              │ (Parquet)   │                                      │
│              └─────────────┘                                      │
│                      │                                            │
│                      ▼                                            │
│   用户查询 ──▶ 本地搜索 / 全局搜索 ──▶ LLM 生成 ──▶ 答案         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

| 组件 | 说明 | 输出文件 |
|------|------|----------|
| Text Units | 文本分块单元 | `create_final_text_units.parquet` |
| Entities | 抽取的实体 | `create_final_entities.parquet` |
| Relationships | 实体间关系 | `create_final_relationships.parquet` |
| Communities | 社区/集群 | `create_final_communities.parquet` |
| Community Reports | 社区摘要报告 | `create_final_community_reports.parquet` |

### 3.3 两种搜索模式

**本地搜索 (Local Search)**:
- 适用于具体问题
- 检索相关实体和关系
- 速度快，精度高

**全局搜索 (Global Search)**:
- 适用于概括性问题
- 利用社区报告
- 覆盖面广，适合总结

---

## 4. 项目集成架构

### 4.1 在本项目中的位置

```
记忆图谱项目
│
├── src/indexing/
│   ├── graph_builder.py      ◄── GraphRAG 索引构建封装
│   └── data_loader.py        ◄── 历史数据加载
│
├── src/retrieval/
│   ├── memoir_retriever.py   ◄── 基于 GraphRAG 索引的检索
│   └── memoir_parser.py      ◄── 回忆录解析（查询预处理）
│
└── data/
    ├── input/                ◄── 历史事件原始数据
    └── output/               ◄── GraphRAG 索引输出
        ├── create_final_entities.parquet
        ├── create_final_relationships.parquet
        ├── create_final_communities.parquet
        └── ...
```

### 4.2 集成流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     项目集成架构                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   历史事件数据                                                   │
│   (维基百科、档案等)                                             │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐                                               │
│   │ DataLoader  │  加载 TXT/JSON/CSV 格式                       │
│   └─────────────┘                                               │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐     ┌──────────────────┐                      │
│   │GraphBuilder │────▶│ Microsoft        │                      │
│   │(索引构建)   │     │ GraphRAG         │                      │
│   └─────────────┘     │ (实体/关系抽取)  │                      │
│                       └──────────────────┘                      │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────────────────────┐                       │
│   │         索引文件 (Parquet)           │                       │
│   │  • entities  • relationships         │                       │
│   │  • communities  • text_units         │                       │
│   └─────────────────────────────────────┘                       │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────┐                                          │
│   │ MemoirRetriever  │  检索相关历史背景                        │
│   └──────────────────┘                                          │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────┐                                          │
│   │LiteraryGenerator │  文学化润色                              │
│   └──────────────────┘                                          │
│         │                                                        │
│         ▼                                                        │
│      最终输出                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 知识图谱构建流程

### 5.1 GraphBuilder 类实现

我们在 `src/indexing/graph_builder.py` 中封装了 GraphRAG 的索引构建：

```python
from src.indexing import GraphBuilder

# 创建构建器
builder = GraphBuilder(
    input_dir="./data/input",      # 历史数据目录
    output_dir="./data/output",    # 索引输出目录
    llm_provider="deepseek",       # 使用的 LLM
    llm_model="deepseek-chat"      # 模型名称
)

# 构建索引
result = builder.build_index_sync()
print(result.message)
```

### 5.2 配置文件生成

GraphRAG 需要 `settings.yaml` 配置文件，我们的 `GraphBuilder` 会自动生成：

```yaml
# 自动生成的 settings.yaml 示例

encoding_model: cl100k_base

llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: deepseek-chat
  api_base: https://api.deepseek.com/v1
  max_tokens: 4096
  temperature: 0

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  entity_types: [历史事件, 人物, 地点, 时间, 组织, 政策, 社会现象]
  max_gleanings: 1

community_reports:
  max_length: 2000
```

### 5.3 自定义实体类型

针对历史背景场景，我们定义了专用的实体类型：

```python
ENTITY_TYPES = [
    "历史事件",    # 如：改革开放、亚洲金融危机
    "人物",        # 如：邓小平、袁隆平
    "地点",        # 如：深圳、北京
    "时间",        # 如：1988年、90年代
    "组织",        # 如：国务院、深圳市政府
    "政策",        # 如：经济特区政策
    "社会现象",    # 如：下海潮、民工潮
    "经济指标",    # 如：GDP增长率
    "文化运动",    # 如：思想解放运动
]
```

### 5.4 自定义提示词

我们为历史事件场景优化了实体抽取提示词：

```python
# prompts/entity_extraction.txt

给定一段可能与历史事件相关的文本，识别出文本中所有的实体及其关系。

识别所有实体，包括：
- entity_name: 实体名称
- entity_type: 类型（历史事件/人物/地点/时间/组织/政策等）
- entity_description: 详细描述，包括历史背景和重要性

识别实体关系：
- 关系类型：发生于、影响、引发、导致、促进、阻碍等
- 关系强度：1-10分
```

### 5.5 运行索引构建

**方式一：命令行**

```bash
# 进入项目目录
cd D:\projects\Capstone\GraphRAG

# 使用 GraphRAG CLI
python -m graphrag.index --root ./data/output --config ./data/output/settings.yaml
```

**方式二：Python 代码**

```python
from src.indexing import GraphBuilder

builder = GraphBuilder(llm_provider="deepseek")
result = builder.build_index_sync()

if result.success:
    print("索引构建成功！")
    stats = builder.get_index_stats()
    print(f"实体数量: {stats['entities']}")
    print(f"关系数量: {stats['relationships']}")
```

**方式三：Web 界面**

启动 Web 应用后，在"索引管理"标签页点击"构建索引"按钮。

---

## 6. 检索实现

### 6.1 MemoirRetriever 类

我们在 `src/retrieval/memoir_retriever.py` 中实现了基于 GraphRAG 索引的检索：

```python
from src.retrieval import MemoirRetriever

# 创建检索器
retriever = MemoirRetriever(index_dir="./data/output")

# 执行检索
result = retriever.retrieve_sync(
    memoir_text="1988年，我大学毕业来到深圳创业...",
    top_k=10
)

# 查看结果
print(f"找到 {len(result.entities)} 个相关实体")
print(f"找到 {len(result.communities)} 个相关社区报告")
```

### 6.2 检索流程

```
回忆录文本
    │
    ▼
┌─────────────────┐
│  MemoirParser   │  提取时间、地点、关键词
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  生成查询向量   │  "1988年 深圳 创业"
└─────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                   ▼
┌─────────────────┐           ┌─────────────────┐
│  实体检索       │           │  社区报告检索   │
│  (Local Search) │           │  (Global Search)│
└─────────────────┘           └─────────────────┘
    │                                   │
    └──────────────┬────────────────────┘
                   ▼
           ┌─────────────────┐
           │  RetrievalResult│
           │  • entities     │
           │  • relationships│
           │  • communities  │
           │  • text_units   │
           └─────────────────┘
```

### 6.3 读取 Parquet 索引文件

GraphRAG 的索引存储为 Parquet 格式，我们使用 Pandas 读取：

```python
import pandas as pd
from pathlib import Path

output_dir = Path("./data/output/output")

# 读取实体
entities_df = pd.read_parquet(output_dir / "create_final_entities.parquet")
print(entities_df.columns)
# ['id', 'name', 'type', 'description', 'human_readable_id', ...]

# 读取关系
rel_df = pd.read_parquet(output_dir / "create_final_relationships.parquet")
print(rel_df.columns)
# ['source', 'target', 'description', 'weight', ...]

# 读取社区报告
comm_df = pd.read_parquet(output_dir / "create_final_community_reports.parquet")
print(comm_df.columns)
# ['community', 'title', 'summary', 'full_content', 'level', ...]
```

### 6.4 检索匹配算法

我们实现了基于关键词的简单匹配（可扩展为向量检索）：

```python
def _search_entities(self, context: MemoirContext, top_k: int) -> List[Dict]:
    """搜索相关实体"""
    results = []
    
    # 构建搜索词
    search_terms = [context.year, context.location] + context.keywords
    
    for _, row in self._entities_df.iterrows():
        entity_name = str(row.get("name", ""))
        entity_desc = str(row.get("description", ""))
        
        # 计算匹配分数
        score = sum(1 for term in search_terms if term in entity_name or term in entity_desc)
        
        if score > 0:
            results.append({
                "name": entity_name,
                "type": row.get("type"),
                "description": entity_desc,
                "score": score,
            })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
```

---

## 7. 自定义配置

### 7.1 LLM 提供商切换

我们的系统支持多个 LLM 提供商，GraphRAG 索引构建时可以选择：

```python
# 使用 Deepseek
builder = GraphBuilder(llm_provider="deepseek")

# 使用 Qwen
builder = GraphBuilder(llm_provider="qwen", llm_model="qwen-plus")

# 使用 Google Gemini
builder = GraphBuilder(llm_provider="gemini", llm_model="gemini-1.5-flash")
```

### 7.2 分块参数调整

根据历史文档的特点调整分块：

```yaml
chunks:
  size: 1200      # 每块大小（字符数）
  overlap: 100    # 块间重叠
  group_by_columns: [id]
```

对于较长的历史文档，可以增大 `size`；对于需要保持上下文连贯性的文档，增大 `overlap`。

### 7.3 社区检测参数

控制知识图谱的社区划分：

```yaml
cluster_graph:
  max_cluster_size: 10    # 每个社区最大实体数

community_reports:
  max_length: 2000        # 社区报告最大长度
  max_input_length: 8000  # 输入LLM的最大长度
```

---

## 8. 常见问题

### Q1: 索引构建失败，提示 API 错误？

**解决方案**：
1. 检查 `.env` 文件中的 API 密钥是否正确
2. 确认 API Base URL 是否正确配置
3. 检查网络连接和代理设置

```bash
# 测试 API 连接
curl -X POST https://api.deepseek.com/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Hello"}]}'
```

### Q2: 如何处理大规模历史数据？

**建议**：
1. 分批次处理，每批 100-500 个文档
2. 使用更大的 `parallelization.num_threads` 值
3. 考虑使用 `async_mode: threaded` 加速

```yaml
parallelization:
  stagger: 0.3
  num_threads: 50    # 增加并行度

async_mode: threaded
```

### Q3: 实体抽取不准确怎么办？

**优化方法**：
1. 调整 `entity_extraction` 提示词，添加更多示例
2. 增加 `max_gleanings` 值（多次抽取合并）
3. 定义更精确的 `entity_types`

```yaml
entity_extraction:
  max_gleanings: 2    # 多次抽取
  entity_types: [历史事件, 人物, 地点, 时间, 组织]
```

### Q4: 如何更新已有的索引？

目前 GraphRAG 不支持增量更新，需要重新构建：

```python
# 添加新数据后重新构建
builder = GraphBuilder()
result = builder.build_index_sync()
```

### Q5: 检索结果不相关怎么办？

**改进方法**：
1. 优化 `MemoirParser` 的关键词提取
2. 使用向量检索替代关键词匹配
3. 调整 `top_k` 参数

---

## 参考资源

### 官方资源

- **GitHub**: https://github.com/microsoft/graphrag
- **文档**: https://microsoft.github.io/graphrag/
- **论文**: https://arxiv.org/abs/2404.16130
- **PyPI**: https://pypi.org/project/graphrag/

### 相关项目

- **LlamaIndex GraphRAG**: https://github.com/run-llama/llama_index
- **LangChain**: https://github.com/langchain-ai/langchain
- **RAGFlow**: https://github.com/infiniflow/ragflow

### 社区资源

- **GraphRAG 中文社区**: https://graphragcn.com/
- **Microsoft Research Blog**: https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/

---

## 版本信息

- **文档版本**: 1.0
- **GraphRAG 版本**: >= 0.3.0
- **更新日期**: 2026-02-04
- **作者**: Capstone Project Group2

---

## 附录：完整示例

### A. 从零开始构建历史知识图谱

```python
#!/usr/bin/env python
"""完整的 GraphRAG 使用示例"""

import asyncio
from pathlib import Path

# 1. 导入模块
from src.indexing import GraphBuilder, DataLoader, HistoricalEvent
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator
from src.llm import create_llm_adapter

async def main():
    # 2. 准备历史数据
    events = [
        HistoricalEvent(
            title="深圳经济特区成立",
            date="1980年8月26日",
            location="深圳",
            content="全国人大常委会批准设立深圳经济特区...",
        ),
        HistoricalEvent(
            title="南方谈话",
            date="1992年1月",
            location="深圳、珠海",
            content="邓小平发表著名的南方谈话，确立市场经济改革方向...",
        ),
    ]
    
    # 3. 保存为 GraphRAG 输入格式
    loader = DataLoader(data_dir="./data/input")
    loader.save_as_graphrag_input(events)
    
    # 4. 构建知识图谱索引
    builder = GraphBuilder(llm_provider="deepseek")
    result = builder.build_index_sync()
    
    if not result.success:
        print(f"索引构建失败: {result.message}")
        return
    
    print(f"索引构建成功！实体数: {builder.get_index_stats()['entities']}")
    
    # 5. 创建检索器和生成器
    adapter = create_llm_adapter(provider="deepseek")
    retriever = MemoirRetriever(llm_adapter=adapter)
    generator = LiteraryGenerator(llm_adapter=adapter)
    
    # 6. 处理回忆录
    memoir = "1988年夏天，我从北方的一所大学毕业，怀揣着对未来的憧憬，踏上了南下深圳的火车..."
    
    # 7. 检索历史背景
    retrieval_result = await retriever.retrieve(memoir, top_k=10)
    print(f"检索到 {len(retrieval_result.entities)} 个相关实体")
    
    # 8. 生成文学化描述
    gen_result = await generator.generate(memoir, retrieval_result)
    
    print("\n" + "="*50)
    print("生成的历史背景：")
    print("="*50)
    print(gen_result.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### B. GraphRAG 索引文件结构

```
data/output/
├── settings.yaml              # 配置文件
├── prompts/                   # 提示词目录
│   ├── entity_extraction.txt
│   ├── summarize_descriptions.txt
│   └── community_report.txt
├── cache/                     # 缓存目录
├── reports/                   # 报告目录
└── output/                    # 索引输出
    ├── create_final_entities.parquet
    ├── create_final_relationships.parquet
    ├── create_final_communities.parquet
    ├── create_final_community_reports.parquet
    ├── create_final_text_units.parquet
    └── stats.json
```
