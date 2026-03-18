# 记忆图谱系统技术文档

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [GraphRAG 核心原理](#3-graphrag-核心原理)
4. [各层详解](#4-各层详解)
5. [设计模式解析](#5-设计模式解析)
6. [数据流与调用链](#6-数据流与调用链)
7. [扩展指南](#7-扩展指南)

---

## 1. 系统概述

记忆图谱系统是一个基于 **Microsoft GraphRAG** 和多 LLM 的个人回忆录历史背景自动注入系统。

### 1.1 核心功能

```
回忆录文本 → 实体提取 → 知识图谱检索 → 历史背景生成 → 文学化润色输出
```

### 1.2 技术栈

| 层级 | 技术选型 |
|------|----------|
| 知识图谱 | Microsoft GraphRAG 2.x |
| 向量数据库 | LanceDB |
| LLM 集成 | LiteLLM (统一多模型接口) |
| Web 框架 | FastAPI + Gradio |
| 数据处理 | Pandas + Pydantic |

---

## 2. 整体架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         表示层 (Presentation)                    │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │   Gradio UI     │  │   FastAPI REST  │                       │
│  │   (app.py)      │  │   (api.py)      │                       │
│  └────────┬────────┘  └────────┬────────┘                       │
├───────────┴────────────────────┴────────────────────────────────┤
│                         业务逻辑层 (Business Logic)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐          │
│  │  Generation │  │  Retrieval  │  │   Evaluation    │          │
│  │  文学生成    │  │  检索模块    │  │   评估模块      │          │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘          │
├─────────┴────────────────┴─────────────────┬┴───────────────────┤
│                         LLM 服务层 (LLM Services)                │
│  ┌───────────────────────────────────────────────────────┐      │
│  │                    LLM Router                          │      │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │      │
│  │   │Deepseek │ │  Qwen   │ │ Gemini  │ │ Ollama  │     │      │
│  │   │Adapter  │ │ Adapter │ │ Adapter │ │ Adapter │     │      │
│  │   └─────────┘ └─────────┘ └─────────┘ └─────────┘     │      │
│  └───────────────────────┬───────────────────────────────┘      │
│                          │ LiteLLM                               │
├──────────────────────────┴──────────────────────────────────────┤
│                         数据层 (Data Layer)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐          │
│  │  Indexing   │  │  GraphRAG   │  │    LanceDB      │          │
│  │  数据加载    │  │  知识图谱    │  │    向量存储     │          │
│  └─────────────┘  └─────────────┘  └─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 模块依赖关系

```
src/
├── config/                 # 配置管理
│   └── settings.py         # Pydantic Settings 配置
├── llm/                    # LLM 服务层
│   ├── adapter.py          # 适配器抽象 + 具体实现
│   ├── factory.py          # 工厂模式 - 创建适配器
│   └── router.py           # 路由器 - 多 LLM 调度
├── indexing/               # 索引构建层
│   ├── data_loader.py      # 数据加载器
│   └── graph_builder.py    # GraphRAG 索引构建
├── retrieval/              # 检索层
│   ├── memoir_parser.py    # 回忆录解析器
│   └── memoir_retriever.py # 知识图谱检索
├── generation/             # 生成层
│   ├── prompts.py          # 提示词模板库
│   └── literary_generator.py # 文学润色生成
└── evaluation/             # 评估层
    ├── metrics.py          # 评估指标
    └── evaluator.py        # 评估器
```

---

## 3. GraphRAG 核心原理

### 3.1 什么是 GraphRAG

GraphRAG 是微软提出的知识图谱增强 RAG 架构。与传统 RAG 的区别：

| 特性 | 传统 RAG | GraphRAG |
|------|----------|----------|
| 数据结构 | 平铺的文本块 | 知识图谱 (实体+关系) |
| 检索方式 | 向量相似度 | 图遍历 + 向量检索 |
| 理解粒度 | 局部语义 | 全局结构 |
| 适用场景 | 事实问答 | 复杂推理、多跳问答 |

### 3.2 GraphRAG 索引构建流程

```
原始文本
    │
    ▼
┌─────────────────────────────────────┐
│  1. 文本分块 (Chunking)              │
│     - size: 1200 tokens             │
│     - overlap: 100 tokens           │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  2. 实体抽取 (Entity Extraction)    │  ← LLM 调用
│     - 人物、地点、事件、组织...       │
│     - 生成实体描述                   │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  3. 关系抽取 (Relationship Extract) │  ← LLM 调用
│     - 实体间的关联                   │
│     - 关系描述 + 权重                │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  4. 描述摘要 (Summarization)        │  ← LLM 调用
│     - 合并同一实体的多个描述          │
│     - 生成综合性摘要                 │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  5. 社区检测 (Community Detection)   │
│     - Leiden 算法                    │
│     - 层次化社区结构                 │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  6. 社区报告 (Community Reports)    │  ← LLM 调用
│     - 每个社区生成摘要报告           │
│     - 用于全局搜索                   │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  7. 向量嵌入 (Embedding)             │
│     - 实体描述向量化                 │
│     - 存入 LanceDB                   │
└─────────────────────────────────────┘
```

### 3.3 GraphRAG 检索模式

本项目支持三种检索模式：

#### Local Search（本地搜索）

```
查询: "1988年深圳发生了什么?"
        │
        ▼
┌─────────────────────┐
│  1. 向量相似度匹配   │  → 找到相关实体
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. 图遍历 1-2 跳    │  → 获取关联实体
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. 检索相关文本块   │  → 原始上下文
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. LLM 生成答案     │
└─────────────────────┘
```

#### Global Search（全局搜索）

```
查询: "改革开放的历史影响?"
        │
        ▼
┌─────────────────────┐
│  1. 遍历社区报告     │  → 计算相关性
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. Map 阶段        │  → 每个社区生成中间答案
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. Reduce 阶段     │  → 聚合生成最终答案
└─────────────────────┘
```

### 3.4 本项目的索引配置

项目通过 `graph_builder.py` 生成 GraphRAG 配置：

```yaml
# 核心配置项
models:
  default_chat_model:
    model_provider: gemini          # 或 deepseek/openai
    model: gemini-2.5-flash
  default_embedding_model:
    model: text-embedding-004

extract_graph:
  entity_types: [历史事件, 人物, 地点, 时间, 组织]  # 自定义实体类型

community_reports:
  max_length: 2000                  # 社区报告最大长度

vector_store:
  type: lancedb                     # 向量数据库
  db_uri: output/lancedb
```

---

## 4. 各层详解

### 4.1 LLM 服务层

#### 4.1.1 核心类图

```
                    ┌─────────────────┐
                    │   LLMProvider   │ (Enum)
                    │ DEEPSEEK, QWEN  │
                    │ GEMINI, OLLAMA  │
                    └─────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  LLMAdapter   │  │   LLMRouter   │  │  LLMFactory   │
│  (抽象基类)    │  │  (多LLM调度)   │  │  (工厂方法)    │
└───────┬───────┘  └───────────────┘  └───────────────┘
        │
        ├─── DeepseekAdapter
        ├─── QwenAdapter
        ├─── GeminiAdapter
        ├─── OllamaAdapter
        └─── OpenAIAdapter
```

#### 4.1.2 LLMAdapter 抽象基类

```python
class LLMAdapter(ABC):
    """所有 LLM 适配器的基类"""
    
    def __init__(self, api_key, api_base=None, model=None):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
    
    @abstractmethod
    def _get_litellm_model_name(self) -> str:
        """子类必须实现：返回 LiteLLM 格式的模型名"""
        pass
    
    async def chat(self, messages, temperature=0.7, max_tokens=2048):
        """统一的聊天接口，内部调用 LiteLLM"""
        response = await acompletion(
            model=self._get_litellm_model_name(),
            messages=messages,
            api_key=self.api_key,
            ...
        )
        return LLMResponse(...)
```

#### 4.1.3 LiteLLM 统一调用

不同 LLM 的调用通过 LiteLLM 统一：

| 提供商 | LiteLLM 格式 | 示例 |
|--------|-------------|------|
| Deepseek | `deepseek/{model}` | `deepseek/deepseek-chat` |
| Qwen | `openai/{model}` + api_base | `openai/qwen-plus` |
| Gemini | `gemini/{model}` | `gemini/gemini-2.5-flash` |
| Ollama | `ollama/{model}` | `ollama/qwen3:32b` |
| OpenAI | `{model}` | `gpt-4o-mini` |

### 4.2 索引构建层

#### 4.2.1 DataLoader - 数据加载器

支持三种输入格式：

```python
class DataLoader:
    def load_json(self, filepath) -> List[HistoricalEvent]:
        """JSON 格式：[{title, date, location, content}]"""
    
    def load_csv(self, filepath) -> List[HistoricalEvent]:
        """CSV 格式：必须包含 title, date, location, content 列"""
    
    def load_txt(self, filepath) -> List[HistoricalEvent]:
        """TXT 格式：事件之间用 --- 分隔"""
```

#### 4.2.2 GraphBuilder - 索引构建器

```python
class GraphBuilder:
    def create_settings_yaml(self):
        """生成 GraphRAG 配置文件"""
        # 动态配置 LLM 提供商
        # 设置实体类型、社区报告参数等
    
    async def build_index(self):
        """调用 GraphRAG CLI 构建索引"""
        # 1. 初始化 graphrag init
        # 2. 运行 graphrag index
        # 3. 生成 parquet 文件
```

索引产出物：

```
data/output/output/
├── entities.parquet        # 实体表
├── relationships.parquet   # 关系表
├── communities.parquet     # 社区表
├── community_reports.parquet # 社区报告
├── text_units.parquet      # 文本块
└── lancedb/                # 向量索引
    ├── default-entity-description.lance
    └── default-community-full_content.lance
```

### 4.3 检索层

#### 4.3.1 MemoirParser - 回忆录解析器

```python
class MemoirParser:
    """从回忆录文本中提取结构化信息"""
    
    def parse(self, text) -> MemoirContext:
        """规则提取：正则表达式匹配"""
        year = self._extract_year(text)      # 1988年, 88年
        location = self._extract_location(text)  # 深圳, 北京
        keywords = self._extract_keywords(text)  # 改革开放, 创业
        return MemoirContext(year, location, keywords)
    
    async def parse_with_llm(self, text) -> MemoirContext:
        """LLM 增强：深度语义理解"""
        # 使用 LLM 提取更精确的实体和关键词
```

#### 4.3.2 MemoirRetriever - 检索器

```python
class MemoirRetriever:
    """知识图谱检索"""
    
    def _load_index_data(self):
        """加载 Parquet 索引文件"""
        self._entities_df = pd.read_parquet("entities.parquet")
        self._relationships_df = pd.read_parquet("relationships.parquet")
        self._communities_df = pd.read_parquet("community_reports.parquet")
    
    async def retrieve(self, memoir_text, top_k=10) -> RetrievalResult:
        """执行检索"""
        # 1. 解析回忆录
        context = await self.parser.parse_with_llm(memoir_text)
        
        # 2. 实体匹配（关键词匹配，可扩展为向量检索）
        entities = self._search_entities(context, top_k)
        
        # 3. 关系匹配
        relationships = self._search_relationships(context, top_k)
        
        # 4. 社区报告检索
        communities = self._search_communities(context, top_k)
        
        return RetrievalResult(entities, relationships, communities)
```

### 4.4 生成层

#### 4.4.1 PromptTemplates - 提示词模板库

```python
class PromptTemplates:
    STANDARD = """标准风格 - 平衡的文学性描述"""
    NOSTALGIC = """怀旧风格 - 温暖回忆的笔调"""
    NARRATIVE = """叙事融合 - 与个人故事深度交织"""
    INFORMATIVE = """简洁信息 - 重点突出的背景介绍"""
    CONVERSATIONAL = """对话风格 - 像讲故事一样亲切"""
    
    @classmethod
    def get_template(cls, style: str) -> str:
        return templates.get(style, cls.STANDARD)
```

#### 4.4.2 LiteraryGenerator - 文学生成器

```python
class LiteraryGenerator:
    async def generate(self, memoir_text, retrieval_result):
        """单 LLM 生成"""
        prompt = self._build_prompt(memoir_text, retrieval_result)
        response = await self.llm_adapter.generate(prompt, system_prompt)
        return GenerationResult(content=response.content)
    
    async def generate_parallel(self, memoir_text, retrieval_result, providers):
        """多 LLM 并行生成（用于对比）"""
        return await self.llm_router.generate_parallel(prompt, providers)
```

### 4.5 评估层

#### 4.5.1 三维评估体系

```
┌────────────────────────────────────────────────────────────┐
│                    评估维度                                 │
├──────────────┬──────────────┬──────────────────────────────┤
│  准确性       │  相关性       │  文学性                      │
│  Accuracy    │  Relevance   │  Literary                    │
├──────────────┼──────────────┼──────────────────────────────┤
│ • 时间匹配    │ • 主题一致    │ • 长度适当性                 │
│ • 地点匹配    │ • 关键词重叠  │ • 段落结构                   │
│ • 实体引用    │ • 语义相似度  │ • 过渡词使用                 │
│ • 事实无误    │ • 上下文契合  │ • 表现力                     │
└──────────────┴──────────────┴──────────────────────────────┘
```

#### 4.5.2 Evaluator - 评估器

```python
class Evaluator:
    async def evaluate(self, memoir_text, generated_text, use_llm=True):
        if use_llm:
            return await self._evaluate_with_llm(...)  # LLM 打分
        else:
            return self._evaluate_simple(...)          # 规则打分
    
    def _evaluate_accuracy_simple(self, generated_text, retrieval_result):
        """准确性评估：检查实体引用、时间一致性"""
        
    def _evaluate_relevance_simple(self, memoir_text, generated_text):
        """相关性评估：词汇重叠、关键词匹配"""
        
    def _evaluate_literary_simple(self, memoir_text, generated_text):
        """文学性评估：长度、段落、过渡词"""
```

---

## 5. 设计模式解析

### 5.1 工厂模式 (Factory Pattern)

#### 5.1.1 模式定义

工厂模式是一种创建型设计模式，提供一个创建对象的接口，让子类决定实例化哪一个类。

#### 5.1.2 本项目的实现

```python
# factory.py

# 1. 注册表：映射提供商到适配器类
ADAPTER_REGISTRY: Dict[LLMProvider, Type[LLMAdapter]] = {
    LLMProvider.DEEPSEEK: DeepseekAdapter,
    LLMProvider.QWEN: QwenAdapter,
    LLMProvider.GEMINI: GeminiAdapter,
    LLMProvider.OLLAMA: OllamaAdapter,
    LLMProvider.OPENAI: OpenAIAdapter,
}

# 2. 工厂函数
def create_llm_adapter(
    provider: str = None,
    model: str = None,
    api_key: str = None,
) -> LLMAdapter:
    """
    工厂方法：根据 provider 创建对应的适配器实例
    
    优点：
    1. 解耦：调用者无需知道具体类
    2. 扩展：新增 LLM 只需添加适配器和注册
    3. 配置：自动从环境变量读取 API 密钥
    """
    # 获取提供商枚举
    provider_enum = LLMProvider(provider.lower())
    
    # 从注册表获取适配器类
    adapter_class = ADAPTER_REGISTRY[provider_enum]
    
    # 获取配置（API Key、Base URL）
    api_key = api_key or _get_api_key_from_settings(provider_enum)
    api_base = _get_api_base_from_settings(provider_enum)
    
    # 创建并返回实例
    return adapter_class(api_key=api_key, api_base=api_base, model=model)
```

#### 5.1.3 工厂模式 UML

```
┌─────────────────────────────────────────────────────────────┐
│                     Client (调用方)                          │
│                                                             │
│   adapter = create_llm_adapter(provider="ollama")           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Factory (工厂)                             │
│                                                             │
│   ADAPTER_REGISTRY = {                                      │
│       DEEPSEEK: DeepseekAdapter,                            │
│       OLLAMA: OllamaAdapter,                                │
│       ...                                                   │
│   }                                                         │
│                                                             │
│   def create_llm_adapter(provider):                         │
│       return ADAPTER_REGISTRY[provider](api_key, ...)       │
└─────────────────────────────┬───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │ Deepseek  │       │  Ollama   │       │  Gemini   │
    │  Adapter  │       │  Adapter  │       │  Adapter  │
    └───────────┘       └───────────┘       └───────────┘
```

#### 5.1.4 工厂模式的优势

| 优势 | 本项目体现 |
|------|-----------|
| **解耦** | 业务代码只依赖 `create_llm_adapter()`，不依赖具体适配器类 |
| **扩展性** | 添加 Ollama 只需：1) 写适配器类 2) 注册到 REGISTRY |
| **配置集中** | API 密钥、Base URL 在工厂内统一处理 |
| **测试友好** | 可轻松替换为 Mock 适配器 |

### 5.2 适配器模式 (Adapter Pattern)

#### 5.2.1 模式定义

适配器模式将一个类的接口转换为客户期望的另一个接口，使原本不兼容的类可以一起工作。

#### 5.2.2 本项目的实现

```python
# 问题：不同 LLM 的调用方式不同
# Deepseek: deepseek/deepseek-chat
# Qwen: openai/qwen-plus (需要 api_base)
# Gemini: gemini/gemini-2.5-flash
# Ollama: ollama/qwen3:32b

# 解决：统一适配为相同接口
class LLMAdapter(ABC):
    """统一接口"""
    async def generate(self, prompt, system_prompt) -> LLMResponse:
        pass

class DeepseekAdapter(LLMAdapter):
    def _get_litellm_model_name(self):
        return f"deepseek/{self.model}"  # 适配 Deepseek 格式

class OllamaAdapter(LLMAdapter):
    def _get_litellm_model_name(self):
        return f"ollama/{self.model}"   # 适配 Ollama 格式
```

### 5.3 策略模式 (Strategy Pattern)

#### 5.3.1 在提示词模板中的应用

```python
class PromptTemplates:
    """策略模式：不同风格是不同的策略"""
    
    STANDARD = "..."    # 策略 1
    NOSTALGIC = "..."   # 策略 2
    NARRATIVE = "..."   # 策略 3
    
    @classmethod
    def get_template(cls, style: str) -> str:
        """根据 style 选择策略"""
        return templates[style]
```

使用：

```python
# 运行时动态选择策略
template = PromptTemplates.get_template("nostalgic")
prompt = template.format(memoir_text=text, ...)
```

### 5.4 单例模式 (Singleton Pattern)

#### 5.4.1 在配置管理中的应用

```python
# settings.py
from functools import lru_cache

@lru_cache()  # 确保只创建一个实例
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
```

---

## 6. 数据流与调用链

### 6.1 完整数据流

```
                          用户输入
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Web 层                                                   │
│     Gradio/FastAPI 接收回忆录文本                            │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 解析层                                                   │
│     MemoirParser.parse_with_llm()                           │
│     └─> 提取 year, location, keywords                       │
│     └─> 调用 LLM 深度理解                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 检索层                                                   │
│     MemoirRetriever.retrieve()                              │
│     └─> 加载 parquet 索引                                    │
│     └─> 实体匹配、关系匹配、社区报告检索                       │
│     └─> 返回 RetrievalResult                                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 生成层                                                   │
│     LiteraryGenerator.generate()                            │
│     └─> 构建提示词 (memoir + context)                        │
│     └─> 调用 LLM 生成文学化描述                              │
│     └─> 返回 GenerationResult                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 评估层 (可选)                                            │
│     Evaluator.evaluate()                                    │
│     └─> 三维打分：准确性、相关性、文学性                       │
│     └─> 返回 EvaluationResult                               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                          输出结果
```

### 6.2 代码调用示例

```python
# 完整调用链
from src.llm import create_llm_adapter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator
from src.evaluation import Evaluator

# 1. 创建 LLM 适配器（工厂模式）
adapter = create_llm_adapter(provider="ollama", model="qwen3:32b")

# 2. 创建检索器
retriever = MemoirRetriever(llm_adapter=adapter)

# 3. 创建生成器
generator = LiteraryGenerator(llm_adapter=adapter)

# 4. 创建评估器
evaluator = Evaluator(llm_adapter=adapter)

# 5. 执行完整流程
memoir = "1988年夏天，我从大学毕业，来到了深圳..."

# 检索
retrieval_result = await retriever.retrieve(memoir)

# 生成
generation_result = await generator.generate(memoir, retrieval_result)

# 评估
evaluation_result = await evaluator.evaluate(
    memoir, generation_result.content, retrieval_result
)

print(generation_result.content)
print(f"综合评分: {evaluation_result.overall_score}/10")
```

---

## 7. 扩展指南

### 7.1 添加新的 LLM 提供商

**步骤 1**：在 `adapter.py` 添加枚举和适配器类

```python
# 1. 添加枚举
class LLMProvider(Enum):
    ...
    NEW_PROVIDER = "new_provider"

# 2. 添加适配器类
class NewProviderAdapter(LLMAdapter):
    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.NEW_PROVIDER
    
    @property
    def default_model(self) -> str:
        return "default-model-name"
    
    def _get_litellm_model_name(self) -> str:
        return f"new_provider/{self._get_model()}"
```

**步骤 2**：在 `factory.py` 注册

```python
# 1. 导入
from .adapter import NewProviderAdapter

# 2. 注册
ADAPTER_REGISTRY[LLMProvider.NEW_PROVIDER] = NewProviderAdapter
DEFAULT_MODELS[LLMProvider.NEW_PROVIDER] = "default-model"

# 3. 添加 API Key 映射
def _get_api_key_from_settings(provider, settings):
    key_mapping = {
        ...
        LLMProvider.NEW_PROVIDER: settings.new_provider_api_key,
    }
```

### 7.2 添加新的写作风格

在 `prompts.py` 添加模板：

```python
class PromptTemplates:
    ...
    DRAMATIC = """你是一位戏剧作家，请用富有戏剧张力的笔法..."""
    
    @classmethod
    def list_styles(cls):
        return {
            ...
            "dramatic": "戏剧风格 - 富有张力的表达",
        }
```

### 7.3 添加新的评估维度

在 `evaluator.py` 添加：

```python
class EvaluationDimension(Enum):
    ...
    CREATIVITY = "creativity"  # 新维度

def _evaluate_creativity_simple(self, generated_text):
    """评估创意性"""
    score = 5.0
    # 检查是否使用了新颖的表达...
    return DimensionScore(EvaluationDimension.CREATIVITY, score, "...")
```

---

## 附录

### A. 关键依赖版本

```
graphrag>=0.3.0
litellm>=1.40.0
pandas>=2.0.0
pydantic>=2.7.0
fastapi>=0.111.0
gradio>=4.36.0
```

### B. 环境变量配置

```bash
# LLM API Keys
DEEPSEEK_API_KEY=xxx
QWEN_API_KEY=xxx
GOOGLE_API_KEY=xxx

# 默认配置
DEFAULT_LLM_PROVIDER=ollama
DEFAULT_LLM_MODEL=qwen3:32b
```

### C. 参考资料

- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG 官方文档](https://microsoft.github.io/graphrag/)
- [GraphRAG 论文](https://arxiv.org/abs/2404.16130)
- [LiteLLM 文档](https://docs.litellm.ai/)

---

**文档版本**: v1.0  
**最后更新**: 2026-02-08  
**作者**: Capstone Project Group2
