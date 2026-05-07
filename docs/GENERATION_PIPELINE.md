# GraphRAG 回忆录扩展生成 Pipeline 完整文档

## 概述

本文档描述回忆录扩展生成的完整流程，从输入的简短回忆录文本到输出的文学化扩展内容，包括分章、检索、生成、评估的全部环节。

---

## 一、Pipeline 总览

```
输入回忆录文本
    ↓
[1] 分章 (Segmentation)
    ↓
[2] 章节预算分配 (Budget Allocation)
    ↓
[3] 逐章处理循环:
    ├─ [3.1] 检索 (Retrieval)
    ├─ [3.2] 新内容提取 (Novel Content Extraction)
    ├─ [3.3] 提示词构建 (Prompt Building)
    ├─ [3.4] 跨章上下文管理 (Cross-Chapter Context)
    └─ [3.5] LLM 生成 (Generation)
    ↓
[4] 评估 (Evaluation)
    ├─ [4.1] 段级指标
    ├─ [4.2] 跨章指标
    └─ [4.3] 质量门控
    ↓
输出: 扩展文本 + 评估报告
```

---

## 二、分章 (Segmentation)

**文件**: `src/generation/memoir_segmenter.py`

### 2.1 分章策略

按优先级从高到低：

1. **显式时间边界**: 段落以新年份开头时强制切分
   - 正则: `^\s*(?:19|20)\d{2}\s*年`
   - 例: "1980年，我考上了大学" → 新章节开始

2. **结构边界**: 空行、章节标题
   - 空行分段: `\n\s*\n+`
   - 标题行: `第[一二三...]章`

3. **长度约束**: 
   - 超长块按句切分 (target_max_chars)
   - 过短块合并 (target_min_chars)
   - **不跨时间边界合并**

### 2.2 核心函数

```python
def segment_memoir(
    text: str,
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> List[MemoirSegment]
```

**流程**:
1. 按空行/标题拆出结构块 (`_structural_blocks`)
2. 超长块按句切分 (`_split_oversized`)
3. 过短块合并，但不跨时间边界 (`_merge_short`)
4. 为每段附加元数据 (`SegmentMeta`)

### 2.3 元数据提取

每个 `MemoirSegment` 包含:
- `index`: 段索引
- `text`: 段落文本
- `meta`: 元数据
  - `detected_years`: 提取的年份列表
  - `detected_locations`: 地名列表
  - `detected_figures`: 人物称谓列表
  - `temporal_label`: 时间范围标签 (如 "1972-1977")
  - `split_reason`: 切分原因 (temporal_boundary/paragraph_break/...)

---

## 三、章节预算分配 (Budget Allocation)

**文件**: `src/generation/chapter_budget.py`

### 3.1 扩展系数

根据 `length_bucket` 决定每章的扩展倍数:

| length_bucket | 扩展系数 | 说明 |
|--------------|---------|------|
| "200-400"    | 0.8×    | 精简 |
| "400-800"    | 1.2×    | 略扩写 |
| "800-1200"   | 1.6×    | 丰富扩写 |
| "1200+"      | 2.0×    | 大幅扩写 |

### 3.2 预算计算

```python
def allocate_segment_budgets(
    segments: List[MemoirSegment],
    length_bucket: str,
) -> List[SegmentBudget]
```

**每章预算**:
- `target_chars = 原文字数 × 扩展系数`
- `length_hint = f"{target*0.85}-{target*1.15}字"`
- `max_tokens = min(8000, target*2.2)`

---

## 四、逐章处理循环

**文件**: `src/generation/long_form_orchestrator.py`

### 4.1 并行优化

采用**预取 (Prefetch)** 策略:
- 在生成第 N 章时，并行检索第 N+1 章
- 减少总耗时

```python
# 预取第一章
next_retrieval = asyncio.create_task(retriever.retrieve(segments[0].text, ...))

for i, seg in enumerate(segments):
    rr = await next_retrieval  # 获取当前章检索结果
    
    # 预取下一章
    if i + 1 < len(segments):
        next_retrieval = asyncio.create_task(retriever.retrieve(segments[i+1].text, ...))
    
    # 生成当前章
    gr = await generator.generate(...)
```

### 4.2 检索 (Retrieval)

**文件**: `src/retrieval/memoir_retriever.py`

每章独立检索:
```python
retrieval_result = await retriever.retrieve(
    segment.text,
    top_k=10,
    use_llm_parsing=True,
    mode="keyword",
)
```

**返回**: `RetrievalResult`
- `entities`: 实体列表
- `relationships`: 关系列表
- `communities`: 社区报告
- `text_units`: 文本片段
- `context`: 回忆录上下文 (年份、地点、关键词)

### 4.3 新内容提取 (Novel Content Extraction)

**文件**: `src/generation/novel_content_extractor.py`

**目的**: 区分 RAG 检索结果中的"对齐内容"和"新知识"

#### 4.3.1 分类逻辑

```python
def extract_novel_content(
    memoir_text: str,
    retrieval_result: RetrievalResult,
) -> NovelContentBrief
```

**实体分类**:
- **对齐实体** (`aligned_entities`): 原文已提及
- **新实体** (`novel_entities`): 原文未提及

**匹配策略**:
1. 精确匹配: 实体名直接出现在原文
2. 模糊匹配:
   - 关键词匹配 (实体名的任何关键词出现在原文)
   - 部分匹配 (实体名前4字符出现在原文)

**关系分类**:
- 如果 source 或 target 在原文中提及 → `aligned_relationships`
- 否则 → `novel_relationships`

#### 4.3.2 输出格式

`NovelContentBrief` 包含:
- `novel_entities`: 新实体列表
- `novel_relationships`: 新关系列表
- `aligned_entities`: 对齐实体列表
- `aligned_relationships`: 对齐关系列表
- `summary`: 一句话概括

**用途**:
1. 注入 prompt (明确标注哪些是新知识)
2. 评估时作为 ground truth

### 4.4 提示词构建 (Prompt Building)

**文件**: `src/generation/prompts.py`, `src/generation/literary_generator.py`

#### 4.4.1 Prompt 结构

```python
def _build_prompt(
    memoir_text: str,
    retrieval_result: RetrievalResult,
    style: str,
    length_hint: str,
    chapter_context: str,
) -> str
```

**Prompt 组成**:
1. **系统提示词** (system_prompt): 定义角色和任务
2. **回忆录原文** (`memoir_text`)
3. **时间地点** (`year`, `location`)
4. **对齐内容** (`aligned_context`): 原文已提及的实体
5. **新知识** (`novel_context`): 原文未提及的实体/关系
6. **跨章上下文** (`chapter_context`): 前文概要 + 反重复要点
7. **长度提示** (`length_hint`)

#### 4.4.2 新知识注入格式

```
可用的新知识（可改写表述，但不可添加未提供的实体或细节）：

1. [工农兵大学生]
   工农兵大学生是通过"自愿报名、群众推荐、领导批准、学校复审"方法招收的学生。

2. [张铁生]
   张铁生在大学招生文化考试中写信，被树立为"反潮流的白卷英雄"。

使用规则：
✓ 可以改写上述内容的表述方式，调整语序，使其自然融入叙事
✓ 可以选择性使用（不必全部使用），选择与叙事最相关的 1-3 条
✗ 不可添加上述列表中未提及的实体、人名、地名、机构名
✗ 不可添加上述内容中未提及的具体数据、政策名称、时间节点
✗ 不可推断上述内容中未提及的因果关系、影响或评价
```

### 4.5 跨章上下文管理 (Cross-Chapter Context)

**文件**: `src/generation/chapter_context.py`

**目的**: 确保叙事衔接、避免内容重复

#### 4.5.1 记录章节

```python
def record_chapter(
    index: int,
    content: str,
    entities: List[str],
) -> None
```

**提取信息**:
- `brief`: 前 1-2 句作为概要 (≤60字)
- `time_period`: 年代范围
- `key_phrases`: 高频实义短语 (用于去重)

**key_phrases 提取**:
- 使用 jieba 分词
- 过滤停用词 (代词、虚词、泛化时间词)
- 统计词级 1-gram + 2-gram
- 取频次 ≥2 的 top 8

#### 4.5.2 构建跨章上下文

```python
def build_prompt_section(current_index: int) -> str
```

**注入内容**:
1. **前文概要**: 最近 3 章的概要
2. **反重复要点**: 前文已出现的 key_phrases
3. **章节位置指令**:
   - 开篇: "自然引入时代背景，不要在末尾写总结"
   - 中段: "与前文自然衔接，禁止在末尾添加感悟"
   - 收尾: "可以带有适度的收束感"

#### 4.5.3 重复检测

```python
def detect_repetition_with_previous(
    new_content: str,
    threshold: float = 0.15,
) -> Optional[str]
```

**策略**: 对比当前章与前文的 key_phrases 重叠率
- 重叠率 ≥ 15% → 返回警告
- 触发重试机制 (temperature +0.1, 强化反重复指令)

### 4.6 LLM 生成

**文件**: `src/generation/literary_generator.py`

```python
async def generate(
    memoir_text: str,
    retrieval_result: RetrievalResult,
    style: str,
    length_hint: str,
    temperature: float,
    max_tokens: int,
    chapter_context: str,
) -> GenerationResult
```

**返回**: `GenerationResult`
- `content`: 生成的文本
- `provider`: LLM 提供商
- `model`: 模型名称
- `novel_content_brief`: 新内容摘要 (用于评估)

---

## 五、评估 (Evaluation)

**文件**: `src/evaluation/long_form_eval.py`

### 5.1 评估架构

```
evaluate_long_form()
    ├─ 段级评估 (并发)
    │   ├─ 指标计算 (metrics)
    │   ├─ LLM-as-Judge (可选)
    │   └─ 事实检查 (可选)
    ├─ 篇级指标
    └─ 质量门控
```

### 5.2 段级指标

**文件**: `src/evaluation/metrics.py`, `src/evaluation/novel_content_metrics.py`

#### 5.2.1 基础指标

| 指标 | 计算方法 | 说明 |
|------|---------|------|
| `entity_coverage` | 使用的新实体数 / 可用新实体数 | 新实体使用率 |
| `temporal_coherence` | 年份一致性检查 | 生成文本年份是否与原文一致 |
| `rag_entity_accuracy` | RAG实体在生成文本中的准确率 | 防止实体幻觉 |
| `topic_coherence` | 关键词重叠率 | 主题一致性 |
| `length_score` | 长度是否在合理范围 | 分段评分 |
| `paragraph_structure` | 段落数量检查 | 结构合理性 |
| `transition_usage` | 过渡词使用率 | 叙事流畅性 |
| `descriptive_richness` | 形容词/副词密度 | 文学性 |

#### 5.2.2 新内容指标

**information_gain** (信息增益):
- 衡量生成文本引入了多少新知识
- 基于使用的新实体数量分段评分:
  - 0个 → 0.0 (无新信息)
  - 1个 → 0.4 (少量新信息)
  - 2个 → 0.7 (适量新信息)
  - 3+个 → 1.0 (丰富新信息)

**expansion_grounding** (扩展溯源率):
- 衡量新内容是否有 RAG 来源支撑
- 计算: 有RAG支撑的新事实 / 总新事实
- 对于 expansion 任务: 使用的新实体均来自 RAG，溯源率 = 100%

**匹配方法**:

```python
def _is_mentioned_in_text(entity_name: str, text: str) -> bool
```

1. **精确匹配**: 归一化后的实体名在文本中
2. **反向匹配**: 文本中的词是实体的一部分
3. **部分匹配**: 实体名前4字符在文本中
4. **缩写匹配**: 任意3字符子串在文本中

### 5.3 跨章指标

**文件**: `src/evaluation/metrics.py` (CrossChapterMetrics)

| 指标 | 计算方法 | 说明 |
|------|---------|------|
| `inter_chapter_repetition` | 相邻章节 6-gram 重叠率 | 检测重复内容 |
| `style_consistency` | 句长方差 + 形容词密度方差 | 风格一致性 |
| `summary_sentence_ratio` | 总结性语句占比 | 避免过度总结 |

### 5.4 质量门控 (Quality Gate)

**文件**: `src/evaluation/quality_gate.py`

#### 5.4.1 阈值配置

```python
@dataclass
class QualityThresholds:
    min_segment_score: float = 5.0        # 段级综合分下限
    max_cross_repetition: float = 0.20    # 跨章重叠率上限
    min_fact_score: float = 0.60          # FActScore 最低比率
    min_length_ratio: float = 0.40        # 字数比率下限
    max_length_ratio: float = 2.5         # 字数比率上限
    max_summary_sentence_ratio: float = 0.30  # 总结句占比上限
    min_expansion_grounding: float = 0.40  # 扩展溯源率下限
    min_entity_coverage: float = 0.80     # 实体覆盖率下限
```

#### 5.4.2 检查维度

**单章检查**:
1. 字数检查: 实际字数 / 目标字数
2. 综合分检查: 是否低于阈值
3. 事实检查: FActScore 是否达标
4. 总结性语句检查: 占比是否过高
5. 套话检查: 是否有空泛过渡语
6. 非末章感悟结尾检查: 是否有抒情式收尾

**跨章检查**:
- 相邻章节 6-gram 重叠率

#### 5.4.3 修复建议

```python
@dataclass
class RemediationPlan:
    chapters_to_regenerate: List[int]      # 需重新生成的章节
    reasons: Dict[int, List[str]]          # 失败原因
    prompt_adjustments: Dict[int, str]     # Prompt 调整建议
```

**示例**:
```
第2章需重新生成:
- 原因: 跨章 6-gram 重叠率 25% (阈值 20%)
- 建议: 在后一章 prompt 中注入前一章概要并要求不重复
```

---

## 六、输出

### 6.1 生成结果

```python
@dataclass
class LongFormGenerationResult:
    chapters: List[ChapterGenerationResult]  # 每章的详细结果
    merged_content: str                      # 合并后的完整文本
    full_memoir_text: str                    # 原始回忆录
    segments: List[MemoirSegment]            # 分段信息
    segmentation_report: SegmentationReport  # 分段质量报告
    chapter_context: ChapterContext          # 跨章上下文
```

### 6.2 评估结果

```python
@dataclass
class LongFormEvalResult:
    segments: List[SegmentEvalRecord]        # 每章评估记录
    document_metrics: Dict[str, MetricResult]  # 篇级指标
    cross_chapter_metrics: Dict[str, MetricResult]  # 跨章指标
    aggregated_score: float                  # 加权综合分
    quality_gate: QualityGateResult          # 质量门控结果
```

---

## 七、关键设计决策

### 7.1 为什么每章独立检索？

- **优点**: 每章获取最相关的背景知识
- **缺点**: 可能有重复实体
- **解决**: 通过跨章上下文管理避免重复叙述

### 7.2 为什么区分"对齐内容"和"新知识"？

- **对齐内容**: 原文已提及，用于增强叙事氛围
- **新知识**: 原文未提及，用于扩展信息量
- **评估**: 只统计新知识的使用情况 (information_gain)

### 7.3 为什么不跨时间边界合并段落？

- 保持时间线清晰
- 避免混淆不同年代的事件
- 便于检索到时间相关的背景知识

### 7.4 为什么使用分段评分而非线性评分？

**information_gain 示例**:
- 线性评分: 1个实体/10个可用 = 10% (过低)
- 分段评分: 1个实体 = 40% (合理)
- **原因**: 不相关实体不应拉低分数

---

## 八、使用示例

```python
from src.llm import create_llm_adapter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator
from src.evaluation import evaluate_long_form

# 1. 初始化
llm_adapter = create_llm_adapter(provider="openai", model="gpt-4o")
retriever = MemoirRetriever(index_dir="data/graphrag_output", llm_adapter=llm_adapter)
generator = LiteraryGenerator(llm_adapter=llm_adapter)

# 2. 生成
result = await generator.generate_long_form(
    memoir_text="1980年，我考上了大学...",
    retriever=retriever,
    target_min_chars=300,
    target_max_chars=800,
    use_llm_parsing=True,
    retrieval_mode="keyword",
    style="standard",
    temperature=0.7,
)

# 3. 评估
eval_result = await evaluate_long_form(
    result,
    llm_adapter=llm_adapter,
    use_llm_eval=False,
    enable_fact_check=False,
    enable_quality_gate=True,
)

# 4. 输出
print(f"生成章节数: {len(result.chapters)}")
print(f"综合分数: {eval_result.aggregated_score:.2f}/10")
print(f"质量门控: {'✓ 通过' if eval_result.quality_gate.passed else '✗ 未通过'}")
```

---

## 九、性能优化

1. **并行检索**: 预取下一章检索结果
2. **缓存**: LLM prompt 缓存 (5分钟 TTL)
3. **批量评估**: 所有章节并发评估
4. **超时保护**: 事实检查带超时 (60s/章)

---

## 十、文件清单

| 文件 | 功能 |
|------|------|
| `memoir_segmenter.py` | 回忆录分章 |
| `chapter_budget.py` | 章节预算分配 |
| `long_form_orchestrator.py` | 长文编排主流程 |
| `novel_content_extractor.py` | 新内容提取与分类 |
| `prompts.py` | 提示词模板管理 |
| `chapter_context.py` | 跨章上下文管理 |
| `literary_generator.py` | LLM 生成器 |
| `metrics.py` | 基础评估指标 |
| `novel_content_metrics.py` | 新内容评估指标 |
| `quality_gate.py` | 质量门控 |
| `long_form_eval.py` | 长文评估聚合 |

---

**文档版本**: v1.0  
**最后更新**: 2026-05-07
