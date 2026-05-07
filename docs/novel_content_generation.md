# Novel Content Generation - 新内容生成

## 概述

本文档说明生成模块在"新内容生成"改进后的行为变化，以及如何评估新内容的引入情况。

## 问题背景

**改进前（Aligned Paraphrase）**：生成模块主要对回忆录原文进行文学润色，RAG 检索到的历史背景信息仅作为氛围调料使用，模型很少将检索到的**输入中不存在的新事实**实质性地编织进叙事。

**改进目标**：生成内容应包含"substantively supported by retrieved non-input content"的扩展，例如用户仅输入"80年代经商"，生成时应结合改革开放背景，补充相关的经商政策、事件等 RAG 检索到的新知识。

## 行为对比

### Before: Aligned Paraphrase（对齐式改写）

**输入示例**：
```
1980年，我考上了大学。
```

**RAG 检索到的内容**：
- 实体：恢复高考、邓小平、改革开放、经济特区
- 关系：恢复高考 → 1977年恢复、邓小平 → 推动改革开放
- 背景：1977年恢复高考制度，1980年代初期改革开放政策全面展开

**旧版生成结果**（仅润色，未引入新知识）：
```
一九八零年，我收到了大学录取通知书。那是一个充满希望的年代，
整个国家都在发生着变化。父亲激动得一夜没睡，母亲在一旁默默流泪。
我知道，这张通知书不仅是我的，也是全家人的梦想。
```

**问题**：生成内容虽然文学性强，但没有引入任何 RAG 检索到的新知识（如"恢复高考"、"改革开放"等），仅对原文进行了情感化扩写。

### After: Novel Content Expansion（新内容扩展）

**输入示例**：
```
1980年，我考上了大学。
```

**RAG 检索到的内容**：
- 实体：恢复高考、邓小平、改革开放、经济特区
- 关系：恢复高考 → 1977年恢复、邓小平 → 推动改革开放
- 背景：1977年恢复高考制度，1980年代初期改革开放政策全面展开

**新版生成结果**（引入新知识）：
```
一九八零年，我收到了大学录取通知书。距离恢复高考已经三年，
整个国家都在邓小平推动的改革开放浪潮中焕发生机。父亲激动得一夜没睡，
他说："你赶上了好时候，国家需要人才。"母亲在一旁默默流泪。
我知道，这张通知书不仅是我的，也是全家人的梦想，更是这个时代给予的机会。
```

**改进点**：
1. 引入了"恢复高考"这一 RAG 检索到的新实体
2. 引入了"邓小平推动的改革开放"这一 RAG 检索到的新关系
3. 新知识自然地融入叙事（通过父亲的对话和叙事者的感悟）
4. 所有新内容都有 RAG 来源支撑，没有幻觉

## Prompt 差异对比

### 旧版 Prompt 结构

```
## 可参考的历史背景信息
{context}

写作要求：
4. 将上面提供的历史背景信息自然地编织进叙事中，不要单独罗列历史知识
```

**问题**：
- "可参考"措辞太弱，模型倾向于忽略
- 没有区分哪些是新知识，哪些是对齐内容
- 没有明确要求引入新知识

### 新版 Prompt 结构（改进版）

```
## 与原文相关的背景（用于增强叙事氛围）
{aligned_context}

## 原文未提及的新知识（请选择性融入叙事）
可用的新知识（可改写表述，但不可添加未提供的实体或细节）：

1. [改革开放]
   1978年十一届三中全会确立了改革开放政策
2. [深圳特区]
   深圳是中国最早设立的经济特区之一

使用规则：
✓ 可以改写上述内容的表述方式，调整语序，使其自然融入叙事
✓ 可以选择性使用（不必全部使用），选择与叙事最相关的 1-3 条
✗ 不可添加上述列表中未提及的实体、人名、地名、机构名
✗ 不可添加上述内容中未提及的具体数据、政策名称、时间节点
✗ 不可推断上述内容中未提及的因果关系、影响或评价

写作要求：
4. **重点：从上面的「原文未提及的新知识」中选取 1-3 个与叙事最相关的事实，
   自然地编织进故事中**——例如作为人物对话的内容、环境描写的细节、或叙事者的见闻
5. 融入新知识时，必须基于上面提供的检索信息，不要自行编造未提供的历史事实
```

**改进点**：
- 明确区分"对齐内容"和"新知识"
- **白名单式列举**：用编号（1, 2, 3...）明确列出可用的知识点
- **允许改写**：不要求逐字使用，允许调整表述使其自然融入叙事
- **明确边界**：用 ✓/✗ 清晰标注什么可以做、什么不能做
- **实体约束**：核心是"不可添加未提供的实体"，防止模型自由发挥
- 明确要求选取 1-3 个新事实并融入
- 提供融入方式的具体建议（对话、环境描写、见闻）

## 评估指标

### 1. novel_content_ratio - 新内容引入率

**定义**：生成文本使用了多少 RAG 提供的新知识

**计算方式**：
```
novel_content_ratio = 使用的新实体数量 / 可用新实体数量
```

**评分标准**：
- 0.0 - 0.2：未引入新知识（仅润色）
- 0.2 - 0.5：引入了部分新知识
- 0.5 - 1.0：充分利用了新知识

**示例**：
- 可用新实体：["恢复高考", "邓小平", "改革开放", "经济特区"]（4个）
- 生成文本中使用：["恢复高考", "邓小平"]（2个）
- novel_content_ratio = 2 / 4 = 0.5

### 2. novel_content_grounding - 新内容溯源率

**定义**：生成文本中的新事实是否有 RAG 来源支撑（防幻觉）

**计算方式**：
```
novel_content_grounding = 有RAG来源支撑的新事实数量 / 总新事实数量
```

**评分标准**：
- 0.8 - 1.0：新内容完全有据可查
- 0.5 - 0.8：大部分新内容有据，少量疑似幻觉
- 0.0 - 0.5：大量幻觉，不可接受

**示例**：
- 生成文本中的新事实：["恢复高考", "邓小平", "深圳特区", "华强北"]（4个）
- 有 RAG 来源支撑：["恢复高考", "邓小平", "深圳特区"]（3个）
- 无 RAG 来源支撑（疑似幻觉）：["华强北"]（1个）
- novel_content_grounding = 3 / 4 = 0.75

### 3. expansion_depth - 扩展深度

**定义**：生成文本相对于输入的信息增量

**分类**：
- **shallow（浅层）**：仅润色，未引入新知识
- **moderate（中等）**：引入 1-2 个新事实
- **deep（深度）**：引入 3+ 个新事实并有叙事整合

**评分**：
- shallow: 0.3
- moderate: 0.7
- deep: 1.0

## 安全边界

### 什么算合理扩展？

✅ **合理扩展**：
- 引入 RAG 检索到的实体、事件、关系
- 新内容与原文的时间、地点、主题相关
- 新内容自然融入叙事，不喧宾夺主
- 所有新内容都有 RAG 来源支撑

### 什么算幻觉？

❌ **幻觉**：
- 编造 RAG 中不存在的实体、事件
- 编造具体的人名、地名、数字（RAG 中没有）
- 编造对话内容（RAG 中没有相关信息）
- 编造因果关系（RAG 中没有相关关系）

### 示例对比

**输入**：
```
1980年，我在深圳经商。
```

**RAG 检索到的内容**：
- 实体：深圳经济特区、改革开放、邓小平
- 关系：深圳经济特区 → 1980年设立

**✅ 合理扩展**：
```
一九八零年，我在深圳经商。那一年，深圳被设立为经济特区，
整个城市都在改革开放的浪潮中沸腾。
```
→ 引入了"深圳经济特区"和"1980年设立"，有 RAG 来源支撑

**❌ 幻觉**：
```
一九八零年，我在深圳经商。那一年，我在华强北开了一家电子元件店，
第一个月就赚了三千块。
```
→ "华强北"、"电子元件店"、"三千块"都是编造的，RAG 中没有

## 使用方式

### 1. 生成时自动启用

新内容提取和分类在生成时自动进行，无需额外配置：

```python
from src.generation import LiteraryGenerator
from src.retrieval import MemoirRetriever

retriever = MemoirRetriever(index_dir="data/graphrag_output")
generator = LiteraryGenerator(llm_adapter=llm_adapter)

# 检索
retrieval_result = await retriever.retrieve(memoir_text, top_k=10)

# 生成（自动使用新内容提取）
generation_result = await generator.generate(
    memoir_text=memoir_text,
    retrieval_result=retrieval_result,
    style="standard",
    length_hint="300-500字",
)

# 查看新内容信息
if generation_result.novel_content_brief:
    print(f"可用新知识: {generation_result.novel_content_brief.summary}")
```

### 2. 白名单约束机制（改进版）

#### 工作原理

生成器会自动将 RAG 检索结果格式化为白名单式的 prompt：

```python
# 自动分类
novel_brief = extract_novel_content(memoir_text, retrieval_result)

# 格式化为白名单
formatted = novel_brief.format_for_prompt()
# 返回：
# {
#   "aligned_context": "相关实体：\n- 高考: 1977年恢复高考制度...",
#   "novel_context": "可用的新知识（可改写表述，但不可添加未提供的实体或细节）：\n1. [改革开放]\n   1978年十一届三中全会确立了改革开放政策\n..."
# }
```

#### 约束规则

模型收到的约束规则：

**✓ 允许的操作**：
- 改写表述方式（如"1978年确立改革开放政策" → "两年前确立了新政策"）
- 调整语序使其自然融入叙事
- 选择性使用（不必全部使用，选择与叙事最相关的 1-3 条）

**✗ 禁止的操作**：
- 添加列表中未提及的实体、人名、地名、机构名
- 添加内容中未提及的具体数据、政策名称、时间节点
- 推断内容中未提及的因果关系、影响或评价

#### 实际示例

**输入**：
```
1980年，我考上了大学。
```

**RAG 检索到**：
```
entities = [
    {'title': '改革开放', 'description': '1978年十一届三中全会确立了改革开放政策'},
    {'title': '深圳特区', 'description': '深圳是中国最早设立的经济特区之一'},
]
```

**生成的白名单 Prompt**：
```
可用的新知识（可改写表述，但不可添加未提供的实体或细节）：

1. [改革开放]
   1978年十一届三中全会确立了改革开放政策
2. [深圳特区]
   深圳是中国最早设立的经济特区之一

使用规则：
✓ 可以改写上述内容的表述方式，调整语序，使其自然融入叙事
✓ 可以选择性使用（不必全部使用），选择与叙事最相关的 1-3 条
✗ 不可添加上述列表中未提及的实体、人名、地名、机构名
✗ 不可添加上述内容中未提及的具体数据、政策名称、时间节点
✗ 不可推断上述内容中未提及的因果关系、影响或评价
```

**✅ 合规的生成结果**：
```
1980年，我收到了大学录取通知书。那是改革开放刚刚开始的年代，
两年前的十一届三中全会确立了新政策，整个国家都在发生变化。
```
→ 改写了表述，但没有添加未提供的实体

**❌ 违规的生成结果**：
```
1980年，我收到了大学录取通知书。那是改革开放后的第二年，
包产到户政策让农民有了自主权，父亲说邓小平是个伟人...
         ^^^^^^^^                    ^^^
         未提供的政策                未提供的人名
```
→ 添加了 RAG 未提供的"包产到户"和"邓小平"

#### 验证机制

`novel_content_grounding_metric` 会检查生成文本中的新内容是否都有 RAG 支持：

```python
# 自动检查
grounding = novel_content_grounding_metric(memoir_text, generated_text, novel_brief)

# 如果模型添加了未提供的实体，grounding 分数会降低
# grounding.value: 0.0-1.0（1.0 表示所有新内容都有依据）
# grounding.explanation: "新事实 5 个，有依据 3 个，无依据 2 个"
```

### 3. 评估新内容指标

```python
from src.evaluation import (
    novel_content_ratio_metric,
    novel_content_grounding_metric,
    expansion_depth_metric,
)

# 获取 novel_content_brief
novel_brief = generation_result.novel_content_brief

# 计算指标
ratio = novel_content_ratio_metric(memoir_text, generated_text, novel_brief)
grounding = novel_content_grounding_metric(memoir_text, generated_text, novel_brief)
depth = expansion_depth_metric(memoir_text, generated_text, novel_brief)

print(f"新内容引入率: {ratio.value:.0%} - {ratio.explanation}")
print(f"新内容溯源率: {grounding.value:.0%} - {grounding.explanation}")
print(f"扩展深度: {depth.explanation}")
```

### 3. 长文评估自动集成

长文评估（`evaluate_long_form`）自动包含新内容指标：

```python
from src.evaluation import evaluate_long_form

eval_result = await evaluate_long_form(
    long_form_result,
    llm_adapter=llm_adapter,
    enable_fact_check=True,
)

# 查看每章的新内容评估
for segment in eval_result.segments:
    if segment.novel_content_info:
        print(f"第{segment.segment_index}章:")
        print(f"  新内容引入率: {segment.novel_content_info['novel_content_ratio']:.0%}")
        print(f"  新内容溯源率: {segment.novel_content_info['novel_content_grounding']:.0%}")
```

## 测试验证

运行新内容生成测试：

```bash
# 使用 pytest
conda run -n RAG pytest generation_test/test_novel_content.py -v -s

# 或直接运行
conda run -n RAG python generation_test/test_novel_content.py
```

测试用例：
1. **test_novel_content_extraction** - 验证新内容提取功能
2. **test_novel_content_generation** - 验证生成文本包含新知识
3. **test_novel_content_metrics** - 验证评估指标计算正确

## 常见问题

### Q1: 为什么有时候 novel_content_ratio 为 0？

**A**: 可能的原因：
1. RAG 检索结果质量不高，没有检索到相关的新知识
2. 检索到的实体都在原文中已经提及（没有新实体）
3. LLM 没有遵循 prompt 指令，未引入新知识

**解决方案**：
- 检查 RAG 检索结果，确认有可用的新知识
- 调整检索参数（top_k, retrieval_mode）
- 调整生成参数（temperature, style）

### Q2: 为什么 novel_content_grounding 低于 0.5？

**A**: 说明生成文本中有大量幻觉（无 RAG 来源支撑的新内容）。

**解决方案**：
- 检查 prompt 是否明确禁止编造事实
- 降低 temperature（减少随机性）
- 使用更可靠的 LLM 模型

### Q3: 如何平衡"新内容引入"和"保留原文细节"？

**A**: 这是一个权衡问题：
- 新内容应该是"补充"而非"替代"原文
- prompt 中明确要求"保留原文所有人物、事件、对话、细节"
- 新内容应该自然融入，不喧宾夺主
- 建议引入 1-3 个新事实，而非大量堆砌

## 技术实现

### 核心模块

1. **novel_content_extractor.py** - 新内容提取器
   - `extract_novel_content()` - 从 RAG 结果中提取并分类新内容
   - `NovelContentBrief` - 新内容摘要数据结构

2. **novel_content_metrics.py** - 新内容评估指标
   - `novel_content_ratio_metric()` - 新内容引入率
   - `novel_content_grounding_metric()` - 新内容溯源率
   - `expansion_depth_metric()` - 扩展深度

3. **prompts.json** - Prompt 模板
   - 拆分为"对齐内容"和"新知识"两个区块
   - 明确要求引入新知识

### 数据流

```
memoir_text + RAG retrieval
    ↓
extract_novel_content()
    ↓
NovelContentBrief (aligned vs novel)
    ↓
format_for_prompt()
    ↓
LLM generation (with novel content instructions)
    ↓
generated_text
    ↓
novel_content_metrics (evaluation)
```

## 参考资料

- [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251)
- [SAFE: Semantic Alignment for Factual Evaluation](https://arxiv.org/abs/2403.04445)
- GraphRAG Documentation: https://microsoft.github.io/graphrag/
