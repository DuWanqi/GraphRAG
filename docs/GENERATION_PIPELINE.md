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
2. **结构边界**: 空行、章节标题
3. **长度约束**: 超长块按句切分，过短块合并（不跨时间边界）

### 2.2 核心函数

```python
def segment_memoir(
    text: str,
    target_min_chars: int = 300,
    target_max_chars: int = 800,
) -> List[MemoirSegment]
```

---

## 三、章节预算分配

**文件**: `src/generation/chapter_budget.py`

### 3.1 扩展系数

| length_bucket | 扩展系数 | 说明 |
|--------------|---------|------|
| "200-400"    | 0.8×    | 精简 |
| "400-800"    | 1.2×    | 略扩写 |
| "800-1200"   | 1.6×    | 丰富扩写 |
| "1200+"      | 2.0×    | 大幅扩写 |

---

## 四、逐章处理循环

**文件**: `src/generation/long_form_orchestrator.py`

### 4.1 并行优化

采用预取策略：在生成第 N 章时，并行检索第 N+1 章。

### 4.2 检索

每章独立检索，返回 `RetrievalResult` (entities, relationships, communities)。

### 4.3 新内容提取

**文件**: `src/generation/novel_content_extractor.py`

**目的**: 区分 RAG 检索结果中的"对齐内容"和"新知识"

**分类逻辑**:
- **对齐实体**: 原文已提及
- **新实体**: 原文未提及

**匹配策略**:
1. 精确匹配
2. 关键词匹配
3. 部分匹配（前4字符）

### 4.4 提示词构建

**Prompt 组成**:
1. 系统提示词
2. 回忆录原文
3. 时间地点
4. **对齐内容**: 原文已提及的实体
5. **新知识**: 原文未提及的实体（白名单格式）
6. 跨章上下文
7. 长度提示

**新知识注入格式**:
```
可用的新知识（可改写表述，但不可添加未提供的实体或细节）：

1. [工农兵大学生]
   工农兵大学生是通过"自愿报名、群众推荐、领导批准、学校复审"方法招收的学生。

使用规则：
✓ 可以改写表述、调整语序
✓ 可以选择性使用（1-3条）
✗ 不可添加未提供的实体
✗ 不可添加未提及的数据、政策名称
```

### 4.5 跨章上下文管理

**文件**: `src/generation/chapter_context.py`

**记录信息**:
- `brief`: 前 1-2 句概要
- `time_period`: 年代范围
- `key_phrases`: 高频实义短语（用于去重）

**注入内容**:
1. 前文概要（最近3章）
2. 反重复要点
3. 章节位置指令

---

## 五、评估 (Evaluation)

**文件**: `src/evaluation/long_form_eval.py`

### 5.1 段级指标

| 指标 | 说明 | 计算方法 |
|------|------|---------|
| **entity_coverage** | 新实体使用率 | 使用的新实体数 / 可用新实体数 |
| **temporal_coherence** | 时间一致性 | 年份匹配检查 |
| **topic_coherence** | 主题一致性 | 关键词语义匹配 |
| **information_gain** | 信息增益 | 0个→0.0, 1个→0.4, 2个→0.7, 3+→1.0 |
| **expansion_grounding** | 扩展溯源率 | 有RAG支撑的新事实 / 总新事实 |
| **length_score** | 长度合理性 | 是否在目标区间 |
| **paragraph_structure** | 段落结构 | 段落数量检查 |
| **transition_usage** | 过渡词使用 | 过渡词出现率 |
| **descriptive_richness** | 描写丰富度 | 形容词/副词比例 |

### 5.2 新内容评估

**新事实提取** (三层漏斗):

```
[第一层] 词性粗筛 (jieba)
    ↓ 保留 nr/ns/nt/nz 词性
[第二层] NER 验证 (LAC)
    ↓ 确认 PER/LOC/ORG/TIME
[第三层] 句式补充 (正则)
    ↓ 年份+事件、实体+关系+实体
最终新事实列表
```

**匹配策略** (4种):
1. 实体名精确匹配
2. 反向包含匹配
3. 模糊 n-gram 匹配 (≥60% 重叠)
4. 词级匹配 (≥50% 词匹配)

### 5.3 质量门控

**文件**: `src/evaluation/quality_gate.py`

**阈值配置** (expansion任务):
```python
min_segment_score = 5.0
max_cross_repetition = 0.20
min_expansion_grounding = 0.40
min_entity_coverage = 0.80
```

**检查维度**:
- 长度合理性
- 跨章重复度
- 扩展溯源率
- 实体覆盖率

**输出**:
- `passed`: 是否通过
- `issues`: 问题列表
- `remediation`: 修复建议（需重新生成的章节）

---

## 六、使用示例

```python
from src.llm import create_llm_adapter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator
from src.evaluation import evaluate_long_form

# 初始化
llm_adapter = create_llm_adapter(provider="openai", model="gpt-4o")
retriever = MemoirRetriever(index_dir="data/graphrag_output", llm_adapter=llm_adapter)
generator = LiteraryGenerator(llm_adapter=llm_adapter)

# 生成
result = await generator.generate_long_form(
    memoir_text=memoir_text,
    retriever=retriever,
    target_min_chars=300,
    target_max_chars=800,
    length_bucket="400-800",
)

# 评估
eval_result = await evaluate_long_form(
    result,
    llm_adapter=llm_adapter,
    enable_fact_check=False,
    enable_quality_gate=True,
)
```

---

## 七、文件清单

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

## 八、Development Process (演进历史)

### 8.1 Phase 1: Aligned Paraphrase → Novel Content Expansion

**问题**: 旧版生成仅对原文润色，未引入RAG检索到的新知识。

**改进前**:
```
Prompt: "## 可参考的历史背景信息\n{context}"
结果: "1980年，我收到了大学录取通知书。那是一个充满希望的年代..."
问题: 未引入"恢复高考"、"改革开放"等RAG实体
```

**改进后**:
```
Prompt: 
"## 对齐内容（原文已提及）
- 大学: ...

## 新知识（原文未提及，必须引入1-3条）
1. [恢复高考] 1977年恢复高考制度
2. [改革开放] 1978年十一届三中全会..."

结果: "距离恢复高考已经三年，整个国家都在邓小平推动的改革开放浪潮中..."
```

**关键改进**:
1. 拆分"对齐内容"和"新知识"
2. 明确要求引入新知识
3. 白名单约束（不可添加未提供的实体）

---

### 8.2 Phase 2: 评估指标重新设计

**问题**: 11个段级指标存在严重重复，概念混乱。

**重复问题**:

| 旧指标 | 问题 | 解决方案 |
|--------|------|---------|
| `novel_content_ratio` + `expansion_depth` | 都基于"使用的新实体数量"分段评分 | 合并为 `information_gain` |
| `keyword_overlap` + `semantic_similarity` | 都衡量"生成与原文相似度" | 合并为 `topic_coherence` |
| `novel_content_grounding` + `FActScore` | 都衡量"准确性/溯源性" | 保留 `expansion_grounding`，删除 `FActScore` |
| `paragraph_structure` + `transition_usage` + `descriptive_richness` | 都衡量"文学性" | 合并为 `narrative_quality` |
| `time_consistency` | 逻辑错误（要求生成文本包含参考年份） | 重新设计为 `temporal_coherence` |

**改进前** (11个指标):
```
entity_coverage, time_consistency, keyword_overlap, semantic_similarity,
length_score, paragraph_structure, transition_usage, descriptive_richness,
novel_content_ratio, novel_content_grounding, expansion_depth
```

**改进后** (7个指标):
```
entity_coverage, temporal_coherence, topic_coherence, information_gain,
expansion_grounding, length_score, narrative_quality
```

**关键改进**:
1. 删除重复指标
2. 修复逻辑错误
3. 统一命名规范

---

### 8.3 Phase 3: 新事实提取优化

**问题**: 简单分词提取大量噪音词（"刚刚"、"开始"、"年代"）。

**改进前** (单层分词):
```python
words = jieba.cut(text)
facts = [w for w in words if len(w) >= 2]
# 结果: ['改革开放', '刚刚', '开始', '年代', '录取', '通知书', '整个', '国家']
# 准确率: ~40%
```

**改进后** (三层漏斗):
```python
# 第一层: 词性粗筛 (jieba)
candidates = [w for w, pos in jieba.posseg.cut(text) if pos in ['nr','ns','nt','nz']]

# 第二层: NER 验证 (LAC)
verified = [c for c in candidates if lac.run(c) in ['PER','LOC','ORG','TIME']]

# 第三层: 句式补充 (正则)
compound = re.findall(r'(\d{4})年([^。]{2,15})', text)

# 结果: ['改革开放', '十一届三中全会', '邓小平', '1978']
# 准确率: ~85%
```

**性能对比**:

| 方案 | 准确率 | 速度 | 依赖 |
|------|--------|------|------|
| 单层分词 | 40% | 10ms | jieba |
| 三层完整 | 85% | 65ms | jieba + LAC |
| 三层降级 | 75% | 15ms | jieba |

---

### 8.4 Phase 4: entity_coverage 指标重新解读

**问题**: entity_coverage 低（0-20%）被误判为"RAG未生效"。

**根本原因**:
- RAG检索的是**宏观历史实体**（恢复高考、邓小平、十一届三中全会）
- 生成文本保留**个人叙事实体**（老张、德明、窑洞）
- RAG实体以**背景方式融入**，而非直接罗列

**示例**:
```
RAG实体: ["恢复高考", "邓小平", "十一届三中全会", "知青返城", "高等教育改革"]
生成文本: "广播中传来恢复高考的消息，整个生产队的知青如同被点燃的火药..."
字符串匹配: 只有"恢复高考"匹配
entity_coverage = 1/5 = 20%
```

**错误解读**: 20% → RAG没用上 → 生成质量差

**正确解读**: 
- 核心历史事件（"恢复高考"）被提及 ✓
- 原文细节（"生产队"、"知青"）保留 ✓
- RAG以背景方式有效融入 ✓

**改进方案**:
1. 降低 entity_coverage 权重（0.1 → 0.05）
2. 新增 `rag_used_in_text` 和 `text_only_entities` 字段
3. 综合判断而非只看数值

---

### 8.5 Phase 5: 白名单约束机制

**问题**: 模型添加RAG未提供的实体（幻觉）。

**改进前**:
```
Prompt: "将上面提供的历史背景信息自然地编织进叙事中"
结果: "包产到户政策让农民有了自主权，父亲说邓小平是个伟人..."
       ^^^^^^^^ 未提供              ^^^ 未提供
```

**改进后**:
```
Prompt: 
"使用规则：
✓ 可以改写表述、调整语序
✓ 可以选择性使用（1-3条）
✗ 不可添加未提供的实体
✗ 不可添加未提及的数据、政策名称"

结果: "距离恢复高考已经三年，整个国家都在改革开放浪潮中..."
      ^^^^^^^^ RAG提供        ^^^^^^^^ RAG提供
```

**验证机制**:
```python
grounding = expansion_grounding_metric(memoir_text, generated_text, novel_brief)
# grounding.value: 0.0-1.0
# 如果添加未提供的实体，分数降低
```

---

### 8.6 Phase 6: 质量门控与自动修复

**问题**: 生成结果无质量保障，需人工检查。

**改进前**: 只有评分，无判定标准。

**改进后**: 
1. **阈值配置**（针对expansion任务）
2. **多维度检查**（长度、重复、溯源、覆盖）
3. **修复建议**（需重新生成的章节 + prompt调整）

**示例**:
```python
gate_result = check_quality_gate(eval_result, thresholds)

if not gate_result.passed:
    print(f"未通过: {gate_result.issues}")
    print(f"需重新生成: 第{gate_result.remediation.chapters_to_regenerate}章")
    print(f"调整建议: {gate_result.remediation.prompt_adjustments}")
```

---

### 8.7 关键设计决策

| 决策 | 原因 |
|------|------|
| **每章独立检索** | 避免主题漂移，提高相关性 |
| **预取优化** | 减少总耗时（检索与生成并行） |
| **白名单约束** | 防止幻觉，确保溯源 |
| **三层新事实提取** | 平衡准确率与速度 |
| **跨章上下文管理** | 避免重复，保持衔接 |
| **质量门控** | 自动化质量保障 |
| **扩展系数而非固定总量** | 适应不同长度的原文 |

---

**文档版本**: v2.0  
**最后更新**: 2026-05-07
