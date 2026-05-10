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

**实现注意**: `MemoirRetriever`（`src/retrieval/memoir_retriever.py`）内部使用 `pathlib.Path` 拼接索引子路径（如 `index_dir / "output"`）。调用方若传入字符串 `index_dir="data/graphrag_output"`，构造函数须 **`Path(index_dir)` 规范化**；否则在 Python 中对 `str` 使用 `/` 会触发 `TypeError: unsupported operand type(s) for /: 'str' and 'str'`。

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

**注入时的实体 / 关系描述截断**（`novel_content_extractor.NovelContentBrief.format_for_prompt`）:

- **目的**: GraphRAG 实体描述常为长篇英文；若整段写入 user prompt，会急剧拉长上下文，弱化文末「严格使用规则」，模型更易脱离白名单编造内容。
- **实现**: 模块级函数 `_truncate_for_prompt(text, max_chars=200)`——在约 200 字窗口内优先在 `。`、`！`、`？`、英文句号段落的 `. `、`.\n`、`\n\n`、`\n` 等处断开；若无合适断点则硬截断并追加 `…`。关系的 `description` 字段同样使用该逻辑。
- **易错点（已修复）**: 旧实现仅用 `split("。")` 取首句；英文描述不含中文句号时 **截断完全不生效**，实测可出现单实体描述 **5000+ 字符**、`novel_context` **约 2 万字符**、整段 prompt **约 2.1 万字符**。
- **最少用量**: `prompts.json` 各主风格模板（`standard` / `nostalgic` / `narrative`）及 `system_prompts.default` 要求从白名单选取 **2–4 个实体或事实，至少 2 个**；`format_for_prompt` 文末「严格使用规则」与之对齐（原为「1–3 个」时下界为 1，模型易只嵌入一条以满足约束）。
- **白名单中文展示与正文语言**: 图谱实体 `name` 常为英文全大写（如 `DENG XIAOPING`、`SHENZHEN CITY`）。若在 prompt 顶部「✓ 允许使用的实体名称」中原样罗列，模型倾向**照抄英文专名**嵌入中文正文。
  - **实现**: `_localize_entity_name(name, description)`——英文名且描述前约 300 字中含括号中文、`also known as` 后中文或行首中文标题时，用于 **prompt 内展示名**及关系 `source`/`target` 展示；列表文案为「允许使用的实体名称（中文）」。文末规则补充：**正文须纯中文，禁止出现英文图谱标题式专名**。
  - **评估侧**：仍可对原始英文名做对齐（见 §8.8）；生成侧展示与叙事语言与此独立。
- **附加规则（评估口径）**: 若正文使用中文译名而白名单列为英文，评估层通过映射与 description 全文溯源对齐（**§5.2** 各指标公式、**§5.8** 溯源语义、**§8.8** 演进记录）。

**新知识注入格式**（示意）:
```
可用的新知识（可改写表述，但不可添加未提供的实体或细节）：

1. [工农兵大学生]
   工农兵大学生是通过"自愿报名、群众推荐、领导批准、学校复审"方法招收的学生。

使用规则：
✓ 可以改写表述、调整语序
✓ 建议选择 2-4 条（至少 2 条），与模板一致
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

**主入口**：`evaluate_long_form`（`src/evaluation/long_form_eval.py`）；段级分项实现见 **`metrics.py`**、**`novel_content_metrics.py`**；门控 **`quality_gate.py`**；可选事实检查 **`factscore_adapter.py`**。

下文「段级指标」若在 `novel_content_brief` 上计算，隐含 **已从检索结果挂载 `NovelContentBrief`**、`reference_entities` 一般为 **`novel_entity_names` 前十条**（见 `evaluate_long_form`）。

### 5.1 指标体系总览（简表）

| 指标键名 | 值域（max） | 作用层级 |
|-----------|-------------|----------|
| `entity_coverage` | 0～1（1） | 段 |
| `temporal_coherence` | 0～1（1） | 段 |
| `rag_entity_accuracy` | 0～1（1） | 段 |
| `topic_coherence` | 0～1（1） | 段 |
| `length_score` | 0～1（1） | 段 |
| `paragraph_structure` | 0～1（1） | 段 |
| `transition_usage` | 0～1（1） | 段 |
| `descriptive_richness` | 0～1（1） | 段 |
| `information_gain` | 0～1（1） | 段（依赖 novel brief） |
| `expansion_grounding` | 0～1（1） | 段（依赖 novel brief） |
| `rag_utilization` | 0～1（1） | 段（依赖 novel brief） |
| `hallucination` | 0～1（1） | 段（依赖 novel brief） |
| `year_diversity` | 0～1（1） | 篇 |
| `merged_length` | 0～1（1） | 篇 |
| `inter_chapter_repetition` / `style_consistency` / `summary_sentence_ratio` | 0～1（1） | 跨章（≥2 章） |

逐项公式见 **§5.2**；加权综合分 **§5.3**；门控阈值 **§5.7**。

### 5.2 段级指标：计算公式与取值说明

本节与代码 **逐字对齐**（若报告或脚本展示的「公式摘要」与本节不一致，**以本节为准**）。

#### 5.2.1 `entity_coverage`

- **源码**：`AccuracyMetrics.entity_coverage`
- **`reference_entities`**：长文流水线中为 **至多 10 条**新知识实体名列表。
- **`task_type=="expansion"`（默认）**  
  - 令 `target = min(min_required_entities, len(reference_entities))`，`min_required_entities` 默认 **2**。  
  - `covered` = `reference_entities` 中 **`name in generated_text`（子串）** 命中条数（非模糊）。  
  - **`value = min(covered / target, 1.0)`**（封顶 1.0，不因未用满全部检索实体而降过 1）。  
  - 说明文案区分已用 / 未用列表。
- **`task_type=="summarization"`**  
  - **`value = covered / len(reference_entities)`**。
- **门控**：`QualityThresholds.min_entity_coverage`（默认 **0.80**）**已定义**，但 **`check_quality_gate` 当前未读取该字段**（§5.7）。

#### 5.2.2 `temporal_coherence`

- **源码**：`AccuracyMetrics.temporal_coherence`
- 从正文用正则 `(\d{4})年?` 抽取年份，过滤到 **1000～2100**。
- 无参考年份 → `value=0.5`；正文无有效年份 → `value=0.3`。
- 有参考年 `ref`、容差默认 **±10 年**：对去重后每个年份，落在 `[ref−10, ref+10]` 记为合法。  
  - **`value = 合法年份种类数 / 正文中年份种类总数`**。

#### 5.2.3 `rag_entity_accuracy`

- **源码**：`AccuracyMetrics.rag_entity_accuracy`
- 合并 **对齐实体 + 新知识实体** 列表。
- **`_is_entity_mentioned(name, generated_text)`** 命中则认为该实体「被使用」；当前实现下 **每一条被使用的实体都计为 accurate**（`accuracy = accurate / used`，无 used 则 **value=1.0**。描述级 LLM 核对未启用）。

#### 5.2.4 `topic_coherence`

- **源码**：`RelevanceMetrics.topic_coherence`
- 用 jieba 词性从 **回忆录原文**抽取核心概念（`nr/ns/nt/nz/n`，去黑名单）。
- **`value =（在生成文中被 `_is_concept_preserved` 判为保留的概念数）/ 概念总数`**  
  （保留逻辑：全文归一子串 / 二元片段 / 双字全长等）。

#### 5.2.5 `length_score`

- **源码**：`LiteraryMetrics.length_score`
- **`len(generated_text)`** 按字符数计量。
- 落在 `[optimal_min, optimal_optimal]`（由 `calculate_all_metrics` 传入；**extension 场景**见 `long_form_eval._metrics_for_segment`：**min/optimal/max 由 length_hint + 回忆录长度推导**，较宽）
  - **`value = 1.0`**
- 落在 `[literary_length_min, literary_length_max]` 但不在最优区间：**0.7～1.0 线性插值**
- **否则**：`value = 0.3`

#### 5.2.6 `paragraph_structure`

- **源码**：`LiteraryMetrics.paragraph_structure`
- **`relaxed=False`**：按 `\n` 分段；段数 **1～3→1.0**；**0→0**；**其他→0.6**。
- **`relaxed=True`**（长文段级）：**≤48 段→1.0**，**再多→0.75**，无内容→0。

#### 5.2.7 `transition_usage`

- **源码**：`LiteraryMetrics.transition_usage`
- 固定过渡词列表（如「那时」「后来」「彼时」……）在正文中命中：  
  - **≥2 个不同词**：`value=1.0`；**恰好 1 个**：**0.7**；**0 个**：**0.4**。

#### 5.2.8 `descriptive_richness`

- **源码**：`LiteraryMetrics.descriptive_richness`
- **4 条正则**：感官字类、`XX的`、比喻词、情感单字类等；每条在文中出现则记 1 分。  
- **`value = 命中正则条数 / 4`**。

#### 5.2.9 `information_gain`

- **源码**：`information_gain_metric` → `analyze_novel_content.novel_entities_used` 去重后 **个数**分段：  
  - **0→0.0**，**1→0.4**，**2→0.7**，**≥3→1.0**（与「可用实体总数」**无比例除法**。）
- 「被使用」由 **`_is_mentioned_in_text(实体名, 正文)`**（含中英文地名与拆词映射）判定。

#### 5.2.10 `expansion_grounding`

- **源码**：`expansion_grounding_metric`  
- **若本章未使用任何新知识实体**：`value = 0.0`。  
- **否则**：**`value = 1.0`**（隐含假设：所用实体均在白名单）；若仍有无依据的抽出事实，**仅在 explanation 中提示数量**，不改变上述分值。  
- 另：`NovelContentAnalysis.expansion_grounding` **属性**为 **`len(grounded_facts)/len(new_facts)`**，在写入 `novel_content_info` 时会被 **`MetricResult.expansion_grounding` 覆盖**，故 JSON/报告应与指标对象一致。

#### 5.2.11 `rag_utilization`

- **源码**：`rag_utilization_metric`，按 **`novel_entities_used` 个数**离散：  
  - **0→0.0**，**1→0.3**，**2→0.6**，**3→0.8**，**≥4→1.0**。

#### 5.2.12 `hallucination`

- **源码**：`hallucination_metric`  
- 与 `information_gain` 相同方式抽取 **`all_extracted`**；**回忆录已出现**的子串不归为幻觉候选。  
- **无支撑** = 非回忆录内、且不能与 **`novel_entity_names` 模糊匹配**、且 **`_is_grounded_in_rag(...)`**（拼接 description/关系/snippet，`n-gram` 默认阈值 **0.6**）为假。  
- **`hallucination_rate = 无支撑数 / len(all_extracted)`**（无抽取→率 0）  
- **`value = 1.0 − hallucination_rate`**。  
- **门控**：**未接入**（`quality_gate.py` 注释说明仅评分）。

---

### 5.3 段级加权综合分 `aggregate_scores`（映射到 0～10）

- **源码**：`metrics.aggregate_scores`  
- **`normalized_k = MetricResult.value / MetricResult.max_value`**  
- **默认权重**（仅对已存在的 metric 键求和，`total_weight` 为对这些键权重之和）：  

| 键 | 默认权重 |
|----|-----------|
| `entity_coverage` | 1.5 |
| `temporal_coherence` | 1.0 |
| `rag_entity_accuracy` | 1.5 |
| `topic_coherence` | 1.5 |
| `length_score` | 0.8 |
| `paragraph_structure` | 0.8 |
| `transition_usage` | 0.8 |
| `descriptive_richness` | 0.8 |
| `information_gain` | 1.2 |
| `expansion_grounding` | 1.5 |
| `rag_utilization` | **2.0** |
| `hallucination` | **2.0** |

- **返回值**：**`(Σ normalized_k × weight_k / total_weight) × 10`**，即 **0～10**。  
- **长文中「段级写入门控的综合分」**：`segment_scores`，若启用 Evaluator LLM 总评则用其 **`overall_score`**，否则为该聚合分（见 `evaluate_long_form`）。

### 5.4 篇级指标（合并全文）

#### `year_diversity`（`long_form_eval.document_year_diversity`）

- 提取四位年份；**无年份→0.5**；年份种类 **≤3→1.0**；**种类更多→0.75**。

#### `merged_length`（`LiteraryMetrics.length_score`）

- 对 **`long_form.merged_content`**：令 `L = len(merged_content)`，**`min_length=400`**，**`max_length=50000`**，**`optimal_min=800`**，**`optimal_max = max(800, min(L, 12000))`**，打分规则同 **§5.2.5**。

### 5.5 跨章指标（≥2 章，`CrossChapterMetrics`）

均输出 **max_value=1.0**，**value 越大越好**（除语义上重叠率需在说明里读）。

#### `inter_chapter_repetition`

- **ngram_n** 默认 **6**（去空白字符）；相邻章：`overlap = |A∩B| / min(|A|,|B|)`，`avg_overlap` 为相邻对平均；**`value = max(0, 1 − avg_overlap)`**。

#### `style_consistency`

- 各章平均句长（按 `。！？`切句）；算变异系数 **`cv`**。  
  - **`cv ≤ 0.2 → value=1.0`**；**`cv ≥ 0.6 → 0.3`**；其间线性：**`1 − (cv−0.2)/0.4 × 0.7`**。

#### `summary_sentence_ratio`

- 与门控同源总结句正则；**ratio = 总结句句数 / 总句数**；**`value = max(0, 1 − ratio × 5)`**（ratio=20%→0）。

### 5.6 FActScore（可选，`enable_fact_check=True`）

- **不在**默认 `calculate_all_metrics` 里；仅 **`FActScoreChecker.check`**。
- **流程**：原子化事实（LLM 或规则分解）→ 过滤纯文学描写 → 与回忆录 **关键词规则命中**→ 剩余批次 **LLM 对照「回忆录原文 + retrieval.get_context_text()」**验证。  
- **`factscore = supported_facts / total_facts`**；若 **`total_facts == 0`**（无可验证原子事实），代码中比值取 **0.0**，门控侧若仍将该章 **`fact_scores` 视作已提交**，则可能 **误判为低于 `min_fact_score`**——宜结合 `fact_check` 明细或按需跳过该检查。
- **门控**：若传入 `fact_scores`，**`factscore < min_fact_score`（默认 **0.60**）→ 章级 **error**。

### 5.7 质量门控：完整阈值、`QualityThresholds` 与**实际生效**项

**类**：`quality_gate.QualityThresholds`  
**默认值**与 **`QualityThresholds.for_expansion_task()`** 并列如下（括号内为 **for_expansion_task** 与默认不同的地方）：

| 字段 | `QualityThresholds` 默认 | `for_expansion_task()` |
|------|--------------------------|-------------------------|
| `min_segment_score` | **5.0** | 5.0 |
| `max_cross_repetition` | **0.20** | 0.20 |
| `min_fact_score` | **0.60** | 0.60 |
| `min_length_ratio` | **0.40** | 0.40 |
| `max_length_ratio` | **2.5** | 2.5 |
| `max_summary_sentence_ratio` | **0.30** | 0.30 |
| `min_semantic_similarity` | `None` | **0.15** |
| `min_expansion_grounding` | **0.40** | 0.40 |
| `min_entity_coverage` | **0.80** | 0.80 |
| `min_rag_utilization` | **0.6** | 0.6 |

**重要**：`**check_quality_gate` 当前只对以下阈值读表**——其余字段仅占位，**改之不影响通过/失败**：

| 检查项 | 条件 | severity | `passed` |
|--------|------|----------|----------|
| 字数比 | **`actual/target < min_length_ratio`** | **error** | 不通过 |
| 字数比 | **`actual/target > max_length_ratio`** | **warning** | 仍可过 |
| 段级综合 | **`segment_score < min_segment_score`** | **error** | |
| FActScore | **`fact_scores` 非空且 `< min_fact_score`** | **error** | |
| RAG 利用 | **`rag_utilization.value < min_rag_utilization`** | **error** | |
| 总结句占比 | **`> max_summary_sentence_ratio`** | **warning** | |
| 套话 | **套话正则命中次数 ≥ 2** | **warning** | |
| 非末章软性结尾 | **`_detect_epilogue` 命中末尾 3 句内** | **warning** | |
| 跨章 6-gram | **相邻章 `overlap_ratio > max_cross_repetition`** | **error** | |

章级 **`passed` = 该章无 `severity=="error"` 的 issue**；篇级 **`passed` = 各章 passed 且无跨章 error**。**`hallucination` 不参与门控**（代码注释）。

门控用到的 **相邻章重叠算法**与 **§5.5** `inter_chapter_repetition` 相同（**6-gram**，**`len(A∩B) / min(|A|,|B|)`**）。

---

### 5.8 新内容评估

**主路径代码**: `src/evaluation/novel_content_metrics.py`（`analyze_novel_content`）；与各 **数值指标**的公式见 **§5.2.9～5.2.12**；与其它段级的聚合见 `long_form_eval`。

**新事实抽取**：
- **线上默认**：优先用 LLM 从生成文中抽取专有名词列表；若无 LLM 则用 jieba 词性（`nr/ns/nt/nz`）等规则兜底。
- **文档中的三层漏斗**（词性→LAC→句式）实现在同一文件的下层函数中（如 `_extract_new_facts`），**并非**当前 `analyze_novel_content` 的默认分支，仅少数测试会直接调用。

**RAG 支撑（溯源）判定**：
1. 先在白名单**实体名称**上做模糊对齐（`_find_matching_entity` / `_fuzzy_entity_match`，含中英文地名映射表）。
2. 若名称对不上（例如英文名 `SHENZHEN CITY` 与正文「深圳」），再使用 **`_build_rag_source_text`**：把实体名、各实体 **description**、关系三元组与 snippets 拼接为一片「RAG 来源全文」。
3. **`_is_grounded_in_rag`** 在该全文上做子串、n-gram、词级等匹配；从而在事实来自**条目描述中的措辞**而与图谱「标题」字面不一致时，仍能判为有据。

**幻觉判定口径（与实现对齐）**：对从生成文抽出的每一条「新事实」专名——**不限于地名**，也可为**事件名、机构名、人物名**或其它被 LLM 抽成独立条目的表述——只要经上列 (1)～(3) 任一路径可判定**与当章 RAG 拼接上下文一致或可归因于其中**，即计入**有 RAG 支撑**，在 `hallucination_metric` 等分支中**不当作无支撑（疑似幻觉）**。未能溯源的才进入「无 RAG 支撑」列表。

本节与 **§8.8** 评估侧中英文明名、描述级误判的 Bugfix 一一对应；生成侧 prompt 内中文展示见 **§8.9**。

**（备选）三层漏斗示意图**——用于理解仓库内另一类提取实现，不等于评估主链路：

```
[第一层] 词性粗筛 (jieba)
    ↓ 保留 nr/ns/nt/nz 词性
[第二层] NER 验证 (LAC)
    ↓ 确认 PER/LOC/ORG/TIME
[第三层] 句式补充 (正则)
    ↓ 年份+事件、实体+关系+实体
最终候选新事实列表
```

**`_is_grounded_in_rag` 内摘要策略**（4 类）：实体名归一与映射、全文子串、模糊 n-gram（默认阈值）、多词中文的词级覆盖率等（详见代码）。

**描述 / snippet 中出现即可「有据」（专名泛指）**：  
检索拼接全文里若出现与该专名一致的子串或可映射片段，即可能判为有据，**不要求**图谱上存在一个与该字符串完全相同的**顶点标题**。典型类型包括但不限于：
- **地名**（示例：正文「福建」↔ 「经济特区」描述中的 *Fujian Province* / 「福建省……」）；  
- **机构 / 事件**（示例：正文「女排精神」↔ 「中国女排」长描述中带引号的同一提法）；  
- **人物**（示例：中英文姓名变体已通过 `_fuzzy_entity_match` / 映射表或描述内中文译名对上检索块）。

报告中往往只节选部分实体段落，运行时 **`rag_source_text` 可能比报告节选更长**（含 **`novel_snippets`** 等），故「似乎报告里没看见」不等于未参与匹配。

**解读边界（仍可溯源 ≠ 叙事已证真）**：**有 RAG 支撑**仅表示「与当次检索所得的文本证据可对齐」（含描述中的二次信息），**不等于**逐项证明叙事细节属实（例如某地舍友籍贯、某人具体言行）。若业务需要「仅白名单顶点标题」「仅某类实体」才允许接地，需在 `novel_content_metrics` 中另行收紧 grounding 规则。

**与 §5.2 的分工**：§5.2 给出各指标**数值定义**与门控**生效项**（§5.7）；本节专述 **RAG 溯源与「专名是否算幻觉」的产品语义**。

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
| `novel_content_extractor.py` | 新内容提取与分类；`_truncate_for_prompt`、`_localize_entity_name`；白名单最少用量规则文案 |
| `prompts.json` | 各风格 user 模板与 `system_prompts`（含最少 2 个 RAG 实体等约束） |
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

### 8.7 Phase 7: Prompt 体量失效与检索器路径类型（Bugfix，2026-05-10）

**现象 1 — `novel_context` 失控、约束形同虚设**

- **症状**: 评估报告仍配置白名单与「严格禁止未列出实体」，但生成中频繁出现不在白名单内的专名；Pipeline 实测整段 user prompt 可达 **约 2.1 万字符**，其中 **`novel_context` 约 1.98 万字符**。
- **根因**: `format_for_prompt` 中对实体 `description` 的「最多约 200 字」逻辑依赖 `split("。")`；检索库中英文明细不含中文句号「。」时，`split` 仅返回整段，截断从未生效，英文维基式长描述原样进入 prompt。
- **副作用**: 文末「严格使用规则」与模板中的防幻觉条款被超长正文淹没（注意力与上下文窗口双重压力），与「须使用 RAG 白名单」的目标冲突加剧。
- **修复**:
  - 引入 `_truncate_for_prompt()`：兼容中英文句读与换行，并设硬上限（默认 200 字）。
  - 关系边的 `description` 同步使用该截断，避免关系块同样膨胀。
  - 在白名单规则中补充：**英文 / 拼音条目可用通行中文译名**，须明显对应同一条目。
- **验证（日志）**: 修复后同场景下「美国」实体描述由 **5280 → 179** 字符量级注入；`novel_context_len` **19800 → 3061**；`total_prompt_len` **20970 → 4231**。

**现象 2 — `MemoirRetriever` 初始化崩溃**

- **症状**: `generation_test/test_full_pipeline.py` 等调用传入 `index_dir="data/graphrag_output"`（`str`）时，`_load_data` 中执行 `self.index_dir / "output"` 报错：`TypeError: unsupported operand type(s) for /: 'str' and 'str'`。
- **根因**: 类型标注为 `Path`，但未将调用方传入的字符串转为 `Path`。
- **修复**: `__init__` 内统一执行 `self.index_dir = Path(index_dir) if index_dir is not None else default_root`（并保留默认索引根路径）。

**说明**: 修复后若评估仍报部分「无 RAG 支撑」专名，可能来自 (1) 模型虚构舍友籍贯等地名；(2) ~~评估侧中英文实体名对齐~~（已为 §8.8 修复）；或截断裁掉的关键句——与上述「完全不截断」属不同层面问题。

---

### 8.8 Phase 8: 评估阶段实体对齐与描述溯源（Bugfix）

**触发场景**：Pipeline 报告显示检索白名单中含 `SHENZHEN CITY`、「经济特区」等，生成文中使用中文「深圳」及来自描述细节的表述（如历史上的典型口号）；评估却仍将「深圳」列为无 RAG 支撑，或将 `SHENZHEN CITY` 记为「未使用」。

**现象 1 — 多词英文图谱名无法在正文中等价到中文地名**

- **症状**：`novel_entity_accuracy` / `information_gain` / `rag_utilization` 偏低；报告中「深圳」被列为疑似幻觉，`SHENZHEN CITY` 被列为未使用实体。
- **根因**：`_EN_CN_LOCATION_MAP` 仅以 **单个英文 token** 为键（如 `shenzhen`），图谱标题为 **`SHENZHEN CITY`** 时用整串 `shenzhen city` 查找映射失败；此前亦未拆词回退。
- **修复**（`src/evaluation/novel_content_metrics.py`）：
  - 在 **`_is_mentioned_in_text`**、**`_fuzzy_entity_match`** 中：对英文名按 **拉丁词切分**，逐词查表，任一 variant 映射到中文且正文含该中文即视为命中。
  - **`metrics.py` 中 `_is_entity_mentioned`**（`rag_entity_accuracy` 等沿用）：同样在函数体内复用中英文映射，避免段级与子模块口径不一致。
- **验证**：实测日志中 `SHENZHEN CITY` → `mentioned: true`；章节报告可出现「深圳」归入有 RAG 支撑的新事实，且统计使用实体含 `SHENZHEN CITY`。

**现象 2 — 仅用实体「标题」做溯源，忽略 description**

- **症状**：`_build_rag_source_text`、**`_is_grounded_in_rag`** 已在仓库中存在，但未接入 **`analyze_novel_content`**；新事实与白名单对齐时等价于「只比对实体名字符串」，正文中复述自 **长描述里的句子或口号**（与 title 字面不同）易被误判为幻觉。
- **修复**：
  - 在 **`analyze_novel_content`** 中：对每个抽取到的新事实先 `_find_matching_entity`；失败后对 **`_build_rag_source_text(novel_content_brief)`** 调用 **`_is_grounded_in_rag`**。
  - 将 **`hallucination_metric`** 的无支撑判定与上述逻辑对齐（名称模糊匹配 + 描述全文溯源），避免与 `grounded_facts` 口径矛盾。
  - **`_is_grounded_in_rag`**：补充与 **`_fuzzy_entity_match`** 一致的实体名归一逻辑，及对中文地做名的 **反向英文 variant** 在 RAG 全文中的检索。

**残余边界**：若模型输出为 **训练记忆** 中的短语、且图谱 **任一** 条目 description 中英文均不包含可匹配的子串或 n-gram 重叠（例如口号仅有中文改写、向量库却无对应中文句），规则引擎仍可能标为「无描述支撑」——需在产品层决定是否引入口号白名单或语义相似度，本修复不假定通用常识等价于 RAG 证据。

**语义补充**：凡专名只要能在 **description / 关系 / snippet** 中与 RAG 拼接全文对齐即可判有据、不记入「无支撑幻觉」，不限于地名；典型例子见 **§5.8** 末段。**有 RAG 支撑**仍不等于叙事细粒度事实已获逐条背书，参见该节「解读边界」。

**涉及文件**：`src/evaluation/novel_content_metrics.py`，`src/evaluation/metrics.py`（`_is_entity_mentioned`）。

---

### 8.9 Phase 9: 最少 RAG 用量与白名单中文展示（生成侧 Bugfix）

**现象 1 — 检索命中多条但正文几乎只用 1 条**

- **症状**: `rag_utilization`、`information_gain` 偏低；质量门控报「须明确要求使用检索到的背景知识」；视觉上扩展不充分。
- **根因**: 模板与 `novel_context` 内规则均写「选取 **1–3** 个」——最小合规用量为 **1**，模型倾向省力策略。
- **修复**: `prompts.json` 中 `standard` / `nostalgic` / `narrative` 用户模板及 `system_prompts.default` 改为 **2–4 个（至少 2 个）**；`format_for_prompt` 内「严格使用规则」同步为「至少使用 **2** 个实体或事件」。

**现象 2 — 正文混入英文图谱标题**

- **症状**: 中文回忆录段落中出现 `DENG XIAOPING`、`SHENZHEN CITY`、`TRIAL OF LIN BIAO AND JIANG QING GROUPS` 等与文风割裂的英文专名。
- **根因**: 白名单首行「✓ 允许使用的实体名称」以图谱 **`name` 原始英文字符串** 列出；模型将其当作合法字面称谓复制进输出。
- **修复**（`src/generation/novel_content_extractor.py`）：
  - **`_localize_entity_name(name, description)`**：英文名时在描述前部抽取括号中文、`also known as` 后中文或行首中文等，用于 prompt **展示标题**与关系的 **source/target 展示**；清单前缀改为「实体名称（中文）」。
  - **规则文案**：明确要求叙事为纯中文，**禁止**正文中出现英文图谱标题式专名（可点名示例）。
- **残余边界**: 描述前部不含任何可抽取中文 pattern 时仍退回英文名——可考虑检索流水线回填中文别名或扩大抽取规则。

**涉及文件**：`src/generation/novel_content_extractor.py`，`src/generation/prompts.json`。

---

### 8.10 关键设计决策

| 决策 | 原因 |
|------|------|
| **每章独立检索** | 避免主题漂移，提高相关性 |
| **预取优化** | 减少总耗时（检索与生成并行） |
| **Prompt 注入描述截断** | 控制 `novel_context` 体量，保证白名单规则可读、可遵守 |
| **最少融入 2 个白名单实体** | 避免模型仅用单条「搪塞」RAG 约束，提高利用率与时代信息量 |
| **白名单展示名中文化** | 降低模型照抄英文 `title` 污染中文叙事的概率 |
| **新事实抽取 + RAG grounding** | 主线为 LLM/词性规则；可选用三层漏斗候选生成； grounding 对齐实体名称并拼接实体 **description**/关系/snippet；**任一专名可追溯则不计入 hallucination「无支撑」** |
| **跨章上下文管理** | 避免重复，保持衔接 |
| **质量门控** | 自动化质量保障 |
| **扩展系数而非固定总量** | 适应不同长度的原文 |

---

**文档版本**: v2.6  
**最后更新**: 2026-05-10
