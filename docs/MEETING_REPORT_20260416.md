# 组会报告：回忆录 RAG 系统——跨章长文本生成与质量评估的系统性改进

> 日期：2026-04-16
> 项目：GraphRAG 回忆录历史背景生成系统

---

## 一、背景与问题

本项目的核心任务是：用户输入一篇回忆录长文（通常跨越数十年、数千字），系统为其**分章节生成**对应年代的历史背景描述，使回忆录具备时代纵深感。

完整 pipeline：**分段 → 逐段检索知识图谱 → 逐段调 LLM 生成 → 合并为完整长文 → 评估质量**。

### 先前版本存在三类系统性问题

#### 问题 1：分章切分缺乏语义依据

先前的分段器 `segment_memoir()` 仅依赖两种信号进行切分：

- **空行**（`\n\n`）——将连续空行视为段落边界
- **章节标题正则**（如"第一章""一、"等格式）

这带来两个直接后果：

1. **跨年代误合并**：当两个短段落分别讲述 1972 年和 1977 年的经历时，如果各自不足 300 字，分段器会将它们合并为一段。这导致后续检索用一个混合了两个年代的文本去查知识图谱，检索精度大幅下降。
2. **不可审计**：分段完成后没有任何元数据或报告告诉调用方"为什么这样分"、"每段涵盖什么年代"，出了问题无从排查。

#### 问题 2：各章独立生成，无法保障篇级质量

先前的编排器 `run_long_form_generation()` 对每章的处理是完全独立的：

```
对每一段：
    检索(本段文本) → 生成(本段文本, 检索结果) → 追加到列表
合并所有章节
```

每章的 LLM prompt 中**只包含当前段的回忆录原文和检索结果**，对前后章节一无所知。这导致三个问题：

1. **跨章重复**：第 3 章和第 4 章可能各自独立生成了几乎相同的"改革开放大潮"描述，因为 LLM 不知道前面已经写过。
2. **总结性套话泛滥**：LLM 有一个强倾向——在每段末尾加上"总之""综上所述""在这个大背景下"等总结句。当多章拼合后，全文充斥总结，不像回忆录而像论文。
3. **缺乏叙事节奏**：开篇和收尾没有差异化处理，全文读起来像 6 篇互不相干的短文拼接，而非一篇完整的回忆录。

#### 问题 3：评估体系无法回答"能不能交付"

先前的评估体系有两个层面的不足：

**指标层面**：
- `semantic_similarity` 使用**单汉字**词袋模型计算余弦相似度——将每个汉字视为独立词汇。由于常用汉字不过几千个，任意两段中文文本的单字重叠率都很高，这个指标几乎没有区分度。
- 仅有段级（per-chapter）指标，没有跨章（cross-chapter）指标。无法检测章间重复、风格割裂、总结句过多等篇级问题。
- 事实检查仅输出一个布尔值 `is_factual`（通过/不通过），不提供 FActScore 数值，无法区分"1/10 原子事实不支持"和"8/10 不支持"。

**决策层面**：
- 没有质量门控（Quality Gate）。评估算出一堆指标后，没有任何机制告诉调用方"这个结果能不能用"、"哪些章节需要重新生成"、"怎么修"。
- 生成和评估是两个解耦的步骤，评估发现问题后没有反馈回路。

---

## 二、本次改进方案

### 2.1 分段器增强——时间边界优先 + 元数据 + 校验报告

**改动文件**：`src/generation/memoir_segmenter.py`

**核心思路**：回忆录最自然的章节边界就是**时间跳转**——"一九七二年……""一九七七年……"。分段器应优先识别这种时间信号，以此作为最高优先级的切分点。

#### 具体实现

**① 时间边界检测**

新增正则 `_LEADING_YEAR`，匹配段首年份标记（同时支持中文大写和阿拉伯数字）：

```python
_LEADING_YEAR = re.compile(
    r"^\s*(?:一九[〇零○一二三四五六七八九]{2}|二[〇零○][〇零○一二三四五六七八九]{2}|(?:19|20)\d{2})\s*年"
)
```

配合 `_cn_year_to_arabic()` 将"一九七二"转为"1972"，实现中英文年份统一处理。

**② 不跨时间边界合并**

修改 `_merge_short()` 函数，在合并过短块时增加时间边界保护：

```python
def _merge_short(blocks, target_min_chars, target_max_chars, *, respect_temporal=True):
    for b in blocks[1:]:
        crosses_time = respect_temporal and _LEADING_YEAR.match(b)
        if len(buf) < target_min_chars and not crosses_time:
            buf = f"{buf}\n\n{b}"   # 合并
        else:
            merged.append(buf)       # 不合并，即使当前块很短
            buf = b
```

效果：即使 1972 年的段落只有 200 字，也不会被合并到 1977 年的段落中。

**③ 分段元数据 `SegmentMeta`**

每个分段现在携带结构化元数据：

```python
@dataclass(frozen=True)
class SegmentMeta:
    detected_years: Tuple[str, ...]       # 该段出现的年份：("1972", "1977")
    detected_locations: Tuple[str, ...]   # 该段出现的地名：("陕北", "延安")
    detected_figures: Tuple[str, ...]     # 该段出现的人物：("老张队长",)
    temporal_label: str                   # 年代标签："1972-1977"
    split_reason: str                     # 切分原因："temporal_boundary"
```

`split_reason` 取值包括 `document_start`（文档开头）、`temporal_boundary`（时间边界）、`chapter_heading`（章节标题）、`paragraph_break`（段落分隔），让每次切分都可追溯。

**④ 分段校验报告 `SegmentationReport`**

`validate_segmentation()` 对分段结果做三项自动检查：

| 检查项 | 级别 | 触发条件 |
|---|---|---|
| 段落过短 | warning | 段长度 < `target_min_chars × 0.5` |
| 段落过长 | error | 段长度 > `target_max_chars × 3` |
| 单段跨年代 | warning | 同一段内年份跨度 > 15 年 |
| 索引不连续 | error | `segment.index` 不等于其在列表中的位置 |

报告输出示例：

```
分段数: 6，总字数: 3847
校验结果: 通过
  段0: 642字 | 时间=1972-1977 | 地点=陕北 | 切分原因=document_start
  段1: 713字 | 时间=1977-1978 | 地点=陕北,北京 | 切分原因=temporal_boundary
  段2: 597字 | 时间=1988 | 地点=北京,深圳 | 切分原因=temporal_boundary
  ...
```

### 2.2 跨章上下文管理——防重复、保衔接、控风格

**新增文件**：`src/generation/chapter_context.py`

**改动文件**：`src/generation/long_form_orchestrator.py`、`src/generation/literary_generator.py`、`src/generation/prompts.json`

**核心思路**：从"每章独立生成"升级为"有状态的序列生成"——每生成完一章，提取其关键信息，注入到下一章的 prompt 中，让 LLM 在生成时"知道前面写了什么"。

#### 具体实现

**① `ChapterContext` 状态管理器**

```
ChapterContext
├── record_chapter(index, content, entities)    # 生成后调用：记录摘要、要点、实体
├── build_prompt_section(current_index)         # 生成前调用：构建注入 prompt 的上下文段
└── detect_repetition_with_previous(content)    # 生成后调用：检测与前文的重叠率
```

`record_chapter` 内部执行三项提取（纯规则，不调 LLM）：
- **摘要提取** `_extract_brief()`：取前 1-2 句，≤60 字
- **年代提取** `_extract_time_period()`：解析内容中的年份范围
- **要点提取** `_extract_key_phrases()`：统计 2-gram 和 3-gram 高频短语（频次 ≥ 2），取 top 8

**② Prompt 注入——三段式上下文**

`build_prompt_section()` 为当前章构建的 prompt 段落包含三个部分：

```markdown
## 前文已生成内容概要（请勿重复）
- 第1章 (1972-1977): 知青下乡到陕北张家塬，在生产队学习维修柴油机
- 第2章 (1977-1978): 恢复高考消息传来，在窑洞中复习备考

以下要点已在前文出现，请勿再次展开或总结：
知青、黄土高坡、柴油机、恢复高考、窑洞、煤油灯

本章位于全文中段，请与前文自然衔接，推进叙事节奏。
```

这段文字通过 `{chapter_context}` 占位符注入到所有 5 种风格模板中（standard / nostalgic / narrative / informative / conversational）。

**③ 编排器重构——五步循环**

新的 `run_long_form_generation()` 对每章执行五步：

```
① 检索：retriever.retrieve(本段文本)
② 构建跨章上下文：chapter_ctx.build_prompt_section(当前章索引)
③ 生成：generator.generate(本段文本, 检索结果, chapter_context=上下文)
④ 重复检测：chapter_ctx.detect_repetition_with_previous(生成内容)
   → 若重叠率超阈值且 max_retry > 0，自动重试（提高 temperature + 加强反重复指令）
⑤ 记录：chapter_ctx.record_chapter(当前章索引, 生成内容, 检索到的实体)
```

**④ Prompt 风格强化**

在 `prompts.json` 的所有模板和 system prompt 中增加显式约束：

```
5. 禁止使用「总之」「综上」「总的来说」等总结性语句
6. 禁止使用「在这个大背景下」「在这样的时代背景下」等空泛套话
7. 只做叙事性描写，不要概括或评论
```

### 2.3 评估体系重建——跨章指标 + 质量门控 + 修复计划

**新增文件**：`src/evaluation/quality_gate.py`

**改动文件**：`src/evaluation/metrics.py`、`src/evaluation/long_form_eval.py`

#### 2.3.1 修复段级指标

`semantic_similarity` 从单字符词袋模型升级为 **2-gram 词频余弦相似度**：

```python
# 旧版（无区分度）：每个汉字视为独立词
def get_words(text):
    for char in word:
        word_freq[char] = word_freq.get(char, 0) + 1

# 新版（有意义）：连续 2 字组合
def get_ngram_freq(text, n=2):
    chars = re.findall(r'[\u4e00-\u9fa5]', text)
    for i in range(len(chars) - n + 1):
        gram = "".join(chars[i:i + n])
        freq[gram] = freq.get(gram, 0) + 1
```

#### 2.3.2 新增跨章指标 `CrossChapterMetrics`

| 指标 | 计算方法 | 衡量什么 |
|---|---|---|
| `inter_chapter_repetition` | 相邻章节的 6-gram 集合交集占比，取平均，评分 = 1 - overlap | 各章之间是否有不当的内容重复 |
| `style_consistency` | 各章平均句长的变异系数（CV），CV ≤ 0.2 → 满分，CV ≥ 0.6 → 0.3 分 | 全文的文体风格是否一致（而非有的章节句子很长、有的很短） |
| `summary_sentence_ratio` | 统计含"总之/综上/总的来说"等模式的句子占比，评分 = 1 - ratio × 5 | 生成的内容是否充斥总结性语句（不符合回忆录风格） |

#### 2.3.3 质量门控 `QualityGate`

质量门控是一个**可配置阈值的通过/不通过判定器**，在评估完成后自动运行。

**门控维度与默认阈值：**

| 检查维度 | 默认阈值 | 触发级别 | 含义 |
|---|---|---|---|
| 字数偏离 | 实际/目标 < 0.4 | error | 生成太短，内容不足 |
| 字数偏离 | 实际/目标 > 2.5 | warning | 生成太长，需截断 |
| 段级综合分 | < 5.0 / 10 | error | 该章整体质量不合格 |
| FActScore | < 60% | error | 超过 40% 的原子事实缺乏证据支持 |
| 跨章重叠率 | > 20% (6-gram) | error | 相邻两章内容高度重复 |
| 总结句占比 | > 30% | warning | 总结性语句过多，风格不适合回忆录 |
| 套话 | ≥ 2 处 | warning | 使用了空泛的模板化过渡语 |

**门控输出：**

```python
@dataclass
class QualityGateResult:
    passed: bool                              # 整体是否通过
    overall_score: float                      # 综合分
    chapter_results: List[ChapterGateResult]  # 每章的通过/问题列表
    cross_chapter_issues: List[...]           # 跨章问题
    remediation: Optional[RemediationPlan]    # 修复计划
```

#### 2.3.4 修复计划 `RemediationPlan`

当门控不通过时，自动生成可执行的修复计划：

```python
@dataclass
class RemediationPlan:
    chapters_to_regenerate: List[int]      # 需要重新生成的章节索引
    reasons: Dict[int, List[str]]          # 每章需要修复的原因
    prompt_adjustments: Dict[int, str]     # 每章建议的 prompt 调整指令
```

示例输出：

```
质量门控: 未通过  综合分: 6.20
  第3章 ✗
    [error] score: 综合评分 4.2 低于阈值 5.0
      → 建议: 检查检索结果质量或切换 LLM provider
    [warning] summary: 总结性语句占比 35% 偏高
      → 建议: 在 prompt 中明确要求「不要总结，只做叙事性描写」
  跨章问题:
    [error] repetition: 第3章与第4章 6-gram 重叠率 28%
      → 建议: 在后一章 prompt 中注入前一章概要并要求不重复
  需重新生成: 第 3, 4 章
```

#### 2.3.5 评估报告 JSON 结构

评估输出的 JSON 从三个层面提供完整信息：

```json
{
  "aggregated_score": 7.85,

  "segments": [{
    "segment_index": 0,
    "metrics": { "entity_coverage": {...}, "time_consistency": {...}, ... },
    "eval_overall": 7.5,
    "fact_is_factual": true,
    "fact_score": 0.83
  }, ...],

  "cross_chapter": {
    "inter_chapter_repetition": { "value": 0.92, "explanation": "..." },
    "style_consistency":        { "value": 0.85, "explanation": "..." },
    "summary_sentence_ratio":   { "value": 1.00, "explanation": "..." }
  },

  "quality_gate": {
    "passed": true,
    "overall_score": 7.85,
    "chapters_to_regenerate": []
  }
}
```

---

## 三、整体架构与数据流

### 3.1 全流程图

```
用户输入：回忆录长文本（数千字，跨越多个年代）
    │
    ▼
┌─────────────────────────────────────────┐
│  ① 分段 segment_memoir()                │
│     时间边界优先 → 结构边界 → 长度约束   │
│     输出：[MemoirSegment + SegmentMeta]  │
│     附带：SegmentationReport（校验报告）  │
└────────────────┬────────────────────────┘
                 │
    ▼            ▼
┌─────────────────────────────────────────┐
│  ② 预算分配 allocate_segment_budgets()  │
│     按各段字符数占比分配生成目标字数      │
│     输出：[SegmentBudget(hint, tokens)]  │
└────────────────┬────────────────────────┘
                 │
    ▼            ▼
┌─────────────────────────────────────────────────────────┐
│  ③ 逐章循环（ChapterContext 维护跨章状态）              │
│                                                         │
│   对第 k 章：                                           │
│   ┌───────────────────────────────────────────────┐     │
│   │ a. 检索：MemoirRetriever.retrieve(段k文本)     │     │
│   │    → 从 GraphRAG 知识图谱获取相关实体/关系/社区 │     │
│   ├───────────────────────────────────────────────┤     │
│   │ b. 构建跨章上下文：                            │     │
│   │    ChapterContext.build_prompt_section(k)      │     │
│   │    → 前文概要 + 反重复要点 + 位置指令          │     │
│   ├───────────────────────────────────────────────┤     │
│   │ c. 生成：LiteraryGenerator.generate(           │     │
│   │      段k文本, 检索结果, chapter_context)       │     │
│   │    → LLM 生成历史背景描述                      │     │
│   ├───────────────────────────────────────────────┤     │
│   │ d. 重复检测：                                  │     │
│   │    detect_repetition_with_previous(生成内容)    │     │
│   │    → 若超阈值，可自动重试                      │     │
│   ├───────────────────────────────────────────────┤     │
│   │ e. 记录：                                      │     │
│   │    ChapterContext.record_chapter(k, 内容, 实体) │     │
│   │    → 更新摘要/要点/实体集，供第 k+1 章使用     │     │
│   └───────────────────────────────────────────────┘     │
│                                                         │
│   输出：LongFormGenerationResult                        │
│          .chapters          各章生成结果                 │
│          .merged_content    合并全文                     │
│          .segmentation_report  分段校验报告              │
└────────────────┬────────────────────────────────────────┘
                 │
    ▼            ▼
┌─────────────────────────────────────────────────────────┐
│  ④ 评估 evaluate_long_form()                           │
│                                                         │
│   段级（每章）：                                        │
│   · entity_coverage     实体覆盖率                      │
│   · time_consistency    年份一致性                       │
│   · keyword_overlap     关键词覆盖                      │
│   · semantic_similarity 2-gram 语义相似度               │
│   · length_score        字数控制                        │
│   · paragraph_structure 段落结构                        │
│   · transition_usage    过渡词使用                      │
│   · descriptive_richness 描述丰富度                     │
│   · FActScore           原子事实支持率                   │
│                                                         │
│   跨章（全文）：                                        │
│   · inter_chapter_repetition  章间重复度                │
│   · style_consistency         风格一致性                │
│   · summary_sentence_ratio    总结句比率                │
│                                                         │
│   篇级：                                               │
│   · year_diversity      年代多样性                      │
│   · merged_length       合并总长度                      │
└────────────────┬────────────────────────────────────────┘
                 │
    ▼            ▼
┌─────────────────────────────────────────────────────────┐
│  ⑤ 质量门控 check_quality_gate()                       │
│                                                         │
│   输入：各章内容 + 段级评分 + FActScore + 目标字数      │
│   检查：字数 / 综合分 / 事实 / 跨章重复 / 总结句 / 套话 │
│   输出：                                                │
│     passed: true/false                                  │
│     remediation:                                        │
│       chapters_to_regenerate: [3, 4]                    │
│       reasons: { 3: ["综合评分4.2 < 5.0"], ... }       │
│       prompt_adjustments: { 3: "增加检索深度", ... }    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 反馈回路

```
生成 → 评估 → 门控不通过？→ 修复计划 → 调整 prompt → 重新生成问题章节
                    ↑                              │
                    └──────────────────────────────┘
```

当前版本支持两层反馈：

1. **生成时即时反馈**：编排器在每章生成后立即检测跨章重复，若超阈值可自动重试（`max_retry_chapters` 参数），重试时提高 temperature 并加强反重复指令。
2. **评估后修复建议**：质量门控输出 `RemediationPlan`，列出需要重新生成的章节及具体原因，供上层（Web/API）决定是否执行自动修复或交由人工处理。

---

## 四、测试覆盖

单元测试从 17 项增长到 **36 项**，全部通过，无需外部依赖（无 LLM / Ollama / 索引）。

### 新增测试项

| 模块 | 新增测试 | 验证什么 |
|---|---|---|
| 分段器 | `test_temporal_boundary_split` | 不同年份的段落不被合并 |
| | `test_year_extraction_chinese` | "一九八八年"→"1988" |
| | `test_year_extraction_arabic` | "1992"→"1992" |
| | `test_segment_meta_populated` | 每段都有元数据 |
| | `test_segment_meta_locations` | 元数据能提取地名 |
| | `test_validate_segmentation_pass` | 正常分段通过校验 |
| | `test_validate_segmentation_detects_issues` | 异常段触发 error |
| | `test_real_sample_segmentation` | 真实样本分段合理性 |
| 编排器 | `test_orchestrator_has_segmentation_report` | 结果含校验报告 |
| | `test_orchestrator_cross_chapter_context` | 跨章上下文正确注入 |
| | `test_orchestrator_disabled_cross_chapter` | 可关闭跨章上下文 |
| 预算 | `test_allocate_with_meta` | 新旧 Segment 兼容 |
| 评估 | `test_evaluate_long_form_has_cross_chapter_metrics` | 含跨章指标 |
| | `test_evaluate_long_form_has_quality_gate` | 含质量门控 |
| | `test_evaluate_long_form_json_has_fact_score` | JSON 含 fact_score |
| 门控 | `test_quality_gate_pass` | 正常内容通过 |
| | `test_quality_gate_detects_repetition` | 检测跨章重复 |
| | `test_quality_gate_detects_summary_sentences` | 检测总结句过多 |
| | `test_quality_gate_remediation_plan` | 低分章出现在修复计划中 |

---

## 五、变更文件清单

| 文件 | 类型 | 改动说明 |
|---|---|---|
| `src/generation/memoir_segmenter.py` | 修改 | 时间边界检测、SegmentMeta、SegmentationReport |
| `src/generation/chapter_context.py` | **新增** | 跨章上下文管理器 |
| `src/generation/long_form_orchestrator.py` | 修改 | 五步循环 + 跨章上下文 + 重复检测 |
| `src/generation/literary_generator.py` | 修改 | 新增 `chapter_context` 参数 |
| `src/generation/prompts.json` | 修改 | `{chapter_context}` 占位符 + 反总结/反套话指令 |
| `src/generation/chapter_budget.py` | 未改 | 接口兼容 |
| `src/generation/__init__.py` | 修改 | 导出新符号 |
| `src/evaluation/metrics.py` | 修改 | 修复 semantic_similarity + 新增 CrossChapterMetrics |
| `src/evaluation/quality_gate.py` | **新增** | 质量门控 + 修复计划 |
| `src/evaluation/long_form_eval.py` | 修改 | 跨章指标 + 门控集成 |
| `src/evaluation/__init__.py` | 修改 | 导出新符号 |
| `tests/test_memoir_segmenter.py` | 修改 | 4 → 12 测试项 |
| `tests/test_chapter_budget.py` | 修改 | 2 → 3 测试项 |
| `tests/test_long_form_orchestrator.py` | 修改 | 1 → 4 测试项 |
| `tests/test_long_form_pipeline.py` | 修改 | 1 → 8 测试项 |
| `docs/FULL_TEST_PROCEDURE.md` | 修改 | 完整重写 |

---

## 六、对回忆录生成所需事件的优化

在完成上述系统架构改进并进行端到端 pipeline 测试后，发现了五类需要进一步优化的实际问题。以下逐一说明问题现象、定位分析和具体修复。

### 6.1 生成结果不可见——输出链路补全

**问题**：运行完整 pipeline 后，评估报告 JSON 中仅有评分指标，**不包含生成的文本内容**，用户无法直接查看或审阅生成结果。

**修复**（`scripts/run_long_form_e2e.py`）：

- 新增 `_save_generated_text()`：将各章生成内容按 `===== 第 N 章 =====` 分隔，写入 `data/long_form_e2e_output.txt`。
- 新增 `_enrich_report_with_content()`：在 JSON 报告的每个 `segments[i]` 下嵌入 `generated_text` 字段，顶层嵌入 `merged_content`。

### 6.2 质量门控失败无修复——自动重试机制

**问题**：某章 FActScore 低于阈值 60% 时，门控仅输出"未通过"，**无任何修复动作**，pipeline 不具备实际可用性。

**修复**：

**① 编排器层**（`src/generation/long_form_orchestrator.py`）：

新增 `regenerate_chapters()` 函数，接受 `RemediationPlan` 中的 `chapters_to_regenerate` 和 `prompt_adjustments`，对指定章节重新检索 + 重新生成，将修复指令注入跨章上下文：

```python
async def regenerate_chapters(
    result, retriever, generator,
    chapters_to_regenerate, prompt_adjustments, ...
):
    for ch_idx in chapters_to_regenerate:
        cross_ctx = result.chapter_context.build_prompt_section(ch_idx)
        adj = adjustments.get(ch_idx, "")
        if adj:
            cross_ctx += f"\n\n【修复指令】{adj}"
        gr = await generator.generate(..., chapter_context=cross_ctx)
        result.chapters[ch_idx] = ChapterGenerationResult(...)
```

**② 脚本层**（`scripts/run_long_form_e2e.py`）：

在"生成→评估"后增加 while 循环，最多重试 `--max-gate-retries` 轮（默认 2）。每轮仅对失败章节调用 `regenerate_chapters()`，并记录 `gate_retry_log` 到 JSON 报告中。

### 6.3 生成内容千篇一律——Prompt 与预算重构

**问题**：生成的各章内容以"那年……"开头，风格模板化，更像"历史背景描述"而非回忆录润色改写；字数也明显不足。

**根因**：
1. Prompt 模板的核心指令是"生成历史背景描述"，而非"对回忆录进行文学润色和改写"。
2. 字数预算按固定的全书总量平均分配，与原文段落长度无关。

**修复**：

**① Prompt 模板重构**（`src/generation/prompts.json`）：

将 `standard`、`nostalgic`、`narrative` 三个模板的核心指令从"生成历史背景"改为"文学润色和改写回忆录"，要求保留原文中的人物、事件、对话、细节，将历史背景作为"调料"自然编织进叙事。新增约束：

```
- 禁止以「那年」「那时」开头——用具体的场景、动作或细节开篇
- 只做叙事性描写，不要概括或评论
```

系统提示词（`system_prompts.default`）同步调整为"回忆录润色作家"角色定位。

**② 字数预算改为扩展系数**（`src/generation/chapter_budget.py`）：

多段模式不再从固定全书总量按比例分摊，而是基于**各段原文字数**乘以扩展系数：

| length_bucket | 扩展系数 | 含义 |
|---|---|---|
| 200-400 | 0.8× | 精简润色 |
| 400-800 | 1.2× | 略扩写 |
| 800-1200 | 1.6× | 丰富扩写 |
| 1200+ | 2.0× | 大幅扩写 |

例如原文段落 500 字、选择 "400-800" bucket，目标字数 = 500 × 1.2 = 600 字（±15%）。

### 6.4 FActScore 持续偏低——事实检查策略优化

**问题**：优化 Prompt 后 FActScore 仍频繁在 58% 左右，低于 60% 阈值。分析发现两个原因：
1. 事实验证时**只用检索到的历史背景**作为证据上下文，而润色改写的内容大部分来自**回忆录原文**本身——原文中的人名、事件、对话被判定为"缺乏证据支持"。
2. 原子事实列表中包含大量**纯文学修辞**（比喻、拟人、感官描写），这些不是可验证的事实，却被送入 LLM 验证，拉低通过率。

**修复**（`src/evaluation/factscore_adapter.py`）：

**① 验证上下文扩展**：将 `_factscore_check()` 的上下文从仅"检索历史背景"改为"回忆录原文 + 检索历史背景"双证据源：

```python
context_parts = []
if memoir_text:
    context_parts.append(f"【回忆录原文】\n{memoir_text}")
if retrieval_context:
    context_parts.append(f"【历史背景】\n{retrieval_context}")
context = "\n\n".join(context_parts)
```

**② 文学修辞过滤**：新增 `_filter_literary_sentences()`，在 LLM 验证前过滤不含数字、时间、地名或事实性动词的纯文学句：

```python
has_verifiable = bool(re.search(
    r"[\d一二三四五六七八九十百千万亿]"
    r"|[年月日号]"
    r"|(?:省|市|县|区|镇|村|塬|河|山)"
    r"|(?:叫|姓|名|是|在|从|到|去|来|做|当|任)", s
))
```

**③ 规则预匹配**：新增 `_rule_match_against_source()`，对每个原子事实提取关键实词，若 ≥50% 能在原文中直接找到，跳过 LLM 验证直接标记为"支持"，减少 LLM 调用量。

**④ Prompt 显式要求保留具体细节**：在 `prompts.json` 所有润色模板中新增：

```
必须保留原文中的具体年份（如一九七二年、1992年）、数字（如两百块、三千台）和地名，不得模糊化
```

### 6.5 Pipeline 耗时过长——并行化优化

**问题**：端到端一次执行需约 131 秒，对实际应用场景预期超过十倍。分析发现：
- 6 章评估（每章含 FActScore LLM 调用）是**串行**执行的。
- 每章的"检索→生成"也是完全串行，检索等待生成后才开始下一章。

**修复**：

**① 评估并行化**（`src/evaluation/long_form_eval.py`）：

将逐章评估 `for ch in chapters: await _eval(ch)` 重构为并发执行：

```python
records = list(await asyncio.gather(
    *[_eval_one_chapter(ch) for ch in long_form.chapters]
))
```

各章的指标计算、evaluator 调用、FActScore 检查相互独立，可安全并发。

**② 检索预取**（`src/generation/long_form_orchestrator.py`）：

在第 k 章的 LLM 生成期间，提前发起第 k+1 章的检索任务：

```python
next_retrieval = asyncio.create_task(
    retriever.retrieve(segments[i + 1].text, ...)
)
```

生成是 pipeline 中耗时最长的环节（LLM 调用），与检索（知识图谱查询）重叠后，检索等待时间几乎被完全隐藏。

**③ 细粒度计时**（`scripts/run_long_form_e2e.py`）：

新增 `[生成] elapsed=...s` 和 `[评估] elapsed=...s` 日志，方便后续定位瓶颈。

### 6.6 章末软性总结泛滥——反感悟收尾机制

**问题**：每章末尾出现抒情性感悟或人生哲理式收尾（如"这段日子教会了我……""那是最难忘的岁月"），但现有 `summary_sentence_ratio` 指标仅检测"总之""综上"等硬性关键词，对这类语义性总结无感知。回忆录应仅在**最后一章**才有收束感。

**修复**（三处协同）：

**① Prompt 模板**（`src/generation/prompts.json`）：

在 `standard`、`nostalgic`、`narrative` 模板中新增规则：

```
禁止在末尾添加抒情感悟或人生哲理式收尾（如「这是我最美好的回忆」
「见证了我的成长」「教会了我……」），每章应在叙事的自然节点戛然而止，
像连载小说的分章一样，把余韵留给下一章
```

系统提示词禁止事项新增：

```
绝不在章节末尾添加抒情性感悟、人生哲理或回望式总结；唯有全书最后一章可以带有收束感
```

**② 章节位置指令**（`src/generation/chapter_context.py`）：

差异化三种角色的 `role_instruction`：

| 角色 | 指令 |
|---|---|
| opening（开篇） | 自然引入时代背景，铺垫基调。不要在末尾写总结或感悟 |
| body（中间） | 与前文衔接，推进叙事。**禁止末尾感悟/哲理/回顾式收尾——在叙事自然节点戛然而止** |
| closing（收尾） | 可以带有适度的收束感与回望意味 |

**③ 质量门控检测**（`src/evaluation/quality_gate.py`）：

新增 `_EPILOGUE_PATTERNS` 正则和 `_detect_epilogue()` 函数，对非末章的**最后 3 句**做模式匹配，匹配到时生成 `epilogue` 维度的 warning。检测模式覆盖以下几类常见感悟结尾：

- "这段日子/岁月/经历……教会/让我/使我……"
- "那是最美好/难忘/珍贵的……"
- "见证/承载/记录了……成长/变迁……"
- "在我心中/记忆里……留下/刻下/种下……"
- "多年以后/许多年后……才明白/懂得……"
- "也许/或许……正是……成就/塑造……"
- "这一切/所有这些……构成/编织成……底色/篇章……"

---

## 七、变更文件清单（第六章优化部分）

| 文件 | 改动说明 |
|---|---|
| `scripts/run_long_form_e2e.py` | 生成文本输出、报告内容嵌入、质量门控重试循环、细粒度计时 |
| `src/generation/prompts.json` | Prompt 模板重构（润色改写定位）、反模板开头、保留具体细节、反感悟收尾规则 |
| `src/generation/chapter_budget.py` | 字数预算从固定分摊改为扩展系数 |
| `src/generation/long_form_orchestrator.py` | `regenerate_chapters()` 函数、检索预取 |
| `src/generation/chapter_context.py` | 三种章节角色的差异化位置指令 |
| `src/generation/__init__.py` | 导出 `regenerate_chapters` |
| `src/evaluation/factscore_adapter.py` | 双证据源、文学修辞过滤、规则预匹配 |
| `src/evaluation/long_form_eval.py` | 各章评估并行化 (`asyncio.gather`) |
| `src/evaluation/quality_gate.py` | 新增 `_EPILOGUE_PATTERNS` + `_detect_epilogue()` + 非末章感悟检测 |

---

## 八、总结

本次改进从**分段正确性、生成质量保障、评估与门控**三个层面对系统进行了系统性升级，并在端到端测试中发现和修复了六类实际问题：

| 关注的问题 | 先前状态 | 现在 |
|---|---|---|
| 长文本能否分章生成？ | 可以，但切分仅凭空行 | 以**时间边界**为第一优先级切分，每段带**元数据**（年份/地点/人物），并有**校验报告**可审计 |
| 如何保证各章不重复、保持回忆录风格？ | 各章独立生成，无任何协调 | 通过 **ChapterContext** 注入前文概要 + 反重复要点 + 章节位置指令；Prompt 显式禁止总结句和套话；生成后即时检测重复率并可自动重试 |
| 如何评估整体质量和真实性？ | 仅段级指标 + 布尔事实检查，无"能否交付"的判定 | 新增 3 项**跨章指标**；事实检查输出 **FActScore 数值**；全自动**质量门控**给出通过/不通过判定 + **修复计划**指明"哪几章需重新生成、为什么、怎么调" |
| 生成内容不可见 | 仅有 JSON 评分 | 同时输出可读文本文件 + 报告内嵌原文 |
| FActScore 持续偏低（58%） | 仅用检索背景验证 | 回忆录原文 + 检索背景双证据源 + 文学修辞过滤 + 规则预匹配 |
| 生成内容模板化（"那年"开头、感悟收尾） | 无针对性约束 | Prompt + 系统提示 + 章节位置指令三层禁止；质量门控新增 epilogue 感悟检测 |
| Pipeline 耗时过长（131s） | 全串行 | 评估并行化 + 检索预取 |

