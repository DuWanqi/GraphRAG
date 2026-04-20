# 组会报告：回忆录 RAG 系统——长文分段生成与生成结果检索的逻辑梳理

> 日期：2026-04-16
> 项目：GraphRAG 回忆录历史背景生成系统

---

## 〇、一句话概要

用户输入一篇跨越数十年的回忆录 → 系统**按时间边界分段** → **逐段用段文本去知识图谱检索** → **逐段调 LLM 生成润色文本（生成时注入前文摘要防重复）** → 拼合成完整长文 → **多维度评估 + 质量门控**决定能不能交付，不能就自动重试。

---

## 一、Pipeline 全景：五步走

```
输入：回忆录原文（数千字，跨数十年）
     │
     ▼
 ┌──────────────────────────────────────────┐
 │  STEP 1  分段  segment_memoir()          │ ← memoir_segmenter.py
 │  做什么：把长文切成若干章                  │
 │  怎么切：时间边界 > 标题/空行 > 长度兜底   │
 │  输出  ：List[MemoirSegment]（每段带元数据）│
 └───────────────┬──────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────────────┐
 │  STEP 2  分配字数预算                     │ ← chapter_budget.py
 │  做什么：给每段定一个"生成多少字"的目标     │
 │  怎么算：段原文字数 × 扩展系数（0.8~2.0×） │
 │  输出  ：List[SegmentBudget]              │
 └───────────────┬──────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────────────────────────┐
 │  STEP 3  逐章循环：检索 → 生成 → 记录               │ ← long_form_orchestrator.py
 │                                                      │
 │  对第 k 章，依次做 5 件事（详见第三节）：              │
 │   3a. 用段文本去知识图谱检索                          │
 │   3b. 拼装跨章上下文（前面写了什么、别重复什么）        │
 │   3c. 调 LLM 生成润色文本                             │
 │   3d. 检测和上一章的重复率，超标就重试                  │
 │   3e. 记录本章摘要/要点，供下一章使用                  │
 │                                                      │
 │  输出：LongFormGenerationResult                       │
 │        ├── chapters[]         各章生成结果             │
 │        ├── merged_content     拼合全文                 │
 │        └── segmentation_report 分段校验报告            │
 └───────────────┬──────────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────────────────────────┐
 │  STEP 4  评估  evaluate_long_form()                  │ ← long_form_eval.py
 │                                                      │
 │  做什么：从三个层面打分（详见第四节）                   │
 │  · 段级 8 项指标 + FActScore                          │
 │  · 跨章 3 项指标（重复/风格/总结句）                   │
 │  · 篇级 2 项指标（年代多样性/合并长度）                │
 │                                                      │
 │  怎么做：各章并行评估（asyncio.gather）                │
 │  输出  ：LongFormEvalResult（含 JSON 报告）            │
 └───────────────┬──────────────────────────────────────┘
                 │
                 ▼
 ┌──────────────────────────────────────────────────────┐
 │  STEP 5  质量门控  check_quality_gate()              │ ← quality_gate.py
 │                                                      │
 │  做什么：决定"能不能交付"，不能就给修复方案             │
 │  检查  ：字数/综合分/事实/跨章重复/总结句/套话/感悟结尾│
 │  输出  ：QualityGateResult                            │
 │          ├── passed: true/false                       │
 │          └── remediation: 哪几章要重生成、为什么、怎么改│
 └──────────────────────────────────────────────────────┘
```

**反馈回路**：门控不通过 → 取出 `RemediationPlan` → 调 `regenerate_chapters()` 只重做失败章 → 再评估，最多循环 2 轮。

---

## 二、STEP 1 详解：长文如何分段

> 文件：`src/generation/memoir_segmenter.py`
> 核心函数：`segment_memoir(text) → List[MemoirSegment]`

### 2.1 分段策略（按优先级从高到低）

| 优先级 | 切分信号 | 实现方式 | 举例 |
|---|---|---|---|
| **最高** | 段首出现新年份 | 正则 `_LEADING_YEAR` 匹配"一九七二年""1992年"等 | "一九七二年春天……"开头的段落一定单独成章 |
| 次高 | 章节标题 | 正则 `_CHAPTER_LINE` 匹配"第一章""一、"等 | "第三章 北上求学" |
| 次低 | 空行 | `re.split(r"\n\s*\n+", text)` | 两个连续空行 |
| **兜底** | 长度约束 | 超长块按句号切(`_split_oversized`)；过短块合并(`_merge_short`) | 200 字的段合并到前一段，但**不跨时间边界合并** |

#### "不跨时间边界合并"的具体做法

```python
# _merge_short() 中的关键判断：
crosses_time = respect_temporal and _LEADING_YEAR.match(b)
if len(buf) < target_min_chars and not crosses_time:
    buf = f"{buf}\n\n{b}"   # 合并——当前块太短且下一块不是新年份
else:
    merged.append(buf)       # 不合并——即使当前块只有 200 字
    buf = b
```

效果：讲 1972 年的 200 字和讲 1977 年的 300 字**绝不会被合并成一段**，保证后续检索的年代精度。

### 2.2 每段附带什么元数据

分段完成后，每段自动提取 `SegmentMeta`：

```
SegmentMeta
├── detected_years      ("1972", "1977")       ← 正则 _YEAR_PATTERN 提取所有四位年份
├── detected_locations  ("陕北", "延安")        ← 在预定义地名词表 _LOCATION_KEYWORDS 中做字符串匹配
├── detected_figures    ("老张队长",)           ← 正则 _PERSON_PATTERNS 匹配"老X""X队长"等称谓
├── temporal_label      "1972-1977"            ← 取 min(years)-max(years)
└── split_reason        "temporal_boundary"    ← 说明这个切分是因为什么
```

`split_reason` 四种取值：`document_start` | `temporal_boundary` | `chapter_heading` | `paragraph_break`。

### 2.3 分段质量自动校验

`validate_segmentation()` 对分段结果跑 4 项检查，输出 `SegmentationReport`：

| 检查项 | 严重级别 | 触发条件 |
|---|---|---|
| 段落过短 | warning | `len(段) < min_chars × 0.5` |
| 段落过长 | error | `len(段) > max_chars × 3` |
| 单段年代跨度过大 | warning | 一段内首尾年份差 > 15 年 |
| 段索引不连续 | error | `segment.index ≠ 列表位置` |

---

## 三、STEP 3 详解：逐章"检索→生成"的完整流程

> 文件：`src/generation/long_form_orchestrator.py`
> 核心函数：`run_long_form_generation()`

对每一章，依次执行以下 5 个子步骤：

### 3a. 检索——用段文本查知识图谱

```python
rr = await retriever.retrieve(seg.text, top_k=10, mode="keyword")
```

- **输入**：当前段的回忆录原文（比如讲 1972 年在陕北的 600 字）
- **做什么**：`MemoirRetriever` 从 GraphRAG 知识图谱中检索与这段文本相关的实体、关系、社区报告
- **输出**：`RetrievalResult`，包含 `.entities`（实体列表）、`.communities`（社区摘要）、`.context`（结构化上下文，含 year/location/keywords）

**性能优化——检索预取**：在第 k 章 LLM 生成期间（最耗时），提前 `asyncio.create_task` 发起第 k+1 章的检索，使检索延迟被隐藏：

```python
# 生成第 k 章的同时，预取第 k+1 章的检索
if i + 1 < len(segments):
    next_retrieval = asyncio.create_task(retriever.retrieve(segments[i+1].text, ...))
```

### 3b. 构建跨章上下文——告诉 LLM "前面写了什么"

```python
cross_ctx = chapter_ctx.build_prompt_section(当前章索引)
```

> 文件：`src/generation/chapter_context.py`，类 `ChapterContext`

这一步做三件事，拼成一段文字注入 prompt：

**第一块——前文概要**（最近 3 章的摘要）：
```
## 前文已生成内容概要（请勿重复）
- 第1章 (1972-1977): 知青下乡到陕北张家塬，在生产队学习维修柴油机
- 第2章 (1977-1978): 恢复高考消息传来，在窑洞中复习备考
```

**第二块——反重复要点**（之前所有章的高频短语，最多 12 个）：
```
以下要点已在前文出现，请勿再次展开或总结：
知青、黄土高坡、柴油机、恢复高考、窑洞、煤油灯
```

> **短语提取方案的演进**：反重复要点的质量取决于"高频短语"的提取方式。我们先后尝试了两种方案（详见下方 [附：两种短语提取方案的对比](#附两种短语提取方案的对比字符滑窗-vs-jieba-分词)），最终采用 **jieba 分词方案**，提取出的要点均为有内容区分度的实义词（如"生产队""柴油机""老张"），而非"我们""时候"等通用虚词。

**第三块——章节位置指令**（开篇/中间/结尾差异化）：

| 位置 | 指令内容 |
|---|---|
| 开篇（第 0 章） | "自然引入时代背景，铺垫基调。不要在末尾写总结或感悟" |
| 中间章 | "与前文衔接，推进叙事。禁止末尾感悟/哲理——在叙事自然节点戛然而止" |
| 末章 | "可以带有适度的收束感与回望意味" |

### 3c. 调 LLM 生成——Prompt 的完整结构

```python
gr = await generator.generate(
    memoir_text=seg.text,          # 本段回忆录原文
    retrieval_result=rr,           # 3a 的检索结果
    style="standard",              # 风格模板
    length_hint="425-575字",       # 3b 分配的字数预算
    chapter_context=cross_ctx,     # 3b 构建的跨章上下文
)
```

> 文件：`src/generation/literary_generator.py`，方法 `LiteraryGenerator.generate()`
> 模板文件：`src/generation/prompts.json`

最终发给 LLM 的 prompt 结构如下（以 standard 风格为例）：

```
[System Prompt]
  你是一位优秀的回忆录润色作家。保留原文人物/事件/对话，
  融入历史背景，禁止总结句、套话、"那年"开头、感悟收尾……

[User Prompt]
  一、回忆录原文（必须保留其中的人名、地名、对话、情节）
     {seg.text}

  二、时间背景
     年份：{检索结果解析出的年份}
     地点：{检索结果解析出的地点}

  三、可参考的历史背景信息
     {检索结果的文本摘要——来自知识图谱的实体/社区描述}

  四、跨章上下文（3b 构建的内容）
     {前文概要 + 反重复要点 + 位置指令}

  五、写作要求（11 条硬约束）
     保留原文人物和细节、保留具体年份/数字/地名、
     补充感官描写、字数控制在 {length_hint}、
     禁止总结/套话/那年开头/感悟收尾……
```

### 3d. 重复检测——生成后立即校验

```python
rep_warning = chapter_ctx.detect_repetition_with_previous(gr.content)
```

**实现方式**：提取当前章的高频短语（2-gram/3-gram），与之前所有章累积的短语集合做交集。重叠率 ≥ 15% 时返回警告。

**自动重试**：若检测到重复且 `max_retry_chapters > 0`：
- temperature +0.1（增加随机性）
- 在跨章上下文末尾追加："⚠️ 严格要求：上一次生成与前文高度重叠，请务必从不同角度切入"
- 重新调 LLM 生成

### 3e. 记录——为下一章准备上下文

```python
chapter_ctx.record_chapter(seg.index, gr.content, entities=[检索到的实体名列表])
```

内部做三件事（**均为纯规则提取，不调用 LLM**）：

1. **`_extract_brief(content)`** → 本章摘要（≤60 字）

   不做任何摘要式理解或生成，而是直接按标点截取原文的开头：先以句号/问号/感叹号为分隔符将全文切成句子列表，取第 1 句；若第 1 句不足 20 字且还有后续句，则再拼接第 2 句；若拼接结果超过 60 字则截断并加省略号。示意：

   ```
   生成文本："一九七二年春天，父亲带着全家来到陕北。张家塬的窑洞嵌在黄土坡上。生产队长老张……"
                          ↓ 按"。"切句
   句子列表：["一九七二年春天，父亲带着全家来到陕北",
             "张家塬的窑洞嵌在黄土坡上",
             "生产队长老张……"]
                          ↓ 取第 1 句（22 字，≥20 字，够用）
   摘要：    "一九七二年春天，父亲带着全家来到陕北"
   ```

   选择规则截取而非 LLM 摘要，是因为此步发生在**每章生成完毕后的关键路径上**（第 k 章记录 → 第 k+1 章立即使用），调 LLM 会引入额外延迟和成本；而且前文概要只需要让 LLM 知道"第 k 章大概在讲什么"，首句截取已足够。

2. **`_extract_time_period(content)`** → 提取年份范围，如 "1972-1977"

   用正则从全文提取所有四位年份（中文"一九七二年"和阿拉伯"1972年"均可识别），取 min-max 拼成范围字符串。

3. **`_extract_key_phrases(content)`** → **jieba 分词 → 停用词过滤 → 词级 1-gram + 相邻词 2-gram 统计**，取频次≥2 的 top 8

这些信息存入 `ChapterRecord`，下一章的 3b 会用到。

#### 附：两种短语提取方案的对比（字符滑窗 vs jieba 分词）

`_extract_key_phrases` 负责从每章生成文本中提取"高频实义短语"，供反重复要点和跨章重复检测使用。我们先后尝试了两种实现。下面先分别说明原理，再用同一段文本做直观对比。

##### 方案 A：字符级滑动窗口（旧，已弃用，保留为 `_extract_key_phrases_char_sliding`）

**原理**：不做分词，先将文本中所有非汉字字符（标点、数字、空格、英文）全部删除，得到一串连续汉字。然后用一个固定宽度的窗口（n=2 和 n=3）在这串汉字上逐字符滑动，每滑一步取出窗口内的字符片段作为一个 n-gram，统计每个 n-gram 出现的次数，最后按频次排序取 top-k。

```
原文片段：  "生产队里分了二十亩旱地，父亲每天天不亮就出门"
             ↓ 去掉标点
连续汉字串："生产队里分了二十亩旱地父亲每天天不亮就出门"
             ↓ 滑动窗口 n=2
 2-gram：   生产 / 产队 / 队里 / 里分 / 分了 / 了二 / 二十 / 十亩 / 亩旱 / 旱地 / 地父 / 父亲 / ...
             ↓ 滑动窗口 n=3
 3-gram：   生产队 / 产队里 / 队里分 / 里分了 / 分了二 / ...
```

**过滤手段**：仅用 6 个硬编码虚词组合（"的是""了的""在了""是的""和的""不是"）做黑名单过滤，其余 n-gram 全部保留参与计数。

**缺陷**：

| 缺陷 | 从上例可以看到 | 影响 |
|---|---|---|
| **跨词边界噪声** | "产队""队里""里分""地父" 都不是有意义的词——它们跨越了词的边界 | 反重复提示被无意义碎片占据 |
| **停用词覆盖极小** | "分了""了二" 等虚词组合不在 6 条黑名单中，全部漏网；"我们""时候" 等高频通用词同理 | 提示 LLM "请勿再次展开'我们、时候'" 毫无意义 |
| **子串冗余计数** | "生产队" 同时产生 "生产"(2-gram) + "产队"(2-gram) + "生产队"(3-gram) 三条计数 | top-8 名额被同一个词的碎片占满 |

根本原因：中文是无空格语言，字符滑窗无法感知"词"在哪里结束、下一个"词"从哪里开始，所以会大量产出跨越词边界的无意义片段。

##### 方案 B：jieba 分词 + 停用词表（当前采用）

**原理**：先用 jieba 分词器将文本切分为**词**的序列（jieba 基于前缀词典 + 动态规划找最大概率切分路径，对中文有良好的词边界识别能力）。分词后逐词过滤：只保留 ≥2 个汉字的中文词，并排除 `_STOPWORDS` 停用词表中的 80+ 个高频通用词（代词、虚词、泛化时间词、趋向补语等）。过滤后，统计词级 1-gram（单词本身）和词级 2-gram（相邻词拼接），按频次取 top-k。

```
原文片段：  "生产队里分了二十亩旱地，父亲每天天不亮就出门"
             ↓ jieba 分词
词序列：    生产队 / 里 / 分 / 了 / 二十 / 亩 / 旱地 / ， / 父亲 / 每天 / 天不亮 / 就 / 出门
             ↓ 过滤：只保留 ≥2 字中文词 + 排除停用词
保留词：    生产队 / 旱地 / 父亲 / 每天 / 天不亮 / 出门
             ↓ 统计
 词级 1-gram：  生产队(1)  旱地(1)  父亲(1)  每天(1)  天不亮(1)  出门(1)
 词级 2-gram：  生产队旱地(1)  旱地父亲(1)  父亲每天(1)  每天天不亮(1)  天不亮出门(1)
```

**三层过滤机制**：

| 过滤层 | 做什么 | 效果 |
|---|---|---|
| **jieba 分词** | 按词边界切分，天然避免跨词碎片 | "生产队" 是一个完整的词，不会拆出 "产队""队里" |
| **正则 `[\u4e00-\u9fa5]{2,}`** | 过滤单字词（"了""的""在""里""分"）、标点、数字 | 单字助词/介词/量词全部排除 |
| **`_STOPWORDS` 停用词表（80+ 条）** | 过滤代词（"我们""他们"）、虚词（"因为""可以"）、泛化时间词（"时候""当时"）、趋向补语（"起来""出来"）等 | 只保留有内容区分度的实义词 |

##### 直观对比：同一段文本的完整处理过程

测试输入（约 500 字，讲述 1972 年到陕北张家塬的经历）：

> 一九七二年春天，父亲带着我们一家从县城出发，坐了一天一夜的长途汽车，终于到了陕北的张家塬。那时候的张家塬还是一片黄土高坡，窑洞零零散散地嵌在山腰上。生产队长老张是个黑脸汉子，说话嗓门大得像打雷。他带我们去看分配的窑洞，窑洞里头黑洞洞的，只有一盏煤油灯。母亲叹了口气，把行李一件件搬进去。那年冬天特别冷，黄土高坡上的风像刀子一样割脸。生产队里分了二十亩旱地，父亲每天天不亮就出门，跟着老张学种糜子。我们几个孩子也不闲着，放学后就去打猪草。那时候日子虽然苦，但大家都是这样过的。后来生产队买了一台柴油机，父亲因为在县城学过一点机械，就被派去学习维修柴油机。

**单章提取结果**：

| 排名 | 方案 A 字符滑窗 | 方案 B jieba 分词 |
|---|---|---|
| 1 | **我们** (freq=5) | 父亲 (freq=4) |
| 2 | 生产队 (freq=4) | 窑洞 (freq=4) |
| 3 | 父亲 (freq=4) | 老张 (freq=3) |
| 4 | **时候** (freq=4) | 生产队 (freq=3) |
| 5 | 窑洞 (freq=4) | 柴油机 (freq=3) |
| 6 | **生产** (freq=4) | 县城 (freq=2) |
| 7 | **产队** (freq=4) | 张家 (freq=2) |
| 8 | **那时候** (freq=3) | 黄土 (freq=2) |

（加粗项 = 通用词或碎片，对反重复提示无意义）

| 维度 | 方案 A | 方案 B |
|---|---|---|
| 通用词污染 | "我们""时候""那时候"占 3/8 席 (37.5%) | 0 个 (0%) |
| 碎片/冗余 | "生产""产队"是"生产队"的子串，占 2/8 席 | 无碎片 |
| **有效要点率** | **3/8 = 37.5%** | **8/8 = 100%** |

**跨章场景对比**（第 1 章讲到陕北，第 2 章讲恢复高考）：

要看懂下表需要先区分系统中"反重复"的**两个独立环节**：

- **预防**（3b 步，生成**前**）：把前面所有章的**全部**要点注入当前章 prompt，告诉 LLM"这些前面写过了，别重复"。
- **检测**（3d 步，生成**后**）：提取当前章要点，与前面章的要点做**交集**——交集里的词就是"尽管提醒了，还是重复了"的内容。

| | 方案 A | 方案 B |
|---|---|---|
| 第 1 章提取的要点 | 我们, 生产队, 父亲, 时候, 窑洞, 生产, 产队, 张家塬 | 父亲, 窑洞, 老张, 生产队, 柴油机, 县城, 张家, 黄土 |
| ❶ **预防**：生成第 2 章前注入 prompt | "以下要点已在前文出现，请勿再次展开：我们、生产队、父亲、时候、窑洞、生产、产队、张家塬" — 其中"我们""时候"等通用词占了近半席位，浪费 token 且无实际约束力 | "以下要点已在前文出现，请勿再次展开：父亲、窑洞、老张、生产队、柴油机、县城、张家、黄土" — 全部是实义词，精准告知 LLM 哪些叙事元素已有充分描写 |
| 第 2 章生成后提取的要点 | 窑洞里, 窑洞, 洞里, 生产队, 煤油灯, 生产, 产队, 母亲 | 窑洞, 生产队, 母亲, 煤油灯, 复习 |
| ❷ **检测**：第 1 章 ∩ 第 2 章 | {生产, 产队, 生产队, 窑洞} — 4 个，但"生产""产队"是"生产队"的碎片，实际只有 2 个真实重叠 | {生产队, 窑洞} — 2 个，均为真实跨章重叠 |

> **为什么交集中没有"母亲""煤油灯""复习"？**
> 因为它们是第 2 章**新引入**的内容，第 1 章没有出现过，所以不构成"跨章重复"——这恰恰是正常的叙事推进。但它们会被记录到 `_all_key_phrases` 中，等到生成第 3 章时，就会被注入第 3 章的 prompt，防止第 3 章再次重复第 2 章的"母亲""煤油灯""复习"。

如果有第 3 章（讲改革开放），预防环节注入的内容会是前两章的全部要点（取最近 12 个）：

| | 方案 A | 方案 B |
|---|---|---|
| ❶ 生成第 3 章前注入 prompt | "请勿再次展开：窑洞, 生产, 产队, 张家塬, 窑洞里, 洞里, 生产队, 煤油灯, 产队, 母亲, …" — 大量碎片和重复 | "请勿再次展开：老张、生产队、柴油机、县城、张家、黄土、窑洞、母亲、煤油灯、复习" — 前两章的实义要点一目了然 |

##### 小结

| | 方案 A 字符滑窗 | 方案 B jieba 分词 |
|---|---|---|
| 分词依赖 | 无（纯字符操作） | 需要 jieba（已在项目依赖中） |
| 词边界感知 | 无 — 跨词碎片多 | 有 — 基于词典+统计的分词 |
| 停用词过滤 | 6 条硬编码 | 80+ 条，覆盖 6 大类 |
| 有效要点率 | ~37.5% | ~100% |
| 对 prompt 的实际价值 | 低 — 大量无意义词浪费 token、可能误导 LLM | 高 — 每个要点都指向具体的叙事元素 |

---

## 四、STEP 4 详解：评估怎么做

> 文件：`src/evaluation/long_form_eval.py`
> 核心函数：`evaluate_long_form()`

### 4.1 段级指标（每章独立计算）

| 指标 | 文件/类 | 计算方式 |
|---|---|---|
| `entity_coverage` | metrics.py / AccuracyMetrics | 检索到的实体有多少出现在生成文本中 |
| `time_consistency` | metrics.py / AccuracyMetrics | 生成文本是否包含检索上下文中的参考年份 |
| `keyword_overlap` | metrics.py / RelevanceMetrics | 检索关键词在生成文本中的匹配数 |
| `semantic_similarity` | metrics.py / RelevanceMetrics | 回忆录原文 vs 生成文本的 **2-gram 余弦相似度** |
| `length_score` | metrics.py / LiteraryMetrics | 生成字数是否在预算区间内 |
| `paragraph_structure` | metrics.py / LiteraryMetrics | 段落数量是否合理 |
| `transition_usage` | metrics.py / LiteraryMetrics | 是否使用了"当时""彼时"等过渡词 |
| `descriptive_richness` | metrics.py / LiteraryMetrics | 是否有感官词、比喻、情感词 |
| **FActScore** | factscore_adapter.py | 见下方 4.2 |

综合分 = 8 项指标的加权平均 × 10（归一化到 0-10 分）。

### 4.2 FActScore 事实检查的三步流程

> 文件：`src/evaluation/factscore_adapter.py`，类 `FActScoreChecker`

```
生成文本
   │
   ▼
Step A: 原子事实分解
   规则拆分（按句号切+过滤太短句）或 LLM 拆分
   → ["十一届三中全会于1978年召开", "作者在窑洞中复习", ...]
   │
   ▼
Step B: 过滤纯文学修辞
   _filter_literary_sentences()：只保留含数字/时间词/地名/事实性动词的句子
   → 去掉"仿佛黄土高坡在低吟"这类不可验证的修辞
   │
   ▼
Step C: 逐条验证（两条路径）
   ├─ 快速路径：_rule_match_against_source()
   │    提取事实中的关键实词，≥50% 能在回忆录原文中找到 → 直接标记"支持"
   │
   └─ LLM 路径：_verify_facts_batch()（剩余未匹配的事实）
        上下文 = 回忆录原文 + 检索到的历史背景（双证据源）
        每 5 条一批发给 LLM 判断 supported: true/false
   │
   ▼
FActScore = 被支持的事实数 / 总事实数
```

### 4.3 跨章指标（全文层面）

| 指标 | 计算方式 | 直觉含义 |
|---|---|---|
| `inter_chapter_repetition` | 相邻两章的 6-gram 集合交集 / min(两集合大小)，取平均。评分 = 1 - 平均重叠率 | 各章之间是否在重复说同一件事 |
| `style_consistency` | 各章"平均句长"的变异系数 CV。CV≤0.2→满分，CV≥0.6→0.3 分 | 全文风格是否统一（而非忽长忽短） |
| `summary_sentence_ratio` | 含"总之/综上/总的来说"等词的句子占比。评分 = 1 - 占比×5 | 回忆录不该像论文一样充斥总结句 |

### 4.4 并行化

各章评估相互独立，用 `asyncio.gather` 并发执行：

```python
records = list(await asyncio.gather(*[_eval_one_chapter(ch) for ch in long_form.chapters]))
```

---

## 五、STEP 5 详解：质量门控怎么判定

> 文件：`src/evaluation/quality_gate.py`
> 核心函数：`check_quality_gate()`

### 5.1 逐章检查（7 个维度）

门控函数 `check_quality_gate()` 接收每章的生成文本、评估综合分、FActScore、目标字数等参数，对每章逐一执行以下检查。所有检查均为**纯规则判定，不调用 LLM**（综合分和 FActScore 是由上游 STEP 4 评估阶段计算好后传入的数值，门控本身只做阈值比较）。

#### 5.1a 字数检查

**输入**：`len(章节文本)` 和 `target_chars_per_chapter[idx]`（由 STEP 2 字数预算分配）。

**做法**：计算 `ratio = 实际字数 / 目标字数`，与上下限比较。

```
ratio < 0.4  → error  "字数 180 远低于目标 500 (比率 36%)"
ratio > 2.5  → warning"字数 1400 远超目标 500 (比率 280%)"
```

注意：若调用方未传入 `target_chars_per_chapter`（比如单元测试中），则跳过此项检查。

#### 5.1b 综合分检查

**输入**：`segment_scores[idx]`——由 STEP 4 评估阶段计算的段级综合分（0-10 分）。

> 综合分的计算不在门控内部，而在 `evaluate_long_form()` 中完成：对每章的 8 项段级指标（entity_coverage、time_consistency、keyword_overlap、semantic_similarity、length_score、paragraph_structure、transition_usage、descriptive_richness）做加权平均，再 × 10 归一化到 0-10。门控只是把这个现成的数值与阈值做比较。

**做法**：`score < 5.0 → error`。

#### 5.1c FActScore 检查

**输入**：`fact_scores[idx]`——由 STEP 4 中 `FActScoreChecker` 计算的事实支持率（0-1）。

> FActScore 的计算流程见第四节 4.2：原子事实分解 → 文学修辞过滤 → 规则预匹配 + LLM 验证 → 被支持事实数 / 总事实数。门控同样只是拿到现成数值做阈值比较。

**做法**：`fact_score < 0.60 → error`。

#### 5.1d 总结句检查

**输入**：章节生成文本。

**做法**（函数 `_detect_summary_sentences`）：

1. 按句号/问号/感叹号将全文切成句子列表
2. 逐句用正则 `_SUMMARY_PATTERNS` 匹配，命中即算一个总结句。正则覆盖 15 个模式：
   ```
   总之 | 综上 | 总而言之 | 总的来说 | 总体而言 | 回顾 | 纵观 |
   概括地说 | 简而言之 | 由此可见 | 可以看出 | 不难看出 |
   归根结底 | 一言以蔽之
   ```
3. 总结句占比 = 命中句数 / 总句数
4. `占比 > 0.30 → warning`

#### 5.1e 套话检查

**输入**：章节生成文本。

**做法**（函数 `_detect_boilerplate`）：

用正则 `_BOILERPLATE_PATTERNS` 在全文中做 `findall`，匹配 8 个空泛模板语：

```
在这个大背景下 | 在这样的时代背景下 | 在这种情况下 | 在这一时期 |
正是在这样的 | 值得一提的是 | 不可否认的是 | 众所周知
```

`命中数 ≥ 2 → warning`（偶尔出现一次可以容忍，出现两次以上说明是模板化生成）。

#### 5.1f 非末章感悟收尾检查

**输入**：章节生成文本 + 该章是否为末章。

**做法**（函数 `_detect_epilogue`）：

1. 仅对**非末章**执行（末章允许有适度的收束感）
2. 取全文**最后 3 句**
3. 逐句用正则 `_EPILOGUE_PATTERNS` 匹配 7 类"软性感悟/人生哲理式收尾"模式：

   | 模式类别 | 示例 |
   |---|---|
   | 经历教会了我 | "这段日子**教会**了我……""这些岁月**让我**明白……" |
   | 最+形容词+的 | "那是**最难忘的**岁月""这是一生中**最珍贵的**经历" |
   | 见证/承载了 | "**见证了**我的成长""**承载了**那段青春" |
   | 在心中留下 | "在我**心底留下**了深深的烙印" |
   | 多年以后才 | "**多年以后**我才**明白**那段日子的意义" |
   | 也许正是…成就 | "**也许正是**那些苦难**塑造**了后来的我" |
   | 构成/汇成…底色 | "**这一切构成**了我人生的**底色**" |

4. `匹配到任一模式 → warning`

#### 汇总

| 维度 | 检测方式 | 默认阈值 | 严重级别 |
|---|---|---|---|
| 字数过短 | `len(文本) / 目标字数 < 阈值` | < 0.4 | error |
| 字数过长 | `len(文本) / 目标字数 > 阈值` | > 2.5 | warning |
| 综合分过低 | STEP 4 段级评分 < 阈值 | < 5.0 / 10 | error |
| FActScore 过低 | STEP 4 事实支持率 < 阈值 | < 60% | error |
| 总结句过多 | 正则匹配句数 / 总句数 > 阈值 | > 30% | warning |
| 套话过多 | 正则匹配命中数 ≥ 阈值 | ≥ 2 处 | warning |
| 非末章感悟收尾 | 末尾 3 句正则匹配（仅非末章） | 命中即触发 | warning |

### 5.2 跨章检查

相邻两章 6-gram 重叠率 > 20% → error。

### 5.3 判定逻辑

- 只要有一项 error → `passed = False`
- `passed = False` 时自动生成 `RemediationPlan`：

```python
RemediationPlan:
    chapters_to_regenerate: [3, 4]          # 需要重做的章节
    reasons:  {3: ["综合评分4.2 < 5.0"]}   # 每章的问题列表
    prompt_adjustments: {3: "检查检索结果质量或切换 LLM"}  # 建议的 prompt 调整
```

### 5.4 自动重试闭环

`regenerate_chapters()` 函数（在 `long_form_orchestrator.py` 中）：

```python
for ch_idx in chapters_to_regenerate:
    # 1. 重新检索
    rr = await retriever.retrieve(段文本, ...)
    # 2. 构建跨章上下文 + 追加修复指令
    cross_ctx = chapter_context.build_prompt_section(ch_idx)
    cross_ctx += f"\n\n【修复指令】{prompt_adjustments[ch_idx]}"
    # 3. 重新生成（temperature 略升 +0.05）
    gr = await generator.generate(..., chapter_context=cross_ctx)
    # 4. 替换旧结果
    result.chapters[ch_idx] = new_result
```

E2E 脚本 (`scripts/run_long_form_e2e.py`) 在"生成→评估→门控"后进入 while 循环，最多重试 2 轮，每轮只重做失败章节。

---

## 六、先前版本的三个核心问题及对应修复

| 问题 | 具体现象 | 修复措施 | 对应代码 |
|---|---|---|---|
| 分段不按年代切 | 1972 年和 1977 年的两个短段被合并成一段，检索精度下降 | `_merge_short()` 中加入 `_LEADING_YEAR.match(b)` 时间边界保护，不跨年代合并 | `memoir_segmenter.py` |
| 各章独立生成导致重复/套话/节奏割裂 | 第 3、4 章都写"改革开放大潮"；每章末尾"总之……" | 新增 `ChapterContext` 注入前文摘要+反重复要点+位置指令；prompt 硬禁总结句和套话；反重复要点短语提取从字符滑窗改为 **jieba 分词+停用词过滤**，消除通用虚词污染 | `chapter_context.py`, `prompts.json` |
| 评估无法回答"能不能交付" | 只有段级指标+布尔事实检查，无篇级评估、无门控 | 新增跨章指标 (`CrossChapterMetrics`)、质量门控 (`check_quality_gate`)、修复计划 (`RemediationPlan`) | `metrics.py`, `quality_gate.py`, `long_form_eval.py` |

---

## 七、端到端测试中发现并修复的六个实际问题

### 7.1 生成结果不可见

**现象**：JSON 报告只有评分，看不到生成的文本。
**修复**：`scripts/run_long_form_e2e.py` 新增 `_save_generated_text()` 输出可读文本文件；`_enrich_report_with_content()` 在 JSON 报告中嵌入每章原文。

### 7.2 门控失败无修复动作

**现象**：FActScore < 60% 时只报"未通过"，pipeline 停在那。
**修复**：编排器新增 `regenerate_chapters()` 接受修复计划；E2E 脚本加 while 循环最多重试 2 轮。

### 7.3 生成千篇一律（"那年"开头、像历史课文）

**根因**：Prompt 核心指令是"生成历史背景描述"；字数预算是全书总量均摊。
**修复**：
- Prompt 改为"文学润色和改写回忆录"，要求保留人物/事件/对话，历史背景做"调料"
- 字数预算改为 **段原文字数 × 扩展系数**（`chapter_budget.py`）

### 7.4 FActScore 持续偏低（~58%）

**根因**：验证上下文只有检索到的历史背景，回忆录原文中的人名/对话被判"无据"；纯文学修辞也被当事实验证。
**修复**（`factscore_adapter.py`）：
- 验证上下文改为 **原文 + 检索背景** 双证据源
- 新增 `_filter_literary_sentences()` 过滤不可验证的修辞句
- 新增 `_rule_match_against_source()` 快速规则匹配跳过原文可证的事实

### 7.5 Pipeline 耗时过长（131s）

**修复**：
- 评估并行化：`asyncio.gather` 各章同时评
- 检索预取：第 k 章生成时 `asyncio.create_task` 预取第 k+1 章的检索

### 7.6 章末软性感悟泛滥

**现象**：每章末尾出现"这段日子教会了我""那是最难忘的岁月"，但 `summary_sentence_ratio` 只检测"总之""综上"这类硬关键词，对感悟无感知。
**修复**（三处协同）：
- **Prompt 模板**：新增规则"禁止感悟收尾，在叙事自然节点戛然而止"
- **ChapterContext**：中间章的位置指令明确禁止感悟/哲理收尾；仅末章允许收束
- **质量门控**：新增 `_EPILOGUE_PATTERNS` 正则 + `_detect_epilogue()` 检测非末章最后 3 句

---

## 八、变更文件清单

### 架构改进部分

| 文件 | 类型 | 改动 |
|---|---|---|
| `src/generation/memoir_segmenter.py` | 改 | 时间边界检测、SegmentMeta、SegmentationReport |
| `src/generation/chapter_context.py` | **新增** | 跨章上下文管理器 |
| `src/generation/long_form_orchestrator.py` | 改 | 五步循环 + 跨章上下文 + 检索预取 + `regenerate_chapters()` |
| `src/generation/literary_generator.py` | 改 | 新增 `chapter_context` 参数 |
| `src/generation/prompts.json` | 改 | 润色改写定位 + `{chapter_context}` 占位符 + 全部反约束 |
| `src/generation/chapter_budget.py` | 改 | 字数预算从固定分摊改为扩展系数 |
| `src/evaluation/metrics.py` | 改 | `semantic_similarity` 升级为 2-gram + 新增 `CrossChapterMetrics` |
| `src/evaluation/quality_gate.py` | **新增** | 质量门控 + 修复计划 + 感悟检测 |
| `src/evaluation/long_form_eval.py` | 改 | 跨章指标 + 门控集成 + 评估并行化 |
| `src/evaluation/factscore_adapter.py` | 改 | 双证据源 + 文学修辞过滤 + 规则预匹配 |
| `scripts/run_long_form_e2e.py` | 改 | 文本输出 + 报告嵌入 + 门控重试循环 + 计时 |

### 测试部分（17 → 36 项，全部通过，无外部依赖）

| 模块 | 新增测试 | 验证什么 |
|---|---|---|
| 分段器 | `test_temporal_boundary_split` | 不同年份不合并 |
| | `test_year_extraction_chinese/arabic` | 中英文年份提取 |
| | `test_segment_meta_populated/locations` | 元数据正确填充 |
| | `test_validate_segmentation_pass/detects_issues` | 校验报告 |
| | `test_real_sample_segmentation` | 真实样本分段合理性 |
| 编排器 | `test_orchestrator_has_segmentation_report` | 结果含校验报告 |
| | `test_orchestrator_cross_chapter_context` | 跨章上下文注入 |
| | `test_orchestrator_disabled_cross_chapter` | 可关闭跨章上下文 |
| 评估 | `test_evaluate_long_form_has_cross_chapter_metrics` | 含跨章指标 |
| | `test_evaluate_long_form_has_quality_gate` | 含质量门控 |
| 门控 | `test_quality_gate_pass/detects_repetition/detects_summary_sentences` | 门控各维度 |
| | `test_quality_gate_remediation_plan` | 低分章出现在修复计划中 |

---

## 九、一页纸总结

| 关注的问题 | 改进前 | 改进后 |
|---|---|---|
| 长文怎么分段？ | 只靠空行和标题 | **时间边界优先**切分 + 每段带元数据 + 校验报告 |
| 各章怎么生成？ | 每章独立调 LLM，互不知情 | 逐章序列生成，`ChapterContext` 注入前文摘要 + 反重复要点（jieba 分词+停用词过滤） + 位置指令 |
| 检索怎么做？ | 用段文本查知识图谱，串行 | 同样用段文本查，但**预取下一章的检索**与当前章生成并行 |
| 怎么评估？ | 仅段级指标 + 布尔事实检查 | 段级 8 指标 + FActScore 数值 + 跨章 3 指标 + 篇级 2 指标 |
| 能不能交付？ | 没有判定机制 | **质量门控** 7 维度检查 → 通过/不通过 + 修复计划 + 自动重试 |
| FActScore 偏低 | 只用检索背景验证 | **双证据源**（原文+背景）+ 文学修辞过滤 + 规则预匹配 |
| 风格模板化 | 无约束 | Prompt + 系统提示 + 位置指令三层禁止总结/套话/感悟 |
| 速度 | 全串行 131s | 评估并行 + 检索预取 |

