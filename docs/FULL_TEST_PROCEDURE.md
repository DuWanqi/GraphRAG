# 生成模块完整测试流程

> 适用环境：当前服务器 `/data/jiabao/tempRAG/GraphRAG`
> Python: `/data/jiabao/miniconda3/bin/python` (3.13)
> Ollama: `/data/jiabao/RAG/Ollama/bin/ollama`
> 默认 LLM: gemini (gemini-2.5-flash) — 本流程使用 OpenAI (gpt-5) 验证

---

## 阶段零：Ollama 启动

索引检索依赖 Ollama 提供的 `nomic-embed-text` embedding 服务。
每次机器重启后 Ollama 需要手动拉起。

```bash
cd /data/jiabao/tempRAG/GraphRAG

# 方式一：用项目自带脚本（会自动启动 Ollama + Web 应用）
bash start.sh --ollama_path /data/jiabao/RAG/Ollama --restart_ollama

# 方式二：只启动 Ollama 服务（不启动 Web）
export OLLAMA_MODELS=/data/jiabao/RAG/Ollama/models
nohup /data/jiabao/RAG/Ollama/bin/ollama serve > /tmp/ollama.log 2>&1 &
```

### 验证 Ollama 就绪

```bash
# 等待几秒后检查
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

期望输出包含 `nomic-embed-text` 模型。如果没有：

```bash
/data/jiabao/RAG/Ollama/bin/ollama pull nomic-embed-text
```

**本步做什么：** 启动 embedding 向量化服务（检索的基础设施）。
**期望结果：** API 返回的模型列表包含 `nomic-embed-text`。
**说明什么：** 向量检索服务就绪，后续阶段三～六可正常运行。

---

## 阶段一：单元测试（无需 LLM / Ollama / 索引）

验证分段、预算、编排、评估、质量门控的核心逻辑。

```bash
cd /data/jiabao/tempRAG/GraphRAG

/data/jiabao/miniconda3/bin/python -m pytest tests/ -v
```

**期望**：36 项全部 PASSED。

### 测试清单与每项意义

| 测试文件 | 测试项 | 在验证什么 | 如果失败说明什么 |
|---|---|---|---|
| **test_memoir_segmenter.py** | | | |
| | `test_empty` | 空输入不崩溃 | 边界处理有 bug |
| | `test_short_single_segment` | 短文返回单段 | 短文被错误拆分 |
| | `test_paragraph_split` | 空行/标题切分 | 结构识别失效 |
| | `test_indices_sequential` | 索引连续性 | 后续编排章节编号会混乱 |
| | `test_temporal_boundary_split` | **不同年份的段落不被合并** | 时间边界感知失效，可能把"1972年插队"和"1977年高考"合成一章 |
| | `test_year_extraction_chinese` | 中文年份 "一九八八年"→"1988" | 年份解析失效，元数据不准 |
| | `test_year_extraction_arabic` | 阿拉伯年份 "1992"→"1992" | 同上 |
| | `test_segment_meta_populated` | 每段都有元数据 (SegmentMeta) | 元数据缺失，评估和报告无法工作 |
| | `test_segment_meta_locations` | 元数据能提取地名 | 检索上下文不准 |
| | `test_validate_segmentation_pass` | 正常分段通过校验 | 校验逻辑误报 |
| | `test_validate_segmentation_detects_issues` | 过长段被报告为 error | 校验逻辑漏报 |
| | `test_real_sample_segmentation` | 真实样本分 4～10 段，各有元数据 | 分段粒度不合理 |
| **test_chapter_budget.py** | | | |
| | `test_allocate_proportional` | 长段分到更多生成字数 | 字数预算失衡 |
| | `test_legacy_maps` | 单段模式参数映射正确 | Web 界面单段模式会出错 |
| | `test_allocate_with_meta` | 带元数据的 Segment 不影响预算 | 新旧代码不兼容 |
| **test_long_form_orchestrator.py** | | | |
| | `test_orchestrator_multi_segment_long_memoir` | 段数=检索次数=生成次数，合并含分隔符 | 编排流程断裂 |
| | `test_orchestrator_has_segmentation_report` | 结果包含分段校验报告 | 无法审计分段质量 |
| | `test_orchestrator_cross_chapter_context` | 第一章无前文概要但有开篇指令；第二章起有前文概要 | 跨章上下文注入失效，各章独立生成会导致重复 |
| | `test_orchestrator_disabled_cross_chapter` | 可关闭跨章上下文 | 旧行为不兼容 |
| **test_long_form_pipeline.py** | | | |
| | `test_evaluate_long_form_pipeline_completes` | 评估流程完整走通 | 评估框架结构错误 |
| | `test_evaluate_long_form_has_cross_chapter_metrics` | 包含跨章指标：重复度、风格一致性、总结句比率 | 篇级质量不可衡量 |
| | `test_evaluate_long_form_has_quality_gate` | 包含质量门控结果 | 无法判定是否达到交付标准 |
| | `test_evaluate_long_form_json_has_fact_score` | JSON 输出包含 fact_score | 事实检查结果不完整 |
| | `test_quality_gate_pass` | 正常内容通过门控 | 门控误杀 |
| | `test_quality_gate_detects_repetition` | 完全重复的两章触发失败 | 跨章重复检测失效 |
| | `test_quality_gate_detects_summary_sentences` | 总结性语句过多产生警告 | 风格问题不可发现 |
| | `test_quality_gate_remediation_plan` | 低分章节出现在修复计划中 | 不知道该修哪里 |
| **test_basic.py** | | | |
| | 9 项 | 解析器、指标、数据加载、配置、provider 枚举 | 基础设施损坏 |

---

## 阶段二：环境与配置检查

### 2.1 Python 依赖

```bash
/data/jiabao/miniconda3/bin/pip list | grep -iE "pandas|pydantic-settings|graphrag|pyarrow|lancedb|litellm"
```

必须存在：`pandas`、`pydantic-settings`、`pyarrow`、`litellm`。

**说明什么：** 这些是索引读取(parquet)、配置管理、LLM 调用的运行时依赖。缺任何一个对应功能 ImportError。

### 2.2 `.env` 配置确认

```bash
cat .env | grep -E "^(OPENAI_|DEFAULT_|GRAPHRAG_|OLLAMA_)" 
```

| 变量 | 当前值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | `sk-maX0u...` | OpenAI（代理）密钥 |
| `OPENAI_BASE_URL` | `https://api.shubiaobiao.cn/v1/` | 代理地址 |
| `GRAPHRAG_OUTPUT_DIR` | `./data/graphrag_output` | 索引目录（已有索引） |
| `DEFAULT_LLM_PROVIDER` | `gemini` | 默认 provider（不影响本流程） |

**说明什么：** LLM API 的"接线"是否正确——key/url 不对，所有 LLM 请求都会 401/404。

### 2.3 OpenAI 适配器可用性

```bash
/data/jiabao/miniconda3/bin/python -c "
from src.llm import create_llm_adapter
a = create_llm_adapter(provider='openai', model='gpt-5')
print(f'provider: {a.provider}')
print(f'model:    {a._get_model()}')
print(f'api_base: {a.api_base}')
print('OpenAI gpt-5 适配器就绪')
"
```

**说明什么：** 适配器工厂能根据 provider 正确创建实例，后续生成/评估可调用 LLM。

---

## 阶段三：索引验证

### 3.1 检查索引就绪

```bash
/data/jiabao/miniconda3/bin/python -c "
from src.retrieval import MemoirRetriever
r = MemoirRetriever()
print('索引就绪:', r.is_index_ready())
"
```

**期望**：`索引就绪: True`。

### 3.2 查看索引统计

```bash
/data/jiabao/miniconda3/bin/python -c "
import pandas as pd
from pathlib import Path
out = Path('data/graphrag_output/output')
for f in sorted(out.glob('*.parquet')):
    df = pd.read_parquet(f)
    print(f'{f.name}: {len(df)} 条')
"
```

| 文件 | 记录数 | 含义 |
|---|---|---|
| entities.parquet | 6,827 | 知识图谱实体（人名、地名、事件） |
| relationships.parquet | 9,890 | 实体间关系 |
| communities.parquet | 1,487 | 实体聚类社区 |
| documents.parquet | 3,044 | 原始输入文档 |
| text_units.parquet | 3,044 | 向量检索最小粒度 |

**说明什么：** 索引数据量符合预期。如果 `is_index_ready()=False` 或数字偏差大，检索会返回空结果，LLM 没有历史背景可用。

### 3.3 如需重建索引（可选）

仅在索引损坏或需要更换数据源时执行：

```bash
/data/jiabao/miniconda3/bin/python -c "
from src.indexing import GraphBuilder

builder = GraphBuilder(
    input_dir='data/input',
    output_dir='data/graphrag_output',
    llm_provider='openai',       # 或 gemini / glm / ollama
    llm_model='gpt-5',
    embedding_provider='ollama',
    embedding_model='nomic-embed-text',
)
builder.create_settings_yaml()
builder.create_prompts()
result = builder.build_index_sync()
print(result.success, result.message)
"
```

⚠️ 索引构建耗时较长（数十分钟到数小时），会消耗 LLM API 额度。

---

## 阶段四：分段 + 预算 + 校验报告

```bash
/data/jiabao/miniconda3/bin/python -c "
from src.generation import segment_memoir, allocate_segment_budgets, validate_segmentation
from pathlib import Path

text = Path('tests/fixtures/long_memoir_sample.txt').read_text('utf-8')
segs = segment_memoir(text)
budgets = allocate_segment_budgets(segs, '400-800')
report = validate_segmentation(segs)

print(f'总字符数: {len(text)}')
print(f'分段数:   {len(segs)}')
print()
print('=== 分段详情 ===')
for s, b in zip(segs, budgets):
    meta = s.meta
    preview = s.text[:40].replace(chr(10), ' ')
    print(f'  段{s.index}: {len(s.text)}字 -> {b.length_hint} (max_tokens={b.max_tokens})')
    print(f'    时间={meta.temporal_label if meta else \"?\"} 地点={\",\".join(meta.detected_locations) if meta else \"?\"}')
    print(f'    切分原因={meta.split_reason if meta else \"?\"} [{preview}...]')
print()
print('=== 分段校验报告 ===')
print(report.to_text())
"
```

### 期望结果

| 检查点 | 期望 | 异常及含义 |
|---|---|---|
| 分段数 | 6 段（与样本 6 个自然段一致） | 如果是 1 段→时间边界未识别；>10 段→过度切分 |
| 每段 temporal_label | 各不相同（1972、1977、1988 等） | 如果有段无年份→元数据提取失效 |
| 每段 split_reason | 大部分为 `temporal_boundary` | 如果全是 `paragraph_break`→时间边界优先逻辑未生效 |
| 校验报告 | `通过` | 如果有 error→段长度或索引不合理 |

**本步做什么：** 用真实数据验证分段策略是否按预期工作（时间边界优先、不跨年代合并、元数据完整）。
**说明什么：** 分段是整个 pipeline 的第一步——如果分段不对，后续每一章的检索和生成都基于错误的输入。

---

## 阶段五：端到端脚本（需要 Ollama + LLM）

完整跑通 **分段 → 跨章上下文 → 检索 → 生成 → 评估 → 质量门控** 闭环。

```bash
cd /data/jiabao/tempRAG/GraphRAG

/data/jiabao/miniconda3/bin/python scripts/run_long_form_e2e.py \
    --input tests/fixtures/long_memoir_sample.txt \
    --provider openai \
    --length-bucket "400-800" \
    --retrieval-mode keyword \
    --max-gate-retries 2
```

`--max-gate-retries N`：质量门控未通过时，自动对失败章节重新生成，最多重试 N 轮（默认 2）。
设为 0 则不重试（旧行为）。

### 运行流程

```
阶段 A: 初次生成 6 章
阶段 B: 评估 + 质量门控
阶段 C: 若门控未通过 → 根据 RemediationPlan 重新生成失败章节 → 重新评估（最多 N 轮）
阶段 D: 保存生成文本 → data/long_form_e2e_output.txt
阶段 E: 保存评估报告 → data/long_form_e2e_report.json
```

### 检查点

| 检查项 | 期望值 | 说明 |
|---|---|---|
| `chapters=6` | 6 段全部走通 | 生成没有中途崩溃 |
| 索引加载日志 | `加载实体: 6827 条` | 索引加载正确 |
| 评估摘要 | 含"分章数: 6"和各段评分 | 评估流程完整 |
| 跨章指标 | inter_chapter_repetition ≥ 0.7 | 各章不重复 |
| | style_consistency ≥ 0.6 | 风格一致 |
| | summary_sentence_ratio ≥ 0.8 | 总结句极少 |
| 质量门控 | `通过`（可能经过重试） | 达到交付标准 |
| 事实检查 | ≥ 4/6 通过 | 内容基本可信 |
| 生成文本 | `data/long_form_e2e_output.txt` 已生成 | 可直接阅读各章内容 |
| 报告文件 | `data/long_form_e2e_report.json` 已生成 | 含生成内容 + 评估 + 重试记录 |

### 查看生成结果

```bash
# 查看生成的完整文本
cat data/long_form_e2e_output.txt

# 查看报告摘要
/data/jiabao/miniconda3/bin/python -m json.tool data/long_form_e2e_report.json | head -80
```

### 报告 JSON 结构说明

```json
{
  "aggregated_score": 7.85,      // 段级加权综合分 (0-10)
  "segment_count": 6,
  "segments": [{                 // 每章明细
    "segment_index": 0,
    "generated_text": "...",     // 该章生成的完整文本
    "metrics": {                 // 8 个段级指标
      "entity_coverage": {...},
      "time_consistency": {...},
      "keyword_overlap": {...},
      "semantic_similarity": {...},
      "length_score": {...},
      "paragraph_structure": {...},
      "transition_usage": {...},
      "descriptive_richness": {...}
    },
    "eval_overall": 7.5,
    "fact_is_factual": true,
    "fact_score": 0.83           // FActScore 原子事实支持率
  }],
  "merged_content": "...",       // 所有章合并后的完整文本
  "cross_chapter": {             // 跨章指标
    "inter_chapter_repetition": {"value": 0.92, ...},
    "style_consistency": {"value": 0.85, ...},
    "summary_sentence_ratio": {"value": 1.0, ...}
  },
  "quality_gate": {              // 质量门控
    "passed": true,
    "overall_score": 7.85,
    "chapters_to_regenerate": [] // 需重新生成的章节（空=全部通过）
  },
  "gate_retry_rounds": 1,       // 实际重试轮数（0=初次即通过）
  "gate_retry_log": [{          // 每轮重试的记录（仅有重试时存在）
    "round": 1,
    "chapters_regenerated": [1],
    "reasons": {"1": ["FActScore 50% 低于阈值 60%"]}
  }]
}
```

**本步做什么：** 完整端到端验收——真实调 LLM、真实检索、真实评估、质量门控闭环。
**说明什么：**
- `aggregated_score` 回答「生成质量如何」
- `cross_chapter` 回答「各章之间是否有不当重复/风格割裂/过度总结」
- `quality_gate.passed` 回答「是否达到交付标准」
- `quality_gate.chapters_to_regenerate` 回答「如果不达标，应该重新生成哪几章」
- `gate_retry_log` 回答「重试了几轮、每轮修了什么」
- `data/long_form_e2e_output.txt` 回答「最终生成了什么内容」

---

## 阶段六：Web / API 集成验证

### 6.1 启动

```bash
cd /data/jiabao/tempRAG/GraphRAG

# Gradio 界面（推荐）
/data/jiabao/miniconda3/bin/python run_web.py gradio --port 8000

# 或 FastAPI
/data/jiabao/miniconda3/bin/python run_web.py api --port 8000

# 或用 start.sh 一键启动（含 Ollama）
bash start.sh --ollama_path /data/jiabao/RAG/Ollama --port 8000
```

### 6.2 API 测试

```bash
# 健康检查
curl http://localhost:8000/health

# 索引状态
curl http://localhost:8000/index/status

# 单段生成（不分章——验证旧功能未回归）
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "memoir_text": "一九八八年夏天，我从北京来到深圳，在华强北一家电子公司做跟单员。那时候深圳到处都在盖楼，空气里弥漫着水泥和热沥青的味道。",
    "provider": "openai",
    "style": "standard",
    "length_bucket": "400-800",
    "retrieval_mode": "keyword",
    "chapter_mode": false
  }'

# 分章长文生成（核心新功能）
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"memoir_text\": $(python3 -c "import json; print(json.dumps(open('tests/fixtures/long_memoir_sample.txt').read()))"),
    \"provider\": \"openai\",
    \"chapter_mode\": true,
    \"length_bucket\": \"400-800\"
  }"
```

### 6.3 Gradio 界面手动测试

打开 `http://<host>:8000`：

1. **单段回归**：输入一段短回忆录 → 不勾选分章 → 确认正常生成
2. **长文分章**：粘贴 `tests/fixtures/long_memoir_sample.txt` 全文 → 勾选分章模式 → 确认：
   - 有逐章进度提示
   - 输出按 `---` 分隔为 6 章
   - 各章内容不重复
   - 无"总之""综上"等总结性语句
   - 评估摘要含跨章指标和质量门控结果

---

## 快速参考

```
阶段零  启动 Ollama                              [每次重启后必须]
阶段一  pytest tests/ -v                         [36项，无外部依赖]
阶段二  检查 .env / pip list / adapter            [配置检查]
阶段三  验证索引就绪 + 统计                       [需 Ollama]
阶段四  分段 + 预算 + 校验报告                    [无外部依赖]
阶段五  python scripts/run_long_form_e2e.py      [需 Ollama + LLM，含质量门控重试]
阶段六  python run_web.py + curl / 浏览器         [需 Ollama + LLM]
```

---

## 架构改进摘要（v2）

### 1. 分段器增强

| 维度 | 旧版 | 新版 |
|---|---|---|
| 切分策略 | 仅按空行/标题 | **时间边界优先** + 空行/标题 |
| 合并策略 | 短块无条件合并 | **不跨时间边界合并** |
| 元数据 | 无 | SegmentMeta (年份、地点、人物、切分原因) |
| 校验 | 无 | SegmentationReport (长度/跨年代/索引连续性检查) |

### 2. 跨章上下文管理 (ChapterContext)

| 机制 | 作用 |
|---|---|
| 前文概要注入 | 每章 prompt 包含前 1-3 章的内容摘要，LLM 知道前面写了什么 |
| 反重复要点 | 列出已出现的高频短语，明确要求 LLM 不重复 |
| 章节位置指令 | 开篇/中段/收尾使用不同的写作指令 |
| 重复检测 | 生成后立即检测与前文的 n-gram 重叠率，可触发自动重试 |

### 3. Prompt 增强

| 改动 | 效果 |
|---|---|
| 加入 `{chapter_context}` 占位符 | 跨章上下文可注入所有风格模板 |
| 禁止总结性语句（system prompt + user prompt） | 消除"总之""综上"等不当总结 |
| 禁止套话 | 消除"在这个大背景下"等空泛过渡 |
| 要求"只叙事不评论" | 确保回忆录风格而非论文风格 |

### 4. 评估体系增强

| 层级 | 旧版 | 新版 |
|---|---|---|
| 段级指标 | 8 项（但 semantic_similarity 用单字符） | 8 项（semantic_similarity 改为 2-gram） |
| 跨章指标 | 仅 year_diversity | + **inter_chapter_repetition** + **style_consistency** + **summary_sentence_ratio** |
| 事实检查 | 仅 is_factual 布尔值 | + **FActScore 比率** (支持/总数) 输出到 JSON |
| 质量门控 | 无 | **QualityGate**: 字数、综合分、事实、重复、总结句 5 维检查 |
| 修复建议 | 无 | **RemediationPlan**: 哪些章需重新生成 + 每章的修复建议 |

### 5. 质量门控默认阈值

| 指标 | 阈值 | 含义 |
|---|---|---|
| `min_segment_score` | 5.0 | 段级综合分不低于 5/10 |
| `max_cross_repetition` | 0.20 | 相邻章节 6-gram 重叠不超过 20% |
| `min_fact_score` | 0.60 | FActScore ≥ 60% |
| `min_length_ratio` | 0.40 | 实际字数不低于目标的 40% |
| `max_length_ratio` | 2.50 | 实际字数不超过目标的 250% |
| `max_summary_sentence_ratio` | 0.30 | 总结性语句不超过 30% |
