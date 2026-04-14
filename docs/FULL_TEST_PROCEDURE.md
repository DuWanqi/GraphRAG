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

---

## 阶段一：单元测试（无需 LLM / Ollama / 索引）

验证分段、预算、编排、评估的核心逻辑。

```bash
cd /data/jiabao/tempRAG/GraphRAG

/data/jiabao/miniconda3/bin/python -m pytest tests/ -v
```

**期望**：17 项全部 PASSED。

各测试文件覆盖范围：

| 测试文件 | 覆盖内容 |
|---|---|
| `test_memoir_segmenter.py` | 空输入、短文单段、空行/标题切分、索引连续 |
| `test_chapter_budget.py` | 按比例分配字数、legacy 单段映射 |
| `test_long_form_orchestrator.py` | 段数=检索次数=生成次数、合并含分隔符 |
| `test_long_form_pipeline.py` | `evaluate_long_form` 全链路（Mock，无 LLM） |
| `test_basic.py` | 解析器、指标、数据加载、配置、provider 枚举 |

---

## 阶段二：环境与配置检查

### 2.1 Python 依赖

```bash
/data/jiabao/miniconda3/bin/pip list | grep -iE "pandas|pydantic-settings|graphrag|pyarrow|lancedb|litellm"
```

必须存在：`pandas`、`pydantic-settings`、`pyarrow`、`litellm`。

### 2.2 `.env` 配置确认

```bash
cat .env | grep -E "^(OPENAI_|DEFAULT_|GRAPHRAG_|OLLAMA_)" 
```

关键项：

| 变量 | 当前值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | `sk-maX0u...` | OpenAI（代理）密钥 |
| `OPENAI_BASE_URL` | `https://api.shubiaobiao.cn/v1/` | 代理地址 |
| `GRAPHRAG_OUTPUT_DIR` | `./data/graphrag_output` | 索引目录（已有索引） |
| `DEFAULT_LLM_PROVIDER` | `gemini` | 默认 provider（不影响本流程） |

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

当前已有索引：

| 文件 | 记录数 |
|---|---|
| entities.parquet | 6,827 |
| relationships.parquet | 9,890 |
| communities.parquet | 1,487 |
| documents.parquet | 3,044 |
| text_units.parquet | 3,044 |

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

## 阶段四：分段 + 预算交互验证

```bash
/data/jiabao/miniconda3/bin/python -c "
from src.generation import segment_memoir, allocate_segment_budgets
from pathlib import Path

text = Path('tests/fixtures/long_memoir_sample.txt').read_text('utf-8')
segs = segment_memoir(text)
budgets = allocate_segment_budgets(segs, '400-800')

print(f'总字符数: {len(text)}')
print(f'分段数:   {len(segs)}\n')
for s, b in zip(segs, budgets):
    preview = s.text[:30].replace(chr(10), ' ')
    print(f'  段{s.index}: {len(s.text)}字 -> {b.length_hint} (max_tokens={b.max_tokens})  [{preview}...]')
"
```

期望：6 段，每段 ~400 字，预算按比例分配。

---

## 阶段五：端到端脚本（需要 Ollama + LLM）

完整跑通 **分段 → 检索 → 生成 → 评估** 闭环。

```bash
cd /data/jiabao/tempRAG/GraphRAG

/data/jiabao/miniconda3/bin/python scripts/run_long_form_e2e.py \
    --input tests/fixtures/long_memoir_sample.txt \
    --provider openai \
    --length-bucket "400-800" \
    --retrieval-mode keyword
```

### 检查点

| 检查项 | 期望值 |
|---|---|
| `chapters=6` | 6 段全部走通 |
| 索引加载日志 | `[DEBUG] 加载实体: 6827 条` |
| 评估摘要 | 含"分章数: 6"和各段评分 |
| 报告文件 | `data/long_form_e2e_report.json` 已生成 |

### 查看报告

```bash
/data/jiabao/miniconda3/bin/python -m json.tool data/long_form_e2e_report.json | head -50
```

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

# 单段生成
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

# 分章长文生成
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
   - 评估摘要正常返回

---

## 快速参考

```
阶段零  启动 Ollama                              [每次重启后必须]
阶段一  pytest tests/ -v                         [无外部依赖]
阶段二  检查 .env / pip list / adapter            [配置检查]
阶段三  验证索引就绪 + 统计                       [需 Ollama]
阶段四  分段 + 预算交互验证                       [无外部依赖]
阶段五  python scripts/run_long_form_e2e.py      [需 Ollama + LLM]
阶段六  python run_web.py + curl / 浏览器         [需 Ollama + LLM]
```

## 已验证基准结果（2026-04-15）

- Provider: OpenAI gpt-5 (via shubiaobiao proxy)
- 单元测试: 17/17 PASSED
- E2E 分章: 6 章，生成 28.77s，评估 157.08s
- 加权综合分: 7.85
- 事实检查: 5/6 通过，1/6 待核
