# Novel Content Generation - Implementation Summary

## 实现完成 ✓

本次实现完成了"RAG 新内容生成"功能，使生成模块能够将 RAG 检索到的、输入中未提及的新知识实质性地编织进叙事。

## 实现的文件

### 1. 新增文件

| 文件 | 说明 |
|------|------|
| `src/generation/novel_content_extractor.py` | 新内容提取器，从 RAG 结果中区分"对齐内容"和"新知识" |
| `src/evaluation/novel_content_metrics.py` | 新内容评估指标（引入率、溯源率、扩展深度） |
| `generation_test/test_novel_content.py` | 新内容生成验证测试 |
| `docs/novel_content_generation.md` | 行为差异文档和使用指南 |
| `IMPLEMENTATION_SUMMARY.md` | 本文件 |

### 2. 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `src/generation/prompts.json` | 拆分 context 为 aligned_context 和 novel_context，新增新知识融入指令 |
| `src/generation/literary_generator.py` | `_build_prompt` 使用新内容提取器，GenerationResult 新增 novel_content_brief 字段 |
| `src/generation/__init__.py` | 导出 NovelContentBrief 和 extract_novel_content |
| `src/evaluation/metrics.py` | calculate_all_metrics 新增 novel_content_brief 参数，集成新内容指标 |
| `src/evaluation/long_form_eval.py` | SegmentEvalRecord 新增 novel_content_info 字段，集成新内容评估 |
| `src/evaluation/__init__.py` | 导出新内容评估相关函数 |

## 核心功能

### 1. 新内容提取 (novel_content_extractor.py)

**功能**：从 RAG 检索结果中区分"对齐内容"（输入已有）和"新知识"（输入未提及）

**核心类**：
```python
@dataclass
class NovelContentBrief:
    novel_entities: List[Dict]          # 新实体
    novel_relationships: List[Dict]     # 新关系
    novel_snippets: List[str]           # 新背景片段
    aligned_entities: List[Dict]        # 对齐实体
    aligned_relationships: List[Dict]   # 对齐关系
    summary: str                        # 摘要
```

**核心函数**：
```python
def extract_novel_content(
    memoir_text: str,
    retrieval_result: RetrievalResult,
) -> NovelContentBrief
```

**匹配策略**：
- 精确匹配：实体名直接出现在原文中
- 模糊匹配：考虑中英文变体、简称、部分匹配
- 关键词提取：中文词（2+ 字）、英文词（3+ 字母）、年份

### 2. Prompt 改进 (prompts.json)

**改进前**：
```
## 可参考的历史背景信息
{context}

写作要求：
4. 将上面提供的历史背景信息自然地编织进叙事中
```

**改进后**：
```
## 与原文相关的背景（用于增强叙事氛围）
{aligned_context}

## 原文未提及的新知识（请选择性融入叙事）
{novel_context}

写作要求：
4. **重点：从上面的「原文未提及的新知识」中选取 1-3 个与叙事最相关的事实，
   自然地编织进故事中**——例如作为人物对话的内容、环境描写的细节、或叙事者的见闻
5. 融入新知识时，必须基于上面提供的检索信息，不要自行编造未提供的历史事实
```

**关键改进**：
- 明确区分对齐内容和新知识
- 用"请选择性融入"替代"可参考"，措辞更强
- 明确要求选取 1-3 个新事实
- 提供融入方式的具体建议
- 明确禁止编造（防幻觉）

### 3. 生成流程改造 (literary_generator.py)

**改进前**：
```python
def _build_prompt(...):
    context = retrieval_result.get_context_text()
    return template.format(
        memoir_text=memoir_text,
        context=context,  # 整块注入
        ...
    )
```

**改进后**：
```python
def _build_prompt(...):
    # 提取并分类 RAG 内容
    novel_brief = extract_novel_content(memoir_text, retrieval_result)
    
    # 格式化为两个区块
    formatted = novel_brief.format_for_prompt()
    
    return template.format(
        memoir_text=memoir_text,
        aligned_context=formatted["aligned_context"],
        novel_context=formatted["novel_context"],
        ...
    )
```

### 4. 新内容评估指标 (novel_content_metrics.py)

#### a) novel_content_ratio - 新内容引入率

**定义**：生成文本使用了多少 RAG 提供的新知识

**计算**：
```
novel_content_ratio = 使用的新实体数量 / 可用新实体数量
```

**评分标准**：
- 0.0 - 0.2：未引入新知识
- 0.2 - 0.5：引入了部分新知识
- 0.5 - 1.0：充分利用了新知识

#### b) novel_content_grounding - 新内容溯源率

**定义**：生成文本中的新事实是否有 RAG 来源支撑（防幻觉）

**计算**：
```
novel_content_grounding = 有RAG来源支撑的新事实 / 总新事实
```

**评分标准**：
- 0.8 - 1.0：新内容完全有据可查
- 0.5 - 0.8：大部分新内容有据
- 0.0 - 0.5：大量幻觉，不可接受

#### c) expansion_depth - 扩展深度

**定义**：生成文本相对于输入的信息增量

**分类**：
- shallow（0.3）：仅润色，未引入新知识
- moderate（0.7）：引入 1-2 个新事实
- deep（1.0）：引入 3+ 个新事实并有叙事整合

### 5. 长文评估集成 (long_form_eval.py)

**改进**：
- `SegmentEvalRecord` 新增 `novel_content_info` 字段
- `_metrics_for_segment` 新增 `novel_content_brief` 参数
- `_eval_one_chapter` 自动提取和评估新内容
- 摘要输出包含新内容评估信息
- JSON 输出包含 novel_content 字段

**输出示例**：
```
- 段 0: 指标聚合 7.85 | Evaluator 8.20 | 事实检查 通过 (FActScore 95%) | 新内容引入率 50% | 溯源率 100%
```

## 测试验证

### 测试文件：generation_test/test_novel_content.py

**测试用例**：

1. **test_novel_content_extraction** - 新内容提取功能
   - 验证能正确区分 aligned vs novel
   - 验证有可用的新知识

2. **test_novel_content_generation** - 新内容生成功能
   - 使用极简输入（只提到年份和事件）
   - 验证生成文本包含至少 1 个新实体
   - 验证新实体来自 RAG 检索结果

3. **test_novel_content_metrics** - 新内容评估指标
   - 验证 novel_content_ratio > 0
   - 验证 novel_content_grounding > 0.5
   - 验证 expansion_depth 计算正确

**运行方式**：
```bash
# 使用 pytest
conda run -n RAG pytest generation_test/test_novel_content.py -v -s

# 或直接运行
conda run -n RAG python generation_test/test_novel_content.py
```

**测试特性**：
- 添加 `pytest.mark.timeout(120)` 防止挂起
- 添加 LLM 健康检查，连接失败时 skip
- 添加详细的 print 诊断输出

## 使用示例

### 基本使用

```python
from src.generation import LiteraryGenerator
from src.retrieval import MemoirRetriever

# 检索
retriever = MemoirRetriever(index_dir="data/graphrag_output")
retrieval_result = await retriever.retrieve(memoir_text, top_k=10)

# 生成（自动使用新内容提取）
generator = LiteraryGenerator(llm_adapter=llm_adapter)
generation_result = await generator.generate(
    memoir_text=memoir_text,
    retrieval_result=retrieval_result,
    style="standard",
    length_hint="300-500字",
)

# 查看新内容信息
if generation_result.novel_content_brief:
    print(f"可用新知识: {generation_result.novel_content_brief.summary}")
    print(f"新实体: {generation_result.novel_content_brief.novel_entity_names}")
```

### 评估新内容

```python
from src.evaluation import (
    novel_content_ratio_metric,
    novel_content_grounding_metric,
    expansion_depth_metric,
)

novel_brief = generation_result.novel_content_brief

ratio = novel_content_ratio_metric(memoir_text, generated_text, novel_brief)
grounding = novel_content_grounding_metric(memoir_text, generated_text, novel_brief)
depth = expansion_depth_metric(memoir_text, generated_text, novel_brief)

print(f"新内容引入率: {ratio.value:.0%} - {ratio.explanation}")
print(f"新内容溯源率: {grounding.value:.0%} - {grounding.explanation}")
print(f"扩展深度: {depth.explanation}")
```

### 长文评估（自动集成）

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

## 技术亮点

### 1. 纯规则实现，无 LLM 依赖

新内容提取器使用纯规则实现，不依赖 LLM：
- 快速：无需额外的 LLM 调用
- 可靠：规则明确，结果可预测
- 可调试：易于理解和调试

### 2. 模糊匹配策略

考虑中文文本的特点：
- 中英文变体（"深圳" vs "SHENZHEN"）
- 简称（"改革开放" vs "改革"）
- 部分匹配（"中华人民共和国" vs "中华"）

### 3. 防幻觉机制

多层防护：
- Prompt 层面：明确禁止编造未提供的事实
- 评估层面：novel_content_grounding 检测幻觉
- 文档层面：明确定义合理扩展 vs 幻觉的边界

### 4. 向后兼容

- 新功能可选：如果不提供 novel_content_brief，现有功能不受影响
- 渐进式集成：新指标自动集成到 calculate_all_metrics 和 evaluate_long_form
- 权重可调：aggregate_scores 中新指标的权重可配置

## 验证清单

- [x] 新内容提取器实现完成
- [x] Prompt 模板改进完成（standard/nostalgic/narrative）
- [x] 生成流程改造完成
- [x] 新内容评估指标实现完成
- [x] 长文评估集成完成
- [x] 模块导出更新完成
- [x] 测试文件创建完成
- [x] 文档编写完成
- [x] 导入验证通过

## 下一步

### 立即可做

1. **运行测试验证**：
   ```bash
   conda run -n RAG python generation_test/test_novel_content.py
   ```

2. **对比生成结果**：
   - 使用相同输入，对比改进前后的生成结果
   - 验证新版本确实引入了更多新知识

3. **调整权重**（可选）：
   - 根据实际效果调整 aggregate_scores 中新指标的权重
   - 当前权重：novel_content_ratio=1.2, novel_content_grounding=1.5, expansion_depth=1.0

### 后续优化

1. **提取策略优化**：
   - 当前使用规则匹配，可考虑引入语义相似度
   - 优化模糊匹配的阈值和策略

2. **Prompt 微调**：
   - 根据实际生成效果调整 prompt 措辞
   - 可能需要针对不同风格（standard/nostalgic/narrative）微调

3. **评估指标优化**：
   - novel_content_grounding 当前使用规则检测，可考虑引入 LLM 辅助
   - expansion_depth 可以更细粒度（如 5 档而非 3 档）

4. **A/B 测试**：
   - 对比改进前后的用户满意度
   - 收集用户反馈，持续优化

## 文档

- **使用指南**：`docs/novel_content_generation.md`
- **API 文档**：代码中的 docstring
- **测试文档**：`generation_test/test_novel_content.py` 中的注释

## 联系

如有问题或建议，请参考：
- 代码注释和 docstring
- `docs/novel_content_generation.md`
- 测试用例 `generation_test/test_novel_content.py`
