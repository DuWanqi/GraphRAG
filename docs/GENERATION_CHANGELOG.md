# 生成模块变更日志（Generation）

本文档用于**直白说明**三件事：  
1) 生成模块通过什么方式实现升级；  
2) 具体新增了什么功能；  
3) 最终对整条产品链路贡献了什么能力。

---

## 一句话结论

本轮不是“加几个散点功能”，而是把生成模块升级为两条可切换链路：  

- `chapter_mode=false`：原有单段链路（保持兼容）。  
- `chapter_mode=true`：长文分章链路（分段检索、分段生成、统一评估聚合）。  

这使系统从“适配短输入”扩展到“可稳定处理数千字多年代叙事”。

---

## 通过什么方式实现

### 1) 用“单一编排中心”组织长文生成

- 入口：`src/generation/long_form_orchestrator.py` 的 `run_long_form_generation`。  
- 编排顺序：`segment_memoir` -> 每段 `retrieve` -> 每段 `generate` -> 合并为 `LongFormGenerationResult`。  
- 结果结构化：每章保留 `segment_text`、`retrieval_result`、`generation`、`length_hint`，为评估与追踪提供统一输入。

### 2) 用“预算分配器”控制多段输出规模

- 实现：`src/generation/chapter_budget.py`。  
- 机制：将 UI/API 的 `length_bucket` 转为全书目标字数，再按段落长度比例分配每段 `length_hint` 和 `max_tokens`。  
- 兼容：`legacy_maps_for_single_segment` 继续支持单段旧逻辑。

### 3) 用“段级评估 + 文档聚合”替代长文硬套短文规则

- 实现：`src/evaluation/long_form_eval.py` 的 `evaluate_long_form`。  
- 策略：按章计算 metrics / evaluator，再按段权重聚合总分；事实检查支持分段限流与超时跳过。  
- 输出：同时产出可读摘要与 JSON，可用于 API 返回和验收记录。

### 4) 用“运行参数集中化”降低入口代码复杂度（本次结构优化）

- 新增：`src/generation/runtime_options.py`。  
- 统一内容：
  - `single_segment_generation_config`（单段字数档位映射）  
  - `estimate_long_form_generation_timeout`（长文生成超时预算）  
  - `estimate_long_form_evaluation_timeout`（长文评估超时预算）  
  - `build_long_form_eval_options`（长文评估参数组装）  
- 结果：`web/app.py`、`web/api.py` 不再重复硬编码映射和预算逻辑，阅读与维护成本明显下降。

---

## 添加/实现了什么功能（用户可感知）

### A. 长文分章生成能力

- Web 勾选“分章/长文模式”或 API 传 `chapter_mode=true` 后，系统自动切段并逐章生成。  
- 生成结果按章合并，避免“整篇只做一次检索”导致的后段信息缺失。  

### B. 长文评估可执行、可返回、可验收

- 分章结果可直接进入 `evaluate_long_form`。  
- API 分章路径可返回 `eval_summary`（摘要 + JSON 片段）。  
- 脚本 `scripts/run_long_form_e2e.py` 可跑通“生成 + 评估”闭环。

### C. 单段模式保持原体验

- 默认仍是单段；不开 `chapter_mode` 不改变原始用户路径。  
- 单段输出长度控制仍沿用既有档位行为。

---

## 相关代码结构（优化后推荐认知路径）

按“先看入口，再看实现细节”的顺序：

1. `web/app.py` / `web/api.py`  
   - 只保留流程选择与参数透传（单段/分章切换）。  
2. `src/generation/runtime_options.py`  
   - 集中运行策略：字数档位、超时预算、评估参数。  
3. `src/generation/long_form_orchestrator.py`  
   - 长文链路核心编排。  
4. `src/generation/memoir_segmenter.py` + `src/generation/chapter_budget.py`  
   - 分段策略与预算分配策略。  
5. `src/evaluation/long_form_eval.py`  
   - 分章评估聚合实现。  
6. `src/generation/literary_generator.py`  
   - 单段生成核心与 `generate_long_form` 薄封装。

---

## 对整体更新的贡献（Contribute）

本轮生成模块升级对项目的整体贡献可总结为：

- **能力层**：从“短文生成工具”升级为“短文 + 长文双模式生成系统”。  
- **工程层**：形成可复用的长文编排与评估闭环，支持自动化验收与问题定位。  
- **产品层**：在不破坏旧体验的前提下，给出长输入场景的稳定路径。  
- **维护层**：运行配置集中化后，Web/API 的后续迭代成本更低、行为更一致。

---

## 变更清单（文件级）

- 新增：`src/generation/runtime_options.py`（运行参数集中层）。  
- 更新：`src/generation/__init__.py`（导出统一运行参数 helper）。  
- 更新：`web/app.py`（去除重复映射，统一调用运行参数 helper）。  
- 更新：`web/api.py`（统一单段配置与长文超时/评估参数组装）。  
- 既有长文链路相关：
  - `src/generation/memoir_segmenter.py`
  - `src/generation/chapter_budget.py`
  - `src/generation/long_form_orchestrator.py`
  - `src/evaluation/long_form_eval.py`

---

## 验证方式

```bash
cd tempRAG/GraphRAG
pytest tests/test_basic.py tests/test_memoir_segmenter.py tests/test_chapter_budget.py tests/test_long_form_orchestrator.py tests/test_long_form_pipeline.py -v
# 可选（需已配置 LLM 与索引）
# python scripts/run_long_form_e2e.py
```

---

## 备注

- 并行生成路径已与单次生成对齐使用 `get_system_prompt("default")`，避免系统提示词不一致。
