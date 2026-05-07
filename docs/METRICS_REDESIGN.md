# Expansion任务评估系统重新设计

## 当前Metrics清单与重复分析

### 现有Metrics（共11个段级指标）

| 指标名称 | 类别 | 当前定义 | 测试结果 |
|---------|------|---------|---------|
| **entity_coverage** | 准确性 | 使用了多少个实体（要求至少2个） | 1.00 ✓ |
| **time_consistency** | 准确性 | 是否包含参考年份 | 0.30 ✗ |
| **keyword_overlap** | 相关性 | 关键词精确匹配率 | 0.40 |
| **semantic_similarity** | 相关性 | 2-gram余弦相似度 | 0.23 |
| **length_score** | 文学性 | 长度是否在合理范围 | 0.96 ✓ |
| **paragraph_structure** | 文学性 | 段落结构是否合理 | 1.00 ✓ |
| **transition_usage** | 文学性 | 过渡词使用率 | 0.70 |
| **descriptive_richness** | 文学性 | 描写丰富度 | 0.75 |
| **novel_content_ratio** | 扩展质量 | 使用了多少个新实体（分段评分） | 0.70 |
| **novel_content_grounding** | 扩展质量 | 新实体是否来自RAG | 1.00 ✓ |
| **expansion_depth** | 扩展质量 | 扩展深度（shallow/moderate/deep） | 0.70 |

### 额外指标（质量门控）
| 指标名称 | 定义 | 测试结果 |
|---------|------|---------|
| **FActScore** | 事实陈述的可验证率 | 22.2% ✗ |

---

## 重复问题分析

### 🔴 严重重复

#### 1. **novel_content_ratio vs expansion_depth**
**问题**：这两个指标几乎完全重复

- **novel_content_ratio**（我刚修改的）：
  ```python
  0个实体 → 0.0
  1个实体 → 0.4
  2个实体 → 0.7
  3+个实体 → 1.0
  ```

- **expansion_depth**：
  ```python
  0个实体 → 0.3 (shallow)
  1-2个实体 → 0.7 (moderate)
  3+个实体 → 1.0 (deep)
  ```

**结论**：两者都基于"使用的新实体数量"分段评分，逻辑几乎相同。

---

#### 2. **keyword_overlap vs semantic_similarity**
**问题**：都在衡量"生成文本与原文的相似度"

- **keyword_overlap**: 关键词精确匹配 → 0.40
- **semantic_similarity**: 2-gram余弦相似度 → 0.23

**矛盾**：
- 对keyword_overlap说"需要语义匹配而非精确匹配"
- 对semantic_similarity说"expansion导致语义变化正常"

**结论**：概念混乱，需要统一为一个指标。

---

#### 3. **novel_content_grounding vs FActScore**
**问题**：都在衡量"准确性/溯源性"

- **novel_content_grounding**: 使用的新实体是否来自RAG → 1.00
- **FActScore**: 事实陈述是否可验证 → 22.2%

**矛盾**：
- novel_content_grounding显示100%溯源
- FActScore显示只有22.2%可验证

**结论**：衡量的粒度不同（实体 vs 事实陈述），但目标相同。

---

### 🟡 部分重复

#### 4. **entity_coverage vs novel_content_ratio**
**问题**：都在衡量"实体使用情况"

- **entity_coverage**: 使用了多少个实体（包括对齐实体+新实体）
- **novel_content_ratio**: 使用了多少个新实体

**区别**：
- entity_coverage关注"总实体数"
- novel_content_ratio关注"新实体数"

**结论**：有重叠但不完全相同，可以保留但需要明确区分。

---

## 重新设计的评估维度

### Expansion任务真正需要评估的维度

#### 1. **准确性（Accuracy）** - 核心事实是否正确？
- 使用的RAG实体是否准确描述？
- 是否有幻觉（未在RAG中的事实）？

#### 2. **相关性（Relevance）** - 是否偏离主题？
- 生成内容是否与原文主题一致？
- 核心概念是否保留？

#### 3. **扩展质量（Expansion Quality）** - 新信息的质量如何？
- 引入了多少新信息？
- 新信息是否来自RAG？
- 扩展的深度如何？

#### 4. **文学质量（Literary Quality）** - 叙事是否流畅？
- 长度是否合理？
- 结构是否清晰？
- 描写是否丰富？
- 过渡是否自然？

#### 5. **一致性（Consistency）** - 时间/逻辑是否一致？
- 时间是否合理？
- 前后是否矛盾？

---

## 新的Metrics设计（无重复）

### 📊 推荐的Metrics体系（共8个）

| 维度 | 指标名称 | 定义 | 计算方式 |
|-----|---------|------|---------|
| **准确性** | `rag_entity_accuracy` | 使用的RAG实体是否准确描述 | 检查使用的实体描述是否与RAG一致 |
| **相关性** | `topic_coherence` | 生成内容是否与原文主题一致 | 核心概念保留率（支持同义改写） |
| **扩展-信息量** | `information_gain` | 引入了多少新信息 | 新实体数量的分段评分（0/1/2/3+） |
| **扩展-溯源** | `expansion_grounding` | 新信息是否来自RAG | 使用的新实体是否都在RAG中 |
| **文学-长度** | `length_score` | 长度是否合理 | 保持现有逻辑 |
| **文学-结构** | `narrative_quality` | 叙事质量（结构+过渡+描写） | 合并paragraph_structure + transition_usage + descriptive_richness |
| **一致性-时间** | `temporal_coherence` | 时间是否合理 | 所有年份是否在±10年范围内 |
| **一致性-逻辑** | `logical_consistency` | 前后是否矛盾 | 跨章检查（如果有多章） |

---

## 详细设计

### 1. rag_entity_accuracy（RAG实体准确性）

**目标**：验证使用的RAG实体是否准确描述（替代FActScore）

**计算方式**：
```python
def rag_entity_accuracy(generated_text, rag_entities, rag_descriptions):
    """
    对于expansion任务，不验证所有"事实陈述"，
    只验证"使用的RAG实体是否准确描述"
    """
    score = 0
    total = 0
    
    for entity in rag_entities:
        if entity in generated_text:
            total += 1
            # 提取生成文本中关于该实体的描述
            context = extract_entity_context(entity, generated_text)
            # 验证描述是否与RAG一致
            if is_description_consistent(context, rag_descriptions[entity]):
                score += 1
    
    return score / total if total > 0 else 1.0
```

**与FActScore的区别**：
- FActScore：提取所有"事实陈述"（包括叙事细节）→ 22.2%
- rag_entity_accuracy：只验证RAG实体的描述 → 预期~80-100%

---

### 2. topic_coherence（主题一致性）

**目标**：检查核心概念是否保留（替代keyword_overlap + semantic_similarity）

**计算方式**：
```python
def topic_coherence(generated_text, memoir_text):
    """
    检查核心概念是否保留（支持同义改写）
    """
    # 提取原文核心概念
    core_concepts = extract_core_concepts(memoir_text)
    # 例如：["大学", "高考", "父母", "梦想"]
    
    preserved = 0
    for concept in core_concepts:
        # 检查概念或其同义词是否出现
        if concept in generated_text or has_synonym(concept, generated_text):
            preserved += 1
    
    return preserved / len(core_concepts)
```

**与keyword_overlap/semantic_similarity的区别**：
- keyword_overlap：精确匹配 → 无法识别同义改写
- semantic_similarity：2-gram重叠 → 对expansion任务不适用
- topic_coherence：核心概念保留（支持同义） → 更合理

---

### 3. information_gain（信息增量）

**目标**：衡量引入了多少新信息（替代novel_content_ratio）

**计算方式**：
```python
def information_gain(novel_entities_used):
    """
    基于使用的新实体数量分段评分
    """
    count = len(novel_entities_used)
    
    if count == 0:
        return 0.0  # 无新信息
    elif count == 1:
        return 0.4  # 少量新信息
    elif count == 2:
        return 0.7  # 适量新信息
    else:
        return 1.0  # 丰富新信息
```

**与novel_content_ratio的区别**：
- 重命名为information_gain，更准确地反映其含义
- 逻辑保持不变

---

### 4. expansion_grounding（扩展溯源率）

**目标**：新信息是否来自RAG（保留novel_content_grounding）

**计算方式**：
```python
def expansion_grounding(novel_entities_used, rag_entities):
    """
    检查使用的新实体是否都在RAG中
    """
    if not novel_entities_used:
        return 1.0  # 没有新实体，默认为完全溯源
    
    grounded = all(entity in rag_entities for entity in novel_entities_used)
    return 1.0 if grounded else 0.0
```

**与novel_content_grounding的区别**：
- 重命名为expansion_grounding，更清晰
- 逻辑保持不变

---

### 5. narrative_quality（叙事质量）

**目标**：合并paragraph_structure + transition_usage + descriptive_richness

**计算方式**：
```python
def narrative_quality(generated_text):
    """
    综合评估叙事质量
    """
    # 1. 段落结构（30%权重）
    structure_score = check_paragraph_structure(generated_text)
    
    # 2. 过渡自然度（30%权重）
    transition_score = check_transitions(generated_text)
    
    # 3. 描写丰富度（40%权重）
    richness_score = check_descriptive_richness(generated_text)
    
    return 0.3 * structure_score + 0.3 * transition_score + 0.4 * richness_score
```

**与原指标的区别**：
- 原来：3个独立指标（paragraph_structure, transition_usage, descriptive_richness）
- 现在：1个综合指标（narrative_quality）
- 减少指标数量，避免过度细分

---

### 6. temporal_coherence（时间一致性）

**目标**：修复time_consistency的逻辑错误

**计算方式**：
```python
def temporal_coherence(generated_text, reference_year):
    """
    检查所有年份是否在合理范围内（±10年）
    """
    if not reference_year:
        return 0.5
    
    ref_year_int = int(reference_year)
    years_in_text = extract_all_years(generated_text)
    
    if not years_in_text:
        return 0.3
    
    # 检查所有年份是否在±10年范围内
    valid_years = [y for y in years_in_text 
                   if abs(y - ref_year_int) <= 10]
    
    return len(valid_years) / len(years_in_text)
```

**与time_consistency的区别**：
- time_consistency：只检查是否包含参考年份 → 逻辑错误
- temporal_coherence：检查所有年份是否在合理范围 → 更合理

---

## 删除的Metrics及原因

| 删除的指标 | 原因 |
|-----------|------|
| **expansion_depth** | 与information_gain重复（都基于新实体数量） |
| **keyword_overlap** | 与topic_coherence重复（都衡量相关性） |
| **semantic_similarity** | 与topic_coherence重复（都衡量相关性） |
| **paragraph_structure** | 合并到narrative_quality |
| **transition_usage** | 合并到narrative_quality |
| **descriptive_richness** | 合并到narrative_quality |
| **time_consistency** | 被temporal_coherence替代（修复逻辑错误） |
| **FActScore** | 被rag_entity_accuracy替代（更适合expansion） |

---

## 新旧对比

### 指标数量
- **旧系统**：11个段级指标 + 1个质量门控指标 = 12个
- **新系统**：8个段级指标 = 8个
- **减少**：4个指标（33%）

### 重复情况
- **旧系统**：3组严重重复 + 1组部分重复
- **新系统**：0组重复

### 覆盖维度
- **旧系统**：准确性、相关性、扩展质量、文学性、一致性
- **新系统**：准确性、相关性、扩展质量、文学性、一致性
- **覆盖度**：相同

---

## 实施计划

### Phase 1: 重命名和合并（低风险）
1. `novel_content_ratio` → `information_gain`（重命名）
2. `novel_content_grounding` → `expansion_grounding`（重命名）
3. 删除`expansion_depth`（与information_gain重复）
4. 合并`paragraph_structure` + `transition_usage` + `descriptive_richness` → `narrative_quality`

### Phase 2: 修复逻辑错误（中风险）
1. `time_consistency` → `temporal_coherence`（修复逻辑）
2. `FActScore` → `rag_entity_accuracy`（重新设计）

### Phase 3: 重新设计相关性指标（高风险）
1. 删除`keyword_overlap`和`semantic_similarity`
2. 新增`topic_coherence`（需要实现同义词识别）

---

## 测试案例预期结果

| 指标 | 旧系统 | 新系统 | 说明 |
|-----|-------|-------|------|
| entity_coverage | 1.00 | - | 保留但可能重命名 |
| time_consistency | 0.30 | - | 删除 |
| temporal_coherence | - | ~0.90 | 新增（修复逻辑） |
| keyword_overlap | 0.40 | - | 删除 |
| semantic_similarity | 0.23 | - | 删除 |
| topic_coherence | - | ~0.70 | 新增（支持同义） |
| novel_content_ratio | 0.70 | - | 重命名 |
| information_gain | - | 0.70 | 重命名自novel_content_ratio |
| expansion_depth | 0.70 | - | 删除（重复） |
| novel_content_grounding | 1.00 | - | 重命名 |
| expansion_grounding | - | 1.00 | 重命名自novel_content_grounding |
| paragraph_structure | 1.00 | - | 合并 |
| transition_usage | 0.70 | - | 合并 |
| descriptive_richness | 0.75 | - | 合并 |
| narrative_quality | - | ~0.80 | 新增（合并3个指标） |
| length_score | 0.96 | 0.96 | 保持不变 |
| FActScore | 22.2% | - | 删除 |
| rag_entity_accuracy | - | ~90% | 新增（替代FActScore） |

---

## 下一步

1. 用户确认设计方案
2. 实施Phase 1（重命名和合并）
3. 实施Phase 2（修复逻辑错误）
4. 实施Phase 3（重新设计相关性指标）
5. 运行测试验证
