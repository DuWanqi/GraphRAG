# 记忆图谱系统核心模块实现文档

## 1. 关键词检索模块

### 1.1 实现原理

关键词检索模块是系统的默认检索策略，基于文本匹配实现快速、精准的检索。其核心原理是：

1. **上下文解析**：通过 `MemoirParser` 解析回忆录文本，提取关键信息（年份、地点、关键词）
2. **搜索词构建**：构建包含中英文变体的搜索词集合
3. **多维度匹配**：对实体、关系、社区报告、文本单元进行关键词匹配
4. **分数计算**：基于匹配度计算得分并排序

### 1.2 核心实现

```python
# 核心搜索逻辑
def _search_entities(self, context: MemoirContext, top_k: int) -> List[Dict[str, Any]]:
    """搜索相关实体"""
    if self._entities_df is None or self._entities_df.empty:
        return []
    
    results = []
    
    # 构建搜索词（包含中英文变体）
    search_terms = []
    if context.year:
        search_terms.append(context.year)
        search_terms.append(str(context.year))
    if context.location:
        search_terms.append(context.location)
        # 添加英文变体
        location_map = {
            "深圳": "SHENZHEN", "北京": "BEIJING", "上海": "SHANGHAI",
            "广州": "GUANGZHOU", "香港": "HONG KONG"
        }
        if context.location in location_map:
            search_terms.append(location_map[context.location])
    search_terms.extend(context.keywords)
    
    for _, row in self._entities_df.iterrows():
        # GraphRAG 2.x 使用 title 字段
        entity_name = str(row.get("title", row.get("name", "")))
        entity_desc = str(row.get("description", ""))
        
        # 计算匹配分数（不区分大小写）
        score = 0
        search_text = f"{entity_name} {entity_desc}".upper()
        for term in search_terms:
            if term and term.upper() in search_text:
                score += 1
        
        if score > 0:
            results.append({
                "name": entity_name,
                "type": row.get("type", "unknown"),
                "description": entity_desc,
                "score": score,
            })
    
    # 按分数排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
```

### 1.3 技术特点

- **中英文映射**：内置地点中英文映射表，支持跨语言检索
- **多维度检索**：同时检索实体、关系、社区报告和文本单元
- **高效匹配**：基于字符串匹配，速度快，延迟低
- **兼容GraphRAG版本**：支持GraphRAG 1.x和2.x的文件格式

### 1.4 评测结果

| 指标 | 值 |
|------|----|
| Precision | 0.2476 |
| Recall | 0.4286 |
| F1 | 0.2959 |
| Hit@3 | 42.86% |
| Hit@5 | 42.86% |
| Hit@10 | 42.86% |
| 平均延迟 | 3.38ms |

**结论**：关键词检索是当前性能最佳的检索策略，在F1分数和响应速度方面表现优异。

## 2. 混合检索模块

### 2.1 实现原理

混合检索模块结合了关键词检索和向量检索的优势，通过加权融合两种检索结果，提高检索的准确性和全面性。其核心原理是：

1. **并行检索**：同时执行关键词检索和向量检索
2. **结果融合**：对两种检索结果进行加权融合（关键词得分*0.5 + 向量得分*0.5）
3. **去重排序**：去除重复结果并按融合得分排序

### 2.2 核心实现

```python
def _merge_results(
    self,
    keyword_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """融合关键词和向量检索结果"""
    merged = {}
    
    for item in keyword_results:
        name = item.get("name", item.get("title", ""))
        if name:
            merged[name] = item.copy()
            merged[name]["score"] = item.get("score", 1) * 0.5
            merged[name]["source"] = "keyword"
    
    for item in vector_results:
        name = item.get("name", item.get("title", ""))
        if name:
            if name in merged:
                merged[name]["score"] += item.get("score", 1) * 0.5
                merged[name]["source"] = "hybrid"
            else:
                merged[name] = item.copy()
                merged[name]["score"] = item.get("score", 1) * 0.5
                merged[name]["source"] = "vector"
    
    sorted_results = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
    return sorted_results[:top_k]
```

### 2.3 技术特点

- **加权融合**：通过固定权重（关键词0.5 + 向量0.5）融合两种检索结果
- **多模态信息**：同时利用文本匹配和语义相似度信息
- **结果溯源**：标记结果来源，便于后续分析
- **文本去重**：对文本结果进行去重处理，避免重复内容

### 2.4 向量检索实现

混合检索依赖于向量检索模块，其核心实现如下：

```python
async def search_entities(
    self,
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """向量检索实体"""
    self._connect()
    
    if self._entity_table is None:
        return []
    
    embedding = await self._get_query_embedding(query)
    if embedding is None:
        return []
    
    try:
        query_vector = np.array(embedding, dtype=np.float32)
        
        results = self._entity_table.search(query_vector).limit(top_k).to_pandas()
        
        entities = []
        for _, row in results.iterrows():
            # LanceDB返回的列名是text，不是title/name
            text = row.get("text", "")
            name = text.split(":")[0] if ":" in text else text[:50]
            entities.append({
                "name": name,
                "description": text,
                "type": "unknown",
                "score": float(row.get("_distance", 1.0)),
                "source": "vector",
            })
        
        return entities
        
    except Exception as e:
        print(f"[VectorRetriever] 实体检索失败: {e}")
        return []
```

### 2.5 评测结果

| 策略 | Precision | Recall | F1 | Hit@3 | Hit@5 | Hit@10 | 延迟(ms) |
|------|-----------|--------|-----|-------|-------|--------|----------|
| hybrid_keyword | 0.2476 | 0.4286 | 0.2959 | 42.86% | 42.86% | 42.86% | 6.50 |
| hybrid_vector | 0.0891 | 0.8571 | 0.1599 | 14.29% | 28.57% | 42.86% | 309.16 |

**结论**：
- `hybrid_keyword` 策略保持了与纯关键词检索相同的F1分数，但延迟略有增加
- `hybrid_vector` 策略提高了Recall（0.8571），但Precision较低，导致F1分数不高

## 3. 事实准确性判断模块

### 3.1 实现原理

事实准确性判断模块用于检测生成内容与原始回忆录的事实一致性，识别潜在的幻觉。其核心原理是：

1. **规则快速检查**：使用正则表达式等规则检测时间一致性和数字差异
2. **LLM深度检查**：使用LLM进行实体提取和一致性分析
3. **证据支持度计算**：基于分词级别的词重叠率计算证据支持度
4. **结果去重与评分**：去重不一致项并计算最终的事实性评分

### 3.2 核心实现

```python
async def check(
    self,
    memoir_text: str,
    generated_text: str,
    retrieval_result: Optional[RetrievalResult] = None,
    use_llm: bool = True,
) -> FactCheckResult:
    """执行事实性检查"""
    cache_key = self._cache_key(memoir_text, generated_text)
    if cache_key in self._result_cache:
        return self._result_cache[cache_key]
    
    inconsistencies: List[Inconsistency] = []
    
    # 第一阶段：规则快速检查（无LLM成本）
    time_issues = self._check_time_consistency(memoir_text, generated_text)
    inconsistencies.extend(time_issues)
    
    number_issues = self._check_number_discrepancy(memoir_text, generated_text)
    inconsistencies.extend(number_issues)

    # 计算指标
    entity_coverage = self._calculate_entity_coverage(generated_text, retrieval_result)
    evidence_support = self._calculate_evidence_support(generated_text, retrieval_result)

    # 第二阶段：LLM统一检查（一次调用完成所有分析）
    if use_llm and self.llm_adapter:
        llm_result = await self._unified_llm_check(
            memoir_text, generated_text, retrieval_result
        )
        if llm_result:
            inconsistencies.extend(llm_result.inconsistencies)
            entity_coverage = max(entity_coverage, llm_result.entity_coverage)
            evidence_support = max(evidence_support, llm_result.evidence_support)

    # 去重
    inconsistencies = self._deduplicate_inconsistencies(inconsistencies)
    
    # 计算最终结果
    result = self._compute_result(
        inconsistencies=inconsistencies,
        entity_coverage=entity_coverage,
        evidence_support=evidence_support,
        memoir_text=memoir_text,
        generated_text=generated_text,
    )
    
    self._result_cache[cache_key] = result
    return result
```

### 3.3 幻觉类型

系统识别的幻觉类型包括：

| 幻觉类型 | 严重程度 | 描述 |
|---------|---------|------|
| 时间错位 | 0.7 | 生成文本的时间与回忆录矛盾 |
| 时间延伸 | 0.2 | 生成文本引入了新的时间点作为背景补充 |
| 地点不匹配 | 0.5 | 生成文本出现不相关的地点 |
| 实体幻觉 | 0.4 | 生成文本提及未知人物/组织 |
| 无依据声明 | 0.3 | 缺乏证据支持的具体细节 |
| 事实矛盾 | 0.6 | 与回忆录或历史事实矛盾 |
| 数字差异 | 0.3 | 数字信息与原始文本不符 |

### 3.4 技术特点

- **分层检查**：规则检查作为快速预筛选，LLM检查作为深度分析
- **一次LLM调用**：统一的LLM检查 prompt，一次调用完成所有分析
- **分词级证据计算**：使用jieba分词计算词重叠率，提高证据支持度计算的准确性
- **结果去重**：基于类型+内容的hash去重，避免重复的不一致项
- **多维度评分**：综合考虑时间一致性、地点一致性、实体覆盖率、证据支持度等多个维度

### 3.5 性能优化

1. **缓存机制**：对相同输入的检查结果进行缓存，避免重复计算
2. **并行处理**：规则检查和LLM检查可以并行执行
3. **结果限制**：对数字差异检查结果进行数量限制，避免过多的轻微问题
4. **文本截断**：对长文本进行合理截断，减少LLM处理时间

## 4. 模块间协作

三个模块在系统中协同工作，形成完整的检索-生成-验证流程：

1. **关键词检索**：作为默认检索策略，提供快速、准确的初步结果
2. **混合检索**：在需要更全面结果时使用，结合关键词和向量检索的优势
3. **事实准确性判断**：对生成的历史背景进行事实性检查，确保内容的准确性

### 4.1 数据流

```
用户输入回忆录文本 → 解析上下文 → 执行检索（关键词/混合） → 生成历史背景 → 事实性检查 → 返回结果
```

### 4.2 性能考虑

- **检索速度**：关键词检索速度最快（~3ms），适合实时应用
- **结果质量**：混合检索在某些场景下可以提供更全面的结果
- **验证准确性**：事实性判断确保生成内容的可靠性，但会增加一定的处理时间

## 5. 未来优化方向

1. **混合检索权重优化**：探索动态调整关键词和向量检索的权重，根据查询类型自动选择最优权重
2. **向量模型优化**：尝试使用更适合中文的向量模型，如BGE-base-zh-v1.5，提高向量检索的Precision
3. **事实性判断增强**：增加更多类型的幻觉检测，如事件因果错误、人物关系错误等
4. **性能优化**：进一步优化向量检索的速度，减少混合检索的延迟
5. **多语言支持**：扩展中英文映射表，支持更多语言的跨语言检索

## 6. 结论

本项目实现了三个核心模块：关键词检索、混合检索和事实准确性判断，它们共同构成了一个完整的记忆图谱检索系统。通过评测结果可以看出，关键词检索在当前数据集上表现最佳，而混合检索和事实准确性判断则为系统提供了更多的功能和可靠性。

这些模块的实现不仅满足了基本的检索需求，还通过技术创新提高了系统的性能和准确性，为个人回忆录的智能检索和生成提供了有力的支持。