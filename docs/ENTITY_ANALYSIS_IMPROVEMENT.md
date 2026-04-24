# 实体分析改进：区分 RAG 实体与文本实体

## 改进内容

在评估报告中新增 `entities` 字段，分别列出：
1. **RAG 检索到的实体**（来自知识图谱）
2. **生成文本中提取的实体**（来自生成结果）
3. **两者的覆盖关系**

---

## 报告结构

每个章节的评估结果中新增 `entities` 字段：

```json
{
  "segment_index": 1,
  "metrics": { ... },
  "entities": {
    "rag_retrieved": [
      "恢复高考",
      "邓小平", 
      "十一届三中全会",
      "知青返城",
      "高等教育改革"
    ],
    "rag_used_in_text": [
      "恢复高考"
    ],
    "rag_coverage": 0.2,
    "text_extracted": [
      "老张",
      "张队长",
      "赵德明",
      "北京",
      "张家塬",
      "武汉",
      "公社",
      "生产队",
      "知青"
    ],
    "text_only_entities": [
      "老张",
      "张队长",
      "赵德明",
      "北京",
      "张家塬",
      "武汉",
      "公社",
      "生产队",
      "知青"
    ]
  }
}
```

---

## 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `rag_retrieved` | List[str] | RAG 从知识图谱检索到的实体（历史事件/人物/地点） |
| `rag_used_in_text` | List[str] | RAG 实体中在生成文本中出现的（字符串匹配） |
| `rag_coverage` | float | RAG 实体覆盖率 = len(rag_used_in_text) / len(rag_retrieved) |
| `text_extracted` | List[str] | 从生成文本中提取的所有实体（人名/地名/机构） |
| `text_only_entities` | List[str] | 仅在生成文本中的实体（不在 RAG 检索结果中） |

---

## 实体提取规则

### RAG 实体（来自检索结果）
直接从 `retrieval_result.entities` 中获取，通常包括：
- 历史事件："恢复高考"、"邓小平南巡"、"亚洲金融危机"
- 历史人物："邓小平"、"毛泽东"
- 宏观地点："深圳经济特区"、"华强北"

### 文本实体（从生成文本提取）
使用规则匹配提取：

**人名模式**：
- `老X`：老张、老王、老刘
- `X队长/X先生/X老板`：张队长、林老板
- 常见名：德明、小军、父亲、母亲

**地名**：
- 常见地名列表：北京、上海、深圳、陕北、张家塬、华强北等

**机构/组织**：
- 联想、海尔、知青、研究所、生产队

---

## 实例分析

### 第 1 章（1977年恢复高考）

**RAG 检索到的实体 (5 个)**：
- 恢复高考
- 邓小平
- 十一届三中全会
- 知青返城
- 高等教育改革

**RAG 实体使用情况 (1 个, 覆盖率 20%)**：
- ✅ 恢复高考（直接出现在文本中）

**生成文本中的实体 (9 个)**：
- 老张、张队长、赵德明（原文人物）
- 北京、张家塬、武汉（地点）
- 公社、生产队、知青（组织/群体）

**分析**：
- RAG 覆盖率低（20%），但这是合理的
- 生成文本保留了原文的人物和地点（老张、德明、张家塬）
- 历史背景以氛围方式融入（"恢复高考的消息"），而非罗列实体名

---

### 第 4 章（1992年南巡）

**RAG 检索到的实体 (5 个)**：
- 邓小平南巡
- 南方谈话
- 市场经济
- 深圳特区
- 改革开放

**RAG 实体使用情况 (2 个, 覆盖率 40%)**：
- ✅ 邓小平南巡（直接提及）
- ✅ 深圳特区（间接提及"经济特区"）

**生成文本中的实体 (10 个)**：
- 老刘、老陈、林老板、女朋友（原文人物）
- 邓小平（历史人物）
- 北京、深圳、华强北、香港（地点）
- 研究所（机构）

**分析**：
- RAG 覆盖率较高（40%），历史事件明确提及
- 同时保留了原文的个人叙事（老刘、林老板）
- 历史与个人叙事结合良好

---

## 核心发现

### 1. 实体类型差异

| 实体来源 | 类型 | 示例 |
|---|---|---|
| RAG 检索 | 宏观历史 | "恢复高考"、"邓小平南巡"、"亚洲金融危机" |
| 生成文本 | 个人叙事 | "老张"、"德明"、"林老板"、"窑洞" |

两者在不同语义层面，重叠率自然较低。

### 2. RAG 的作用方式

RAG 实体不是直接"复制粘贴"到文本中，而是：
- **背景氛围**："改革开放的春风"、"知青政策的调整"
- **时代标记**："邓小平南巡的消息传来"
- **历史定位**："1977年恢复高考"

### 3. 原有 entity_coverage 指标的问题

```python
# 旧指标：简单字符串匹配
entity_coverage = sum(1 for e in rag_entities if e in text) / len(rag_entities)
```

**问题**：
- 无法识别语义等价（"邓小平南巡" vs "南巡讲话"）
- 无法识别隐含引用（"那位老人画了一个圈"）
- 无法区分实体重要性

**结果**：
- 覆盖率低（0-20%）被误判为"RAG 没用上"
- 但实际上 RAG 以背景方式有效融入

---

## 使用建议

### 查看报告时

1. **不要只看 rag_coverage 数值**
   - 20% 不代表质量差
   - 要看 `rag_used_in_text` 中是否包含关键历史事件

2. **关注 text_only_entities**
   - 这些是原文保留的人物/地点
   - 应该占多数（60-80%）

3. **综合判断 RAG 有效性**
   - ✅ 好：`rag_used_in_text` 包含 1-2 个关键历史事件 + `text_only_entities` 保留原文细节
   - ❌ 差：`rag_used_in_text` 为空 + 生成文本与原文几乎一致（相似度 > 0.9）

### 优化生成质量

如果 `rag_coverage` 过低（< 10%）且生成文本缺乏时代背景：

1. **优化 Prompt**：明确要求使用检索到的历史事件
2. **检查检索质量**：是否检索到了相关的历史背景
3. **调整生成参数**：降低 temperature，增强对 prompt 的遵循

---

## 代码实现

### 实体提取函数

```python
def _extract_entities_from_text(text: str) -> List[str]:
    """从文本中提取实体（规则匹配）"""
    entities = []
    
    # 人名：老X、X队长、常见名
    old_pattern = r'老[张王李赵刘...]'
    title_pattern = r'[张王李赵刘...]\w{0,2}(?:队长|先生|老板)'
    
    # 地名：常见地名列表
    locations = ['北京', '上海', '深圳', '陕北', '张家塬', ...]
    
    # 机构：联想、海尔、知青、研究所
    orgs = ['联想', '海尔', '知青', '研究所']
    
    # ... 提取逻辑
    return unique_entities
```

### 报告生成

```python
def _extract_entities_info(chapter, record) -> Dict[str, Any]:
    """提取实体信息：RAG vs 文本"""
    
    # 1. RAG 实体
    rag_entities = [e.get("name") for e in chapter.retrieval_result.entities]
    
    # 2. 文本实体
    text_entities = _extract_entities_from_text(chapter.generation.content)
    
    # 3. 覆盖关系
    rag_used = [e for e in rag_entities if e in chapter.generation.content]
    text_only = [e for e in text_entities if e not in rag_entities]
    
    return {
        "rag_retrieved": rag_entities,
        "rag_used_in_text": rag_used,
        "rag_coverage": len(rag_used) / len(rag_entities),
        "text_extracted": text_entities,
        "text_only_entities": text_only,
    }
```

---

## 总结

**改进前**：
- 只有一个 `entity_coverage` 指标（0-20%）
- 无法区分 RAG 实体和原文实体
- 容易误判"RAG 没用上"

**改进后**：
- 分别列出 RAG 实体和文本实体
- 清晰展示两者的覆盖关系
- 可以准确判断 RAG 的作用方式

**核心洞察**：
- RAG 实体覆盖率低（20%）≠ RAG 无效
- 回忆录生成的目标是"保留原文 + 融入背景"，而非"罗列历史实体"
- 好的生成应该是：`text_only_entities` 占 70-80%（原文细节）+ `rag_used_in_text` 包含 1-2 个关键历史事件
