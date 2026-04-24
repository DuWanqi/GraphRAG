# entity_coverage 指标详解

## 一句话解释

**`entity_coverage` 评估的是：RAG 从知识图谱检索到的实体，有多少被用在了生成文本中。**

---

## 详细说明

### 计算公式

```python
# 1. 从检索结果中提取实体名
ref_entities = []
for e in ch.retrieval_result.entities[:15]:  # 取前15个
    name = e.get("name") or e.get("title")
    ref_entities.append(name)

# 2. 检查有多少实体出现在生成文本中
covered = sum(1 for e in ref_entities if e in generated_text)

# 3. 计算覆盖率
entity_coverage = covered / len(ref_entities)
```

### 数据流

```
知识图谱检索
    ↓
retrieval_result.entities = [
    {"name": "恢复高考", "description": "1977年..."},
    {"name": "邓小平", "description": "..."},
    {"name": "十一届三中全会", "description": "..."},
    {"name": "知青返城", "description": "..."},
    {"name": "高等教育改革", "description": "..."}
]
    ↓
提取实体名
    ↓
ref_entities = ["恢复高考", "邓小平", "十一届三中全会", "知青返城", "高等教育改革"]
    ↓
检查生成文本
    ↓
generated_text = "一九七七年十月，广播中传来恢复高考的消息，整个生产队的知青..."
    ↓
字符串匹配
    ↓
covered = 1  # 只有"恢复高考"在文本中
    ↓
entity_coverage = 1/5 = 0.2 (20%)
```

---

## 设计意图

### 原始假设
设计这个指标时，假设：
1. RAG 检索到的实体应该被用在生成文本中
2. 覆盖率越高 = RAG 利用得越好
3. 覆盖率低 = RAG 没用上

### 实际情况
但在回忆录场景中：
1. **RAG 检索的是宏观历史实体**（恢复高考、邓小平南巡、亚洲金融危机）
2. **生成文本主要保留原文的个人叙事实体**（老张、德明、林老板、窑洞）
3. **RAG 实体以背景方式融入**，而非直接罗列实体名

---

## 为什么覆盖率低（0-20%）

### 示例：第 1 章（1977年恢复高考）

**RAG 检索到的实体（5个）**：
```
1. 恢复高考
2. 邓小平
3. 十一届三中全会
4. 知青返城
5. 高等教育改革
```

**生成文本片段**：
```
一九七七年十月，广播中传来恢复高考的消息，整个生产队的知青
如同被点燃的火药，瞬间炸开了锅。我在塬上已经待了五年，心中
对回城的希望早已磨灭，然而这一刻，仿佛又燃起了一丝渴望。

赵德明拉着我一起复习。我们从公社中学借来几本旧课本...
老张队长得知后，特意将我们的活排得轻一些...
```

**字符串匹配结果**：
- ✅ "恢复高考" - 直接出现
- ❌ "邓小平" - 未出现
- ❌ "十一届三中全会" - 未出现
- ❌ "知青返城" - 未出现（虽然有"知青"，但不是完整匹配）
- ❌ "高等教育改革" - 未出现

**entity_coverage = 1/5 = 20%**

### 为什么其他实体没出现？

1. **"邓小平"、"十一届三中全会"**
   - 这是政策层面的历史背景
   - 在个人回忆录中，不会直接写"邓小平决定恢复高考"
   - 而是写"广播中传来恢复高考的消息"（个人视角）

2. **"知青返城"**
   - 虽然文本中有"知青"，但字符串匹配要求完整匹配"知青返城"
   - 实际上"知青如同被点燃的火药"已经暗示了政策影响

3. **"高等教育改革"**
   - 这是宏观政策术语
   - 在个人叙事中，体现为"收到了北京一所大学的录取通知书"

---

## 指标的局限性

### 1. 简单字符串匹配

```python
covered = sum(1 for e in ref_entities if e in generated_text)
```

**问题**：
- 无法识别语义等价："邓小平南巡" vs "南巡讲话" vs "那位老人画了一个圈"
- 无法识别部分匹配："知青返城" vs "知青"
- 无法识别隐含引用："改革开放的春风" 暗指邓小平的政策

### 2. 不区分实体重要性

所有实体权重相同：
- "恢复高考"（核心历史事件）= 1分
- "高等教育改革"（宏观政策术语）= 1分

但实际上，"恢复高考"在个人回忆录中更重要、更可能被提及。

### 3. 不考虑融入方式

只看"有没有出现"，不看"怎么出现"：
- 直接罗列："1977年邓小平决定恢复高考，十一届三中全会通过了高等教育改革方案" → 覆盖率高，但像历史课文
- 背景融入："广播中传来恢复高考的消息，整个生产队的知青如同被点燃的火药" → 覆盖率低，但更像回忆录

---

## 正确的解读方式

### ❌ 错误解读
```
entity_coverage = 20% → RAG 没用上 → 生成质量差
```

### ✅ 正确解读
```
entity_coverage = 20%
    ↓
检查 rag_used_in_text（新增字段）
    ↓
["恢复高考"]  ← 核心历史事件被提及
    ↓
检查 text_only_entities（新增字段）
    ↓
["老张", "德明", "张家塬", "窑洞", ...]  ← 原文细节保留
    ↓
结论：RAG 以背景方式有效融入，原文细节保留良好
```

### 判断标准

| entity_coverage | rag_used_in_text | text_only_entities | 判断 |
|---|---|---|---|
| 20% | ["恢复高考"] | ["老张", "德明", ...] | ✅ 好：核心事件提及，原文保留 |
| 0% | [] | ["老张", "德明", ...] | ⚠️ 警告：RAG 可能未生效 |
| 80% | ["恢复高考", "邓小平", "十一届三中全会", ...] | ["老张"] | ❌ 差：像历史课文，原文丢失 |
| 0% | [] | [] | ❌ 差：生成失败 |

---

## 改进方案

### 短期（已实现）
在报告中新增 `entities` 字段，分别列出：
- `rag_retrieved` - RAG检索到的实体
- `rag_used_in_text` - 在文本中使用的RAG实体
- `text_extracted` - 文本中的所有实体
- `text_only_entities` - 仅在文本中的实体

### 中期
1. **降低 entity_coverage 权重**
   ```python
   # 当前
   综合分 = entity_coverage * 0.1 + ...
   
   # 建议
   综合分 = entity_coverage * 0.05 + ...  # 降低权重
   ```

2. **新增 background_integration 指标**
   ```python
   def background_integration(generated_text, retrieval_result):
       """评估历史背景融入程度（而非简单实体匹配）"""
       score = 0.0
       
       # 1. 检查时代关键词
       era_keywords = ["时代", "背景", "政策", "浪潮", "变革"]
       if any(kw in generated_text for kw in era_keywords):
           score += 0.3
       
       # 2. 检查核心历史事件（宽松匹配）
       for entity in retrieval_result.entities[:5]:  # 只看前5个核心实体
           name = entity.get("name")
           # 直接匹配或部分匹配
           if name in generated_text or any(part in generated_text for part in name.split()):
               score += 0.2
       
       # 3. 检查原文细节保留
       memoir_entities = extract_memoir_entities(memoir_text)
       text_entities = extract_text_entities(generated_text)
       retention_rate = len(set(memoir_entities) & set(text_entities)) / len(memoir_entities)
       score += retention_rate * 0.5
       
       return min(1.0, score)
   ```

### 长期
训练专门的"历史背景融入度"评估模型：
- 输入：回忆录原文 + 检索背景 + 生成文本
- 输出：融入度评分（0-1）
- 训练数据：人工标注的好/坏案例

---

## 总结

### entity_coverage 评估什么？
**RAG 检索到的实体，有多少被用在了生成文本中（字符串匹配）**

### 为什么覆盖率低？
1. RAG实体（宏观历史）vs 文本实体（个人叙事）在不同语义层面
2. RAG 以背景方式融入，而非直接罗列实体名
3. 简单字符串匹配无法识别语义等价和隐含引用

### 如何正确解读？
- ❌ 不要只看数值（20% ≠ 差）
- ✅ 结合 `rag_used_in_text` 和 `text_only_entities` 综合判断
- ✅ 关注核心历史事件是否被提及
- ✅ 关注原文细节是否被保留

### 核心洞察
**好的回忆录生成 = 保留原文细节（70-80%）+ 融入关键历史事件（1-2个）**

而不是：罗列所有 RAG 检索到的历史实体（那样会变成历史课文）
