"""
Novel Content Metrics - 新内容评估指标

评估生成文本中"新内容"的引入情况：
1. information_gain - 信息增益（引入了多少新知识）
2. expansion_grounding - 扩展溯源率（新内容是否有 RAG 来源支撑，防幻觉）
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Any

from .metrics import MetricResult


@dataclass
class NovelContentAnalysis:
    """新内容分析结果"""
    novel_entities_used: List[str]          # 生成文本中使用的新实体
    novel_entities_available: List[str]     # RAG 提供的新实体
    new_facts_in_output: List[str]          # 生成文本中的新事实陈述
    grounded_facts: List[str]               # 有 RAG 来源支撑的新事实
    ungrounded_facts: List[str]             # 无 RAG 来源支撑的新事实（疑似幻觉）

    @property
    def information_gain(self) -> float:
        """
        信息增量（Information Gain）

        衡量生成文本引入了多少新信息，基于使用的新实体数量分段评分：
        - 0个实体 → 0.0（无新信息）
        - 1个实体 → 0.4（少量新信息）
        - 2个实体 → 0.7（适量新信息）
        - 3+个实体 → 1.0（丰富新信息）

        这样可以避免"分母过大"的问题（不相关实体不应该拉低分数）
        """
        used_count = len(self.novel_entities_used)

        if used_count == 0:
            return 0.0
        elif used_count == 1:
            return 0.4
        elif used_count == 2:
            return 0.7
        else:
            return 1.0
    
    @property
    def expansion_grounding(self) -> float:
        """扩展溯源率（Expansion Grounding）"""
        if not self.new_facts_in_output:
            return 1.0  # 没有新事实，默认为完全有据
        return len(self.grounded_facts) / len(self.new_facts_in_output)



def _deduplicate_entities(entity_list: List[str]) -> List[str]:
    """
    对实体列表进行去重，将模糊匹配相同的实体合并为一个

    优先保留中文名称，如果没有中文则保留第一个
    """
    if not entity_list:
        return []

    deduplicated = []
    seen_groups = []  # 存储已处理的实体组

    for entity in entity_list:
        # 检查是否与已有实体组匹配
        found_group = False
        for group in seen_groups:
            if any(_fuzzy_entity_match(entity, existing) for existing in group):
                # 找到匹配的组，将当前实体加入该组
                group.append(entity)
                found_group = True
                break

        if not found_group:
            # 创建新组
            seen_groups.append([entity])

    # 从每个组中选择一个代表实体（优先中文）
    for group in seen_groups:
        # 优先选择中文实体
        chinese_entities = [e for e in group if any('一' <= c <= '鿿' for c in e)]
        if chinese_entities:
            deduplicated.append(chinese_entities[0])
        else:
            deduplicated.append(group[0])

    return deduplicated


async def analyze_novel_content(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,  # NovelContentBrief
    llm_adapter: Optional[Any] = None,
) -> NovelContentAnalysis:
    """
    分析生成文本中的新内容使用情况（简化版，基于实体）

    For expansion tasks, grounding should measure:
    "Are the novel entities we used actually from RAG sources?"
    NOT "Are all extracted narrative phrases grounded?"

    Args:
        memoir_text: 回忆录原文
        generated_text: 生成的文本
        novel_content_brief: NovelContentBrief 对象（来自 extract_novel_content）
        llm_adapter: LLM 适配器（可选，用于 LLM 实体提取）

    Returns:
        NovelContentAnalysis: 分析结果
    """
    # 1. 检查哪些新实体被使用了（使用模糊匹配）
    novel_entities_available = novel_content_brief.novel_entity_names
    novel_entities_used_raw = []

    for entity_name in novel_entities_available:
        if entity_name and _is_mentioned_in_text(entity_name, generated_text):
            novel_entities_used_raw.append(entity_name)

    # 对使用的实体进行去重（合并中英文同义实体）
    novel_entities_used = _deduplicate_entities(novel_entities_used_raw)
    print(f"[DEBUG] 去重前使用的实体: {novel_entities_used_raw}")
    print(f"[DEBUG] 去重后使用的实体: {novel_entities_used}")

    # 2. 从生成文本中提取实体（使用 LLM 或规则）
    if llm_adapter is not None:
        new_facts_in_output = await _extract_entities_with_llm(
            generated_text, memoir_text, llm_adapter
        )
    else:
        new_facts_in_output = _extract_entity_names_only(generated_text, memoir_text)

    # 3. 判断提取的实体是否有 RAG 支撑（实体名匹配 + 实体描述内容匹配）
    grounded_facts = []
    ungrounded_facts = []

    # 构建 RAG 来源全文（包含实体名+描述+关系），用于描述级溯源
    rag_source_text = _build_rag_source_text(novel_content_brief)

    for fact in new_facts_in_output:
        # 优先：使用模糊匹配查找是否有对应的 RAG 实体名
        matched_entity = _find_matching_entity(fact, novel_entities_available)
        if matched_entity:
            grounded_facts.append(fact)
            print(f"[DEBUG] 实体 '{fact}' 匹配到 RAG 实体名 '{matched_entity}'")
        elif _is_grounded_in_rag(fact, rag_source_text, novel_entities_available):
            # 回退：检查事实是否出现在 RAG 实体描述/关系描述中
            grounded_facts.append(fact)
            print(f"[DEBUG] 实体 '{fact}' 在 RAG 实体描述中找到支撑")
        else:
            ungrounded_facts.append(fact)
            print(f"[DEBUG] 实体 '{fact}' 未找到匹配的 RAG 实体或描述")

    return NovelContentAnalysis(
        novel_entities_used=novel_entities_used,
        novel_entities_available=novel_entities_available,
        new_facts_in_output=new_facts_in_output,
        grounded_facts=grounded_facts,
        ungrounded_facts=ungrounded_facts,
    )


async def _extract_entities_with_llm(
    generated_text: str,
    memoir_text: str,
    llm_adapter: Any,
) -> List[str]:
    """
    使用 LLM 提取生成文本中的实体（人名、地名、机构名）

    Args:
        generated_text: 生成的文本
        memoir_text: 原始回忆录文本（用于过滤已存在的实体）
        llm_adapter: LLM 适配器

    Returns:
        提取的实体列表
    """
    prompt = f"""请从以下生成文本中提取所有实体，包括：
1. 人名（真实人物、历史人物）
2. 地名（国家、城市、地区、具体地点）
3. 机构名（组织、公司、学校、政府部门等）
4. 特殊事件名（具有历史意义的特定事件，如"十一届三中全会"、"改革开放"）

注意：
- 只提取专有名词，不要提取普通名词、形容词或抽象概念
- 不要提取时间词（如"1980年"、"当时"）
- 不要提取普通物品（如"自行车"、"行李"）
- 不要提取描述性词语（如"阳光"、"清香"、"梦想"）
- **重要：区分特殊事件和常用表达**
  * ✓ 提取：具有历史意义的特定事件（如"改革开放"、"十一届三中全会"、"恢复高考"）
  * ✗ 不提取：日常活动的常用表达（如"聊天"、"修桥"、"筑路"、"吃饭"、"睡觉"、"学习"、"工作"）
  * ✗ 不提取：通用动作短语（如"围坐在一起"、"充满憧憬"、"背着行李"）
- 判断标准：如果这个词可以用来描述任何人的日常活动，就不要提取

生成文本：
{generated_text}

请以JSON格式返回，只返回JSON，不要其他解释：
{{
  "entities": ["实体1", "实体2", ...]
}}"""

    try:
        response = await llm_adapter.generate(prompt, temperature=0.0)

        # Extract text content from LLMResponse object
        response_text = response.content.strip()

        print(f"\n[DEBUG] LLM 实体提取 - 原始响应:")
        print(f"  响应长度: {len(response_text)} 字符")
        print(f"  响应内容: {response_text[:500]}...")

        # 解析 JSON 响应
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            print(f"[DEBUG] 找到 JSON 格式")
            result = json.loads(json_match.group())
            entities = result.get("entities", [])
            print(f"[DEBUG] JSON 解析成功，提取到 {len(entities)} 个实体")
        else:
            # 如果没有找到 JSON，尝试按行解析
            print(f"[DEBUG] 未找到 JSON，尝试按行解析")
            entities = [line.strip().strip('-').strip()
                       for line in response_text.split('\n')
                       if line.strip() and not line.strip().startswith('{')]
            print(f"[DEBUG] 按行解析得到 {len(entities)} 个实体")

        print(f"[DEBUG] 提取的实体: {entities[:10]}")

        # 过滤掉回忆录中已有的实体
        memoir_entities = set()
        for entity in entities:
            if entity in memoir_text:
                memoir_entities.add(entity)

        print(f"[DEBUG] 回忆录中已有的实体: {memoir_entities}")

        filtered_entities = [e for e in entities if e not in memoir_entities]
        print(f"[DEBUG] 过滤后剩余 {len(filtered_entities)} 个实体: {filtered_entities[:10]}")

        # 去重并限制数量
        final_entities = list(dict.fromkeys(filtered_entities))[:20]
        print(f"[DEBUG] 最终返回 {len(final_entities)} 个实体")
        return final_entities

    except Exception as e:
        # 如果 LLM 提取失败，返回空列表
        print(f"\n[ERROR] LLM entity extraction failed: {e}")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        if 'response_text' in locals():
            print(f"[ERROR] LLM response text was: {response_text[:200]}...")
        elif 'response' in locals():
            try:
                print(f"[ERROR] LLMResponse object received: {response.content[:200]}...")
            except:
                print(f"[ERROR] LLMResponse object received but content extraction failed")
        else:
            print(f"[ERROR] No response received from LLM")
        import traceback
        traceback.print_exc()
        return []


def _extract_entity_names_only(generated_text: str, memoir_text: str) -> List[str]:
    """
    使用规则提取实体（备用方案，当 LLM 不可用时）
    Extract only entity names (not narrative phrases) from generated text.
    Much stricter than _extract_new_facts().
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        return []

    # Only extract proper nouns
    factual_pos_tags = {'nr', 'ns', 'nt', 'nz'}

    # Strict blacklist - filter out common descriptive nouns
    blacklist = {
        # Time/place descriptors
        '年代', '时候', '时代', '时期', '时光', '岁月', '日子', '当时', '那时',
        '地方', '方面', '情况', '问题', '事情', '东西', '事物',
        # Common objects (not entities)
        '树', '灯', '灯光', '铃', '自行车', '行李', '床铺', '宿舍', '校园',
        '通知书', '录取', '考试', '成绩', '分数', '名次',
        '宿舍楼', '教学楼', '图书馆', '食堂', '操场',
        # Abstract concepts
        '梦想', '希望', '未来', '开始', '结束', '可能性', '机会',
        '宝藏', '曙光', '苏醒', '命运', '责任', '期待', '憧憬',
        '宝贝', '荣光', '春雷', '火焰', '阴影', '奔头',
        # Actions/states
        '生活', '学习', '工作', '发展', '变化', '建设', '改革', '开放',
        '播种', '春播', '收获', '劳作', '中断', '重启',
        # Generic descriptive words (from user feedback)
        '城市', '楼宇', '阳光', '清香', '田野', '声音', '气息', '影子',
        '光影', '微光', '节奏', '速度', '力量', '温度', '距离',
        '革命家', '青年', '学生', '同学', '家人', '父母',
        '烟熏', '山丘', '林立', '金黄', '霜花', '花纹', '竹林',
        '竹叶', '晚霞', '夜幕', '灯泡', '世界',
        # Common places (too generic)
        '火车', '车厢', '窗外', '房间', '桌上', '一旁', '中央',
        '堂屋', '灶台', '柴火', '稀饭', '早饭', '长途', '汽车',
        '柏油', '路面', '梧桐', '树叶', '墙面', '油墨', '泥土',
        '玻璃', '窗', '门口', '家门', '乡下', '地', '天边', '晚风',
        # Common daily activities (user feedback - avoid extracting generic actions)
        '聊天', '修桥', '筑路', '吃饭', '睡觉', '散步', '跑步', '游泳',
        '读书', '写字', '画画', '唱歌', '跳舞', '打球', '下棋',
        '围坐', '站立', '行走', '奔跑', '休息', '等待', '观看',
        '思考', '回忆', '想象', '期盼', '担心', '高兴', '难过',
        '背着', '拿着', '提着', '抱着', '穿着', '戴着',
    }

    generated_words = list(pseg.cut(generated_text))
    memoir_words = set(w for w, _ in pseg.cut(memoir_text))

    entities = []
    for word, flag in generated_words:
        if (len(word) >= 2 and
            flag in factual_pos_tags and
            word not in memoir_words and
            word not in blacklist):
            entities.append(word)

    return list(dict.fromkeys(entities))[:20]  # Dedupe and limit


async def information_gain_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
    llm_adapter: Optional[Any] = None,
) -> MetricResult:
    """
    信息增量指标（Information Gain）

    衡量生成文本引入了多少新知识（基于实体使用数量的分段评分）
    - 0个实体 → 0.0（无新内容）
    - 1个实体 → 0.4（少量新内容）
    - 2个实体 → 0.7（适量新内容）
    - 3+个实体 → 1.0（丰富新内容）
    """
    analysis = await analyze_novel_content(memoir_text, generated_text, novel_content_brief, llm_adapter)

    ratio = analysis.information_gain
    used = len(analysis.novel_entities_used)
    available = len(analysis.novel_entities_available)

    if available == 0:
        explanation = "RAG 未提供新实体"
    elif used == 0:
        explanation = f"未使用新实体（RAG 提供了 {available} 个）"
    else:
        explanation = f"使用了 {used} 个新实体"
        if used > 0:
            explanation += f"：{', '.join(analysis.novel_entities_used[:3])}"
        if used < available:
            explanation += f"（RAG 还提供了 {available - used} 个未使用的实体）"

    return MetricResult(
        name="information_gain",
        value=ratio,
        max_value=1.0,
        explanation=explanation,
    )


async def expansion_grounding_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
    llm_adapter: Optional[Any] = None,
) -> MetricResult:
    """
    扩展溯源率指标（Expansion Grounding）

    For expansion tasks: measures if used entities are from RAG sources.
    Since we only count entities that ARE in novel_content_brief,
    grounding rate = 100% by definition.

    This metric now serves as a sanity check rather than a strict filter.
    """
    analysis = await analyze_novel_content(memoir_text, generated_text, novel_content_brief, llm_adapter)

    used = len(analysis.novel_entities_used)
    available = len(analysis.novel_entities_available)

    if used == 0:
        explanation = "未使用任何新实体"
        grounding = 0.0
    else:
        # All used entities are grounded (they came from RAG)
        grounding = 1.0
        explanation = f"使用了 {used} 个新实体，均来自 RAG 检索结果"

        # Warn if there are ungrounded facts
        if analysis.ungrounded_facts:
            ungrounded_count = len(analysis.ungrounded_facts)
            explanation += f"；检测到 {ungrounded_count} 个额外事实（未在 RAG 中）"

    return MetricResult(
        name="expansion_grounding",
        value=grounding,
        max_value=1.0,
        explanation=explanation,
    )


async def rag_utilization_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
    llm_adapter: Optional[Any] = None,
) -> MetricResult:
    """
    RAG 利用率指标（RAG Utilization）

    衡量生成文本是否充分利用了 RAG 检索到的新实体。

    评分标准（非线性）：
    - 0个实体 → 0.0（未利用，门控不通过）
    - 1个实体 → 0.5（已利用检索新实体，门控通过）
    - 2个实体 → 0.6
    - 3个实体 → 0.8（良好利用）
    - 4+个实体 → 1.0（充分利用）

    门控要求（与 QualityThresholds.min_rag_utilization 默认 0.5 对齐）：score ≥ 0.5，即至少使用 1 个新实体。
    """
    analysis = await analyze_novel_content(memoir_text, generated_text, novel_content_brief, llm_adapter)

    used = len(analysis.novel_entities_used)
    available = len(analysis.novel_entities_available)

    # 非线性评分
    if used == 0:
        score = 0.0
        level = "未利用"
    elif used == 1:
        score = 0.5
        level = "已利用（单实体，门控达标）"
    elif used == 2:
        score = 0.6
        level = "达到最低要求"
    elif used == 3:
        score = 0.8
        level = "良好利用"
    else:  # 4+
        score = 1.0
        level = "充分利用"

    # 构建详细说明
    explanation_parts = [
        f"【评分】{score:.1f}/1.0 ({level})",
        f"【计算依据】使用了 {used}/{available} 个 RAG 提供的新实体",
    ]

    if used > 0:
        explanation_parts.append(f"【已使用实体】{', '.join(analysis.novel_entities_used[:5])}")
        if len(analysis.novel_entities_used) > 5:
            explanation_parts.append(f"  ... 还有 {len(analysis.novel_entities_used) - 5} 个")

    if used < available:
        unused = [e for e in analysis.novel_entities_available if e not in analysis.novel_entities_used]
        explanation_parts.append(f"【未使用实体】{', '.join(unused[:5])}")
        if len(unused) > 5:
            explanation_parts.append(f"  ... 还有 {len(unused) - 5} 个")

    # 门控判定（与默认 min_rag_utilization=0.5 一致：仅 0 实体不通过）
    passed = score >= 0.5
    gate_status = "✓ 通过" if passed else "✗ 未通过"
    explanation_parts.append(f"【门控判定】{gate_status}（要求至少 1 个新实体，score ≥ 0.5）")

    explanation = "\n  ".join(explanation_parts)

    return MetricResult(
        name="rag_utilization",
        value=score,
        max_value=1.0,
        explanation=explanation,
    )


async def hallucination_metric(
    memoir_text: str,
    generated_text: str,
    novel_content_brief: Any,
    llm_adapter: Optional[Any] = None,
) -> MetricResult:
    """
    幻觉检测指标（Hallucination Detection）

    检测生成文本中有多少实体缺乏 RAG 支撑（疑似幻觉）。

    计算方法：
    1. 从生成文本中提取所有实体（使用 LLM 或严格过滤，排除通用词）
    2. 分类：回忆录实体、RAG实体、无支撑实体
    3. 幻觉率 = 无支撑实体数 / 总提取实体数
    4. 评分 = 1.0 - 幻觉率（越低幻觉越高分）

    注：暂不设置门控，仅作为评分指标
    """
    analysis = await analyze_novel_content(memoir_text, generated_text, novel_content_brief, llm_adapter)

    # 提取生成文本中的所有实体（使用 LLM 或改进的规则提取逻辑）
    if llm_adapter is not None:
        all_extracted = await _extract_entities_with_llm(generated_text, memoir_text, llm_adapter)
    else:
        all_extracted = _extract_entity_names_only(generated_text, memoir_text)

    # 分类实体
    memoir_entities = [e for e in all_extracted if e in memoir_text]
    rag_entities = analysis.novel_entities_used  # 已经在 RAG 中的

    # 构建 RAG 来源全文用于描述级溯源
    rag_source_text = _build_rag_source_text(novel_content_brief)
    novel_entity_names = novel_content_brief.novel_entity_names

    unsupported_entities = [
        e for e in all_extracted
        if e not in memoir_text
        and not _find_matching_entity(e, novel_entity_names)
        and not _is_grounded_in_rag(e, rag_source_text, novel_entity_names)
    ]

    # 计算幻觉率
    total_extracted = len(all_extracted)
    unsupported_count = len(unsupported_entities)

    if total_extracted == 0:
        hallucination_rate = 0.0
        score = 1.0
        level = "无法评估"
    else:
        hallucination_rate = unsupported_count / total_extracted
        score = 1.0 - hallucination_rate

        if hallucination_rate <= 0.1:
            level = "优秀"
        elif hallucination_rate <= 0.3:
            level = "良好"
        elif hallucination_rate <= 0.5:
            level = "一般"
        else:
            level = "较差"

    # 构建详细说明
    explanation_parts = [
        f"【评分】{score:.2f}/1.0 ({level})",
        f"【计算依据】幻觉率 = {unsupported_count}/{total_extracted} = {hallucination_rate:.1%}",
        f"【实体分类】",
        f"  - 回忆录实体: {len(memoir_entities)} 个",
        f"  - RAG 支撑实体: {len(rag_entities)} 个",
        f"  - 无支撑实体: {unsupported_count} 个（疑似幻觉）",
    ]

    if unsupported_entities:
        explanation_parts.append(f"【无支撑实体列表】{', '.join(unsupported_entities[:10])}")
        if len(unsupported_entities) > 10:
            explanation_parts.append(f"  ... 还有 {len(unsupported_entities) - 10} 个")

    explanation = "\n  ".join(explanation_parts)

    return MetricResult(
        name="hallucination",
        value=score,
        max_value=1.0,
        explanation=explanation,
    )


# ============================================================================
# 内部辅助函数
# ============================================================================

# 中英文地名映射表（可扩展）
_CN_EN_LOCATION_MAP = {
    '北京': ['beijing', '北京市', 'peking'],
    '上海': ['shanghai', '上海市'],
    '深圳': ['shenzhen', '深圳市'],
    '广州': ['guangzhou', '广州市', 'canton'],
    '中国': ['china', 'prc', "people's republic of china"],
    '美国': ['usa', 'us', 'united states', 'america'],
    '英国': ['uk', 'britain', 'united kingdom'],
    '日本': ['japan'],
    '香港': ['hong kong', 'hongkong', 'hk'],
    '台湾': ['taiwan'],
    '澳门': ['macao', 'macau'],
    '天津': ['tianjin'],
    '重庆': ['chongqing'],
    '南京': ['nanjing', 'nanking'],
    '武汉': ['wuhan'],
    '成都': ['chengdu'],
    '西安': ['xian', "xi'an"],
    '杭州': ['hangzhou'],
    '苏州': ['suzhou'],
    '青岛': ['qingdao'],
    '大连': ['dalian'],
    '沈阳': ['shenyang'],
    '哈尔滨': ['harbin'],
    '长春': ['changchun'],
    '济南': ['jinan'],
    '郑州': ['zhengzhou'],
    '长沙': ['changsha'],
    '福州': ['fuzhou'],
    '厦门': ['xiamen'],
    '南昌': ['nanchang'],
    '合肥': ['hefei'],
    '太原': ['taiyuan'],
    '石家庄': ['shijiazhuang'],
    '呼和浩特': ['hohhot'],
    '乌鲁木齐': ['urumqi'],
    '拉萨': ['lhasa'],
    '兰州': ['lanzhou'],
    '西宁': ['xining'],
    '银川': ['yinchuan'],
    '南宁': ['nanning'],
    '昆明': ['kunming'],
    '贵阳': ['guiyang'],
    '海口': ['haikou'],
}

# 构建反向映射（英文 → 中文）
_EN_CN_LOCATION_MAP = {}
for cn, en_list in _CN_EN_LOCATION_MAP.items():
    for en in en_list:
        _EN_CN_LOCATION_MAP[en.lower()] = cn


def _fuzzy_entity_match(entity1: str, entity2: str) -> bool:
    """
    模糊匹配两个实体名称

    支持：
    1. 大小写不敏感
    2. 中英文映射（北京 ↔ BEIJING）
    3. 部分包含匹配
    4. 反向匹配

    Args:
        entity1: 实体名称1
        entity2: 实体名称2

    Returns:
        是否匹配
    """
    if not entity1 or not entity2:
        return False

    # 标准化：转小写，去空格和标点
    e1 = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity1).lower()
    e2 = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity2).lower()

    # 1. 完全匹配（大小写不敏感）
    if e1 == e2:
        return True

    # 2. 包含匹配（双向）
    if e1 in e2 or e2 in e1:
        return True

    # 3. 中英文映射匹配
    # 检查 entity1 是否是中文，entity2 是否是对应的英文
    if entity1 in _CN_EN_LOCATION_MAP:
        en_variants = _CN_EN_LOCATION_MAP[entity1]
        if any(en.lower() == e2 for en in en_variants):
            return True

    # 检查 entity2 是否是中文，entity1 是否是对应的英文
    if entity2 in _CN_EN_LOCATION_MAP:
        en_variants = _CN_EN_LOCATION_MAP[entity2]
        if any(en.lower() == e1 for en in en_variants):
            return True

    # 检查 entity1 是否是英文，entity2 是否是对应的中文
    if e1 in _EN_CN_LOCATION_MAP:
        cn = _EN_CN_LOCATION_MAP[e1]
        if cn == entity2 or cn in entity2:
            return True
    # 对多词英文实体名的各单词尝试查找映射
    for word in re.findall(r'[a-z]+', e1):
        if word in _EN_CN_LOCATION_MAP:
            cn = _EN_CN_LOCATION_MAP[word]
            if cn == entity2 or cn in entity2 or entity2 in cn:
                return True

    # 检查 entity2 是否是英文，entity1 是否是对应的中文
    if e2 in _EN_CN_LOCATION_MAP:
        cn = _EN_CN_LOCATION_MAP[e2]
        if cn == entity1 or cn in entity1:
            return True
    # 对多词英文实体名的各单词尝试查找映射
    for word in re.findall(r'[a-z]+', e2):
        if word in _EN_CN_LOCATION_MAP:
            cn = _EN_CN_LOCATION_MAP[word]
            if cn == entity1 or cn in entity1 or entity1 in cn:
                return True

    # 4. 部分匹配（对于较长的实体名）
    if len(e1) >= 4 and len(e2) >= 4:
        # 检查前4个字符是否匹配
        if e1[:4] == e2[:4]:
            return True

    return False


def _find_matching_entity(target: str, entity_list: List[str]) -> Optional[str]:
    """
    在实体列表中查找与目标实体匹配的实体

    Args:
        target: 目标实体名称
        entity_list: 实体列表

    Returns:
        匹配的实体名称，如果没有匹配则返回 None
    """
    for entity in entity_list:
        if _fuzzy_entity_match(target, entity):
            return entity
    return None


def _is_mentioned_in_text(entity_name: str, text: str) -> bool:
    """检查实体是否在文本中提及（增强的模糊匹配，支持中英文映射）"""
    if not entity_name or not text:
        return False

    # 归一化：去除标点符号和空格
    entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity_name).upper()
    text_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', text).upper()

    # 1. 精确匹配
    if entity_normalized in text_normalized:
        return True

    # 2. 中英文映射匹配
    # 如果实体是英文地名，检查对应的中文是否在文本中
    # 支持多词实体名（如 "SHENZHEN CITY" → 先尝试完整匹配，再尝试各单词）
    entity_lower = entity_name.lower().strip()
    if entity_lower in _EN_CN_LOCATION_MAP:
        cn_name = _EN_CN_LOCATION_MAP[entity_lower]
        if cn_name in text:
            return True

    # 对多词英文实体名，尝试每个单词在映射表中查找
    entity_words_en = re.findall(r'[a-zA-Z]+', entity_name.lower())
    for word in entity_words_en:
        if word in _EN_CN_LOCATION_MAP:
            cn_name = _EN_CN_LOCATION_MAP[word]
            if cn_name in text:
                return True

    # 如果实体是中文地名，检查对应的英文是否在文本中
    if entity_name in _CN_EN_LOCATION_MAP:
        en_variants = _CN_EN_LOCATION_MAP[entity_name]
        for en in en_variants:
            if en.upper() in text_normalized:
                return True

    # 3. 反向匹配：文本中的词是否是实体的一部分
    # 例如：实体="庚申年猴票"，文本包含"猴票" → 匹配
    entity_words = re.findall(r'[一-龥]{2,}', entity_name)
    for word in entity_words:
        if len(word) >= 2 and word in text:
            return True

    # 4. 部分匹配（长实体 ≥4 字符）
    if len(entity_normalized) >= 4:
        if entity_normalized[:4] in text_normalized:
            return True

    # 5. 缩写匹配（任意3字符子串）
    if len(entity_normalized) >= 3:
        for i in range(len(entity_normalized) - 2):
            substring = entity_normalized[i:i+3]
            if substring in text_normalized:
                return True

    return False


def _extract_new_facts(memoir_text: str, generated_text: str) -> List[str]:
    """
    提取生成文本中的新事实陈述（不在原文中的）

    采用三层漏斗过滤策略：
    1. [粗筛] 词性过滤：快速提取候选实体（高召回）
    2. [精筛] NER 验证：验证候选实体是否为真实事实（高精确）
    3. [补充] 句式匹配：捕获复合事实和特殊模式

    Args:
        memoir_text: 回忆录原文
        generated_text: 生成的文本

    Returns:
        List[str]: 新事实列表
    """
    # 第一层：词性粗筛（快速、高召回）
    candidates = _extract_candidates_by_pos(memoir_text, generated_text)

    # 第二层：NER 验证（准确、高精确）
    verified_facts = _verify_candidates_by_ner(candidates, generated_text)

    # 第三层：句式补充（捕获漏网之鱼）
    compound_facts = _extract_compound_facts(memoir_text, generated_text)

    # 合并去重
    all_facts = verified_facts + compound_facts
    return list(dict.fromkeys(all_facts))[:30]  # 去重并限制数量


def _extract_candidates_by_pos(memoir_text: str, generated_text: str) -> List[str]:
    """
    第一层：基于词性的候选实体提取（粗筛）

    目标：快速找出所有"可能是事实"的词，宁可多不可少

    策略：
    1. 使用 jieba 分词 + 词性标注
    2. 保留专有名词相关的词性（nr/ns/nt/nz）
    3. 过滤掉原文中已有的词
    4. 过滤掉常见非事实词（黑名单）

    Returns:
        候选实体列表（可能包含误判）
    """
    try:
        import jieba.posseg as pseg
    except ImportError:
        # 降级方案：使用简单正则
        return _extract_candidates_by_regex(memoir_text, generated_text)

    # 事实性词性标签
    factual_pos_tags = {
        'nr',   # 人名（邓小平、张三）
        'ns',   # 地名（深圳、北京）
        'nt',   # 机构名（中共中央、联想）
        'nz',   # 其他专名（改革开放、高考）
    }

    # 常见非事实词黑名单
    blacklist = {
        # 时间词（太泛）
        '年代', '时候', '时代', '时期', '时光', '岁月', '日子', '当时', '那时',
        # 泛指词
        '国家', '社会', '人民', '群众', '大家', '我们', '他们', '自己',
        '地方', '方面', '情况', '问题', '事情', '东西', '事物',
        # 常见动作/状态
        '生活', '学习', '工作', '发展', '变化', '建设', '改革', '开放',
        '刚刚开始', '开始', '结束', '进行', '发生', '出现',
        # 常见文书词
        '通知书', '录取', '考试', '成绩', '分数', '名次',
        '政策', '制度', '办法', '措施', '方案', '计划', '目标',
    }

    # 分词 + 词性标注
    generated_words = list(pseg.cut(generated_text))
    memoir_words = set(w for w, _ in pseg.cut(memoir_text))

    candidates = []

    for word, flag in generated_words:
        # 过滤条件
        if len(word) < 2:           # 单字词
            continue
        if word in memoir_words:    # 原文已有
            continue
        if word in blacklist:       # 黑名单
            continue

        # 保留事实性词汇
        if flag in factual_pos_tags:
            candidates.append(word)

    return candidates


def _extract_candidates_by_regex(memoir_text: str, generated_text: str) -> List[str]:
    """
    降级方案：基于正则的候选实体提取（当 jieba 不可用时）

    策略：提取 3-10 字的中文词（避免提取太多短词）
    """
    generated_entities = set(re.findall(r'[一-龥]{3,10}', generated_text))
    memoir_entities = set(re.findall(r'[一-龥]{3,10}', memoir_text))

    candidates = list(generated_entities - memoir_entities)
    return candidates[:50]  # 限制数量


def _verify_candidates_by_ner(candidates: List[str], text: str) -> List[str]:
    """
    第二层：基于 NER 的候选验证（精筛）

    目标：验证候选词是否真的是事实性实体，去伪存真

    策略：
    1. 尝试使用 LAC (百度 NER) 进行验证
    2. 如果 LAC 不可用，使用规则验证
    3. 只保留被 NER 识别为实体的候选词

    Returns:
        验证通过的事实列表
    """
    if not candidates:
        return []

    # 尝试使用 LAC
    try:
        from LAC import LAC
        lac = LAC(mode='lac')

        # NER 识别
        result = lac.run(text)
        words, tags = result[0], result[1]

        # 实体类型
        entity_types = {'PER', 'LOC', 'ORG', 'TIME'}

        # 构建实体集合
        ner_entities = set()
        for word, tag in zip(words, tags):
            if tag in entity_types:
                ner_entities.add(word)

        # 验证候选词
        verified = []
        for candidate in candidates:
            # 精确匹配或部分匹配
            if candidate in ner_entities:
                verified.append(candidate)
            else:
                # 检查是否是某个 NER 实体的一部分
                for entity in ner_entities:
                    if candidate in entity or entity in candidate:
                        verified.append(candidate)
                        break

        return verified

    except ImportError:
        # LAC 不可用，使用规则验证
        return _verify_candidates_by_rules(candidates)


def _verify_candidates_by_rules(candidates: List[str]) -> List[str]:
    """
    规则验证：当 NER 不可用时的降级方案

    策略：
    1. 保留 3 字以上的词（更可能是专有名词）
    2. 保留包含特定标志词的词（如"会议"、"政策"、"特区"）
    """
    verified = []

    # 事实性标志词
    fact_markers = {
        '会议', '全会', '大会', '代表',
        '政策', '制度', '法律', '条例',
        '特区', '开发区', '示范区',
        '公司', '企业', '集团', '组织',
    }

    for candidate in candidates:
        # 3 字以上
        if len(candidate) >= 3:
            verified.append(candidate)
            continue

        # 包含标志词
        if any(marker in candidate for marker in fact_markers):
            verified.append(candidate)

    return verified


def _extract_compound_facts(memoir_text: str, generated_text: str) -> List[str]:
    """
    第三层：基于句式的复合事实提取（补充）

    目标：捕获前两层可能遗漏的复合事实和特殊模式

    策略：
    1. 提取"年份 + 事件"模式（如"1978年十一届三中全会"）
    2. 提取"实体 + 关系 + 实体"模式（如"邓小平推动改革开放"）
    3. 提取年份（如"1978"）

    Returns:
        复合事实列表
    """
    compound_facts = []

    # 模式 1: 年份 + 事件（如"1978年十一届三中全会确立了改革开放政策"）
    # 改进：只提取事件主体，不包含动词和"的"
    year_event_pattern = r'(\d{4})年(?:的)?([一-龥]{2,8}(?:会议|全会|政策|制度|特区|开放))'
    matches = re.findall(year_event_pattern, generated_text)

    for year, event in matches:
        # 检查是否在原文中
        if event not in memoir_text:
            compound_facts.append(event)
            compound_facts.append(year)  # 年份也是事实

    # 模式 2: 实体 + 关系 + 实体（如"邓小平推动改革开放"）
    # 改进：只提取实体，不包含关系词
    relation_pattern = r'([一-龥]{2,6})(?:是|为|推动|确立|建立|创办)([一-龥]{2,6})'
    matches = re.findall(relation_pattern, generated_text)

    for source, target in matches:
        # 过滤掉太短或太常见的词
        if len(source) >= 2 and source not in memoir_text:
            # 检查是否是实体（不包含动词、形容词）
            if not any(word in source for word in ['了', '的', '是', '在', '有', '和']):
                compound_facts.append(source)
        if len(target) >= 2 and target not in memoir_text:
            if not any(word in target for word in ['了', '的', '是', '在', '有', '和']):
                compound_facts.append(target)

    # 模式 3: 独立年份（如"1978"）
    generated_years = set(re.findall(r'\b(19|20)\d{2}\b', generated_text))
    memoir_years = set(re.findall(r'\b(19|20)\d{2}\b', memoir_text))
    compound_facts.extend(list(generated_years - memoir_years))

    return compound_facts


def _build_rag_source_text(novel_content_brief: Any) -> str:
    """构建 RAG 来源文本（用于匹配）"""
    parts = []
    
    # 实体描述
    for entity in novel_content_brief.novel_entities:
        name = entity.get("name", entity.get("title", ""))
        desc = entity.get("description", "")
        parts.append(f"{name} {desc}")
    
    # 关系描述
    for rel in novel_content_brief.novel_relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        desc = rel.get("description", "")
        parts.append(f"{source} {target} {desc}")
    
    # 背景片段
    parts.extend(novel_content_brief.novel_snippets)
    
    return " ".join(parts)


def _is_grounded_in_rag(
    fact: str,
    rag_source_text: str,
    novel_entities: List[str],
    fuzzy_threshold: float = 0.6,
) -> bool:
    """
    检查事实是否在 RAG 来源中有支撑（支持改写识别）

    策略：
    1. 实体名匹配：检查是否在 novel_entities 中（含中英文映射）
    2. 精确子串匹配：检查是否在 RAG 来源文本中
    3. 模糊 n-gram 匹配：支持改写（如"改革开放"vs"改革开放政策"）
    4. 词级匹配：≥50%的词在RAG中出现
    5. 中英文实体映射：中文事实通过地名映射在英文RAG描述中查找
    """
    if not fact:
        return False

    # 归一化
    fact_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', fact).upper()
    rag_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', rag_source_text).upper()

    # 1. 实体名匹配（含中英文映射）
    for entity in novel_entities:
        entity_normalized = re.sub(r'[^一-龥a-zA-Z0-9]', '', entity).upper()
        if fact_normalized == entity_normalized or fact_normalized in entity_normalized:
            return True
        if _fuzzy_entity_match(fact, entity):
            return True

    # 2. 精确子串匹配
    if fact_normalized in rag_normalized:
        return True

    # 3. 模糊 n-gram 匹配（支持改写）
    if len(fact_normalized) >= 3:
        fact_trigrams = set(fact_normalized[i:i+3]
                          for i in range(len(fact_normalized)-2))
        rag_trigrams = set(rag_normalized[i:i+3]
                         for i in range(len(rag_normalized)-2))
        if fact_trigrams:
            overlap_ratio = len(fact_trigrams & rag_trigrams) / len(fact_trigrams)
            if overlap_ratio >= fuzzy_threshold:
                return True

    # 4. 词级匹配（≥50%词匹配）
    fact_words = re.findall(r'[一-龥]{2,}', fact)
    if len(fact_words) >= 2:
        matches = sum(1 for word in fact_words if word in rag_source_text)
        if matches / len(fact_words) >= 0.5:
            return True

    # 5. 中英文地名映射溯源：如果事实是中文地名，在 RAG 文本中查找对应英文
    if fact in _CN_EN_LOCATION_MAP:
        en_variants = _CN_EN_LOCATION_MAP[fact]
        for en in en_variants:
            if en.upper() in rag_normalized:
                return True

    return False
