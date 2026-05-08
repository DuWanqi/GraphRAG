#!/usr/bin/env python3
"""
新旧功能对比演示脚本（无需 LLM / Ollama / 索引）。

用法：
  python scripts/demo_comparison.py

演示三组对比：
  对比一  分段策略   旧：仅按空行切分，无元数据 → 新：时间边界优先 + 元数据 + 校验报告
  对比二  跨章上下文 旧：每章 prompt 无前文信息 → 新：注入前文概要 + 反重复要点 + 位置指令
  对比三  评估与门控 旧：仅段级指标，无门控   → 新：跨章指标 + 质量门控 + 修复计划
"""

from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ═══════════════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════════════

CYAN   = "\033[36m"
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def title(text: str):
    w = 72
    print()
    print(f"{BOLD}{'═' * w}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'═' * w}{RESET}")

def subtitle(text: str):
    print(f"\n{YELLOW}{BOLD}{'─' * 60}{RESET}")
    print(f"{YELLOW}{BOLD}  {text}{RESET}")
    print(f"{YELLOW}{BOLD}{'─' * 60}{RESET}")

def label(tag: str, color: str = DIM):
    print(f"\n{color}{BOLD}[{tag}]{RESET}")


# ═══════════════════════════════════════════════════════════════════════
# 加载真实样本
# ═══════════════════════════════════════════════════════════════════════

sample_path = ROOT / "tests" / "fixtures" / "long_memoir_sample.txt"
TEXT = sample_path.read_text("utf-8")


# ═══════════════════════════════════════════════════════════════════════
# 对比一：分段策略
# ═══════════════════════════════════════════════════════════════════════

title("对比一：分段策略")

# --- 旧行为模拟：仅按空行切分，无元数据，短块会合并 ---
subtitle("旧行为：仅按空行切分，无元数据")

def old_segment(text: str, target_min: int = 300):
    """模拟旧版分段：空行切分 + 无条件合并短块"""
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text.strip()) if b.strip()]
    # 合并短块（旧版不考虑时间边界）
    merged = []
    buf = blocks[0] if blocks else ""
    for b in blocks[1:]:
        if len(buf) < target_min:
            buf = f"{buf}\n\n{b}"
        else:
            merged.append(buf)
            buf = b
    if buf:
        merged.append(buf)
    return merged

old_segs = old_segment(TEXT)
print(f"  分段数: {len(old_segs)}")
for i, seg in enumerate(old_segs):
    years = re.findall(r"(?:一九[^\s]{2}|二零[^\s]{2}|(?:19|20)\d{2})\s*年", seg[:100])
    preview = seg[:50].replace("\n", " ")
    print(f"  段{i}: {len(seg)}字  开头年份={years[0] if years else '?'}  [{preview}…]")
    # 没有元数据、没有切分原因、没有校验
print(f"\n  {DIM}→ 无元数据、无切分原因、无校验报告{RESET}")

# --- 新行为 ---
subtitle("新行为：时间边界优先 + 元数据 + 校验报告")

from src.generation import segment_memoir, validate_segmentation, allocate_segment_budgets

new_segs = segment_memoir(TEXT)
budgets = allocate_segment_budgets(new_segs, "400-800")
report = validate_segmentation(new_segs)

print(f"  分段数: {len(new_segs)}")
for s, b in zip(new_segs, budgets):
    meta = s.meta
    preview = s.text[:50].replace("\n", " ")
    print(f"  段{s.index}: {len(s.text)}字 → 预算 {b.length_hint}")
    print(f"    {GREEN}时间={meta.temporal_label}  地点={','.join(meta.detected_locations) or '—'}"
          f"  切分原因={meta.split_reason}{RESET}")
    print(f"    {DIM}[{preview}…]{RESET}")

print(f"\n  {CYAN}校验报告:{RESET}")
for line in report.to_text().split("\n"):
    print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════════
# 对比二：跨章上下文
# ═══════════════════════════════════════════════════════════════════════

title("对比二：跨章 Prompt 注入")

# 模拟已生成 3 章内容
mock_chapters = [
    "1972年的陕北黄土高原，正是知识青年上山下乡运动的高峰期。那一年全国有超过两百万城市青年告别家乡，奔赴农村。在延安地区，像张家塬这样的生产队接收了大量来自西安、兰州等城市的知青。",
    "1977年10月21日，《人民日报》头版刊登了恢复高考的消息。这一决策改变了整整一代人的命运。当年冬天，全国五百七十万考生走进了阔别十一年的考场，最终录取了二十七万人。",
    "1988年的深圳，正处于经济特区建设的第八个年头。华强北已经从一片荒地发展成为初具规模的电子产品集散地。那一年深圳的GDP首次突破百亿元大关。",
]

subtitle("旧行为：第 4 章 prompt 中只有当前段内容")

print(f"""  {DIM}请根据以下信息，为回忆录片段生成1-3段历史背景描述。

  ## 回忆录原文
  一九九二年春天，邓小平南巡的消息像一阵风一样吹遍了深圳……

  ## 时间背景
  年份：1992  地点：深圳

  ## 相关历史信息
  （检索结果……）

  要求：
  1. 生成的内容应能自然衔接在回忆录原文之后
  2. 使用文学性的语言
  3. 字数控制在 85-115字{RESET}

  {RED}→ LLM 不知道前 3 章已经写了什么
  → 可能重复描写"改革开放""经济特区"等已出现内容
  → 无开篇/中段/收尾的角色区分{RESET}""")

# --- 新行为：用 ChapterContext 模拟 ---
subtitle("新行为：第 4 章 prompt 注入跨章上下文")

from src.generation.chapter_context import ChapterContext

ctx = ChapterContext(total_chapters=6)
for i, content in enumerate(mock_chapters):
    ctx.record_chapter(i, content, entities=["深圳", "延安", "华强北"][:i+1])

prompt_section = ctx.build_prompt_section(current_index=3)

print(f"  {DIM}请根据以下信息，为回忆录片段生成1-3段历史背景描述。{RESET}")
print(f"  {DIM}…（回忆录原文 + 时间背景 + 检索结果 同上）…{RESET}")
print()
print(f"  {GREEN}{BOLD}▼ 以下为新增的跨章上下文段落 ▼{RESET}")
for line in prompt_section.split("\n"):
    print(f"  {GREEN}  {line}{RESET}")
print()
print(f"  {DIM}写作要求：{RESET}")
print(f"  {DIM}…（同上）…{RESET}")
print(f"  {GREEN}5. 禁止使用「总之」「综上」等总结性语句{RESET}")
print(f"  {GREEN}6. 禁止使用「在这个大背景下」等空泛套话{RESET}")
print(f"  {GREEN}7. 只做叙事性描写，不要概括或评论{RESET}")

print(f"""
  {CYAN}→ LLM 清楚知道前 3 章分别写了什么（含年代标签 + 内容概要）
  → 高频要点被列出，明确要求不再展开
  → 当前章被告知"位于中段，与前文自然衔接"{RESET}""")


# ═══════════════════════════════════════════════════════════════════════
# 对比三：评估与质量门控
# ═══════════════════════════════════════════════════════════════════════

title("对比三：评估与质量门控")

# 构造模拟生成结果（含一个有问题的章节）
mock_generated = [
    "1972年的陕北黄土高原，正是知识青年上山下乡运动的高峰期。那一年全国有超过两百万城市青年告别家乡。",
    "1977年恢复高考的消息如春雷炸响。当年冬天，五百七十万考生走进了阔别十一年的考场。",
    "1988年的深圳经济特区已初具规模。华强北从荒地变成了电子集散地。",
    # 第4章故意制造问题：总结句多 + 与第3章部分重复
    "总之，1992年南巡讲话意义重大。综上所述，深圳经济特区已初具规模。总的来说改革开放成效显著。在这个大背景下，华强北从荒地变成了电子集散地。",
    "1998年亚洲金融风暴席卷东南亚。泰铢崩盘后多国货币连锁暴跌。",
    "2008年北京奥运会向世界展示了中国的发展成就。那年秋天阳光灿烂。",
]

# --- 旧行为 ---
subtitle("旧行为：仅段级指标，无跨章检测")

from src.evaluation.metrics import calculate_all_metrics, aggregate_scores

print(f"  {DIM}段级指标（以第 4 章为例）:{RESET}")
m = calculate_all_metrics(
    memoir_text="一九九二年春天邓小平南巡……" * 10,
    generated_text=mock_generated[3],
    reference_entities=["深圳", "邓小平"],
    reference_year="1992",
    keywords=["南巡", "改革"],
)
score = aggregate_scores(m)
print(f"    综合分: {score:.2f}")
for name, mr in m.items():
    print(f"    {name}: {mr.value:.2f} ({mr.explanation})")
print(f"""
  {RED}→ 没有跨章重复度指标 — 第3章和第4章的重复内容无法发现
  → 没有总结句/套话检测 — "总之""综上所述"被视为正常内容
  → 没有门控判定 — 不知道这章"能不能用"
  → 没有修复建议 — 不知道该怎么改{RESET}""")

# --- 新行为 ---
subtitle("新行为：跨章指标 + 质量门控 + 修复计划")

from src.evaluation.metrics import CrossChapterMetrics
from src.evaluation.quality_gate import check_quality_gate, QualityThresholds

print(f"  {CYAN}跨章指标（全文 6 章）:{RESET}")
rep = CrossChapterMetrics.inter_chapter_repetition(mock_generated)
sty = CrossChapterMetrics.style_consistency(mock_generated)
summ = CrossChapterMetrics.summary_sentence_ratio(mock_generated)
for mr in [rep, sty, summ]:
    color = GREEN if mr.value >= 0.7 else RED
    print(f"    {color}{mr.name}: {mr.value:.2f}  ({mr.explanation}){RESET}")

print(f"\n  {CYAN}质量门控:{RESET}")
gate = check_quality_gate(
    mock_generated,
    segment_scores=[7.5, 7.0, 6.5, 3.8, 7.2, 6.8],   # 第4章故意低分
    fact_scores=[0.9, 0.85, 0.8, 0.45, 0.88, 0.82],     # 第4章 FActScore 低
    target_chars_per_chapter=[80, 70, 60, 80, 70, 60],
    thresholds=QualityThresholds(min_segment_score=5.0, min_fact_score=0.6),
)
print(f"    {BOLD}整体判定: {GREEN + '通过' if gate.passed else RED + '未通过'}{RESET}")
print(f"    综合分: {gate.overall_score:.2f}")
print()
for cr in gate.chapter_results:
    status = f"{GREEN}✓{RESET}" if cr.passed else f"{RED}✗{RESET}"
    print(f"    第{cr.chapter_index + 1}章 {status}")
    for iss in cr.issues:
        color = RED if iss.severity == "error" else YELLOW
        print(f"      {color}[{iss.severity}] {iss.dimension}: {iss.message}{RESET}")
        print(f"        {CYAN}→ 建议: {iss.suggestion}{RESET}")

if gate.cross_chapter_issues:
    print(f"\n    {BOLD}跨章问题:{RESET}")
    for ci in gate.cross_chapter_issues:
        print(f"      {RED}[{ci.severity}] {ci.dimension}: {ci.message}{RESET}")
        print(f"        {CYAN}→ 建议: {ci.suggestion}{RESET}")

if gate.remediation:
    print(f"\n    {BOLD}修复计划:{RESET}")
    ids = gate.remediation.chapters_to_regenerate
    print(f"      {RED}需重新生成: 第 {', '.join(str(c+1) for c in ids)} 章{RESET}")
    for ch_idx, reasons in gate.remediation.reasons.items():
        print(f"      第{ch_idx+1}章原因:")
        for r in reasons:
            print(f"        - {r}")


# ═══════════════════════════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════════════════════════

title("对比总结")

rows = [
    ("分段策略", "仅空行切分，无元数据", "时间边界优先 + SegmentMeta + 校验报告"),
    ("跨章衔接", "每章独立，prompt 无前文信息", "ChapterContext 注入概要+反重复+位置指令"),
    ("风格控制", "无约束", "prompt 禁止总结句/套话，强制叙事体"),
    ("评估指标", "8 项段级指标", "8 项段级 + 3 项跨章 (重复/风格/总结句)"),
    ("事实检查", "仅布尔 is_factual", "FActScore 支持率 (0-1)"),
    ("质量判定", "无", "QualityGate 5 维阈值检查 → 通过/不通过"),
    ("修复能力", "无", "RemediationPlan → 哪章/为什么/怎么改"),
]

col_w = [14, 38, 44]
header = f"  {'维度':<{col_w[0]}}{'旧版':<{col_w[1]}}{'新版':<{col_w[2]}}"
print(f"\n{BOLD}{header}{RESET}")
print(f"  {'─' * sum(col_w)}")
for dim, old, new in rows:
    print(f"  {dim:<{col_w[0]}}{RED}{old:<{col_w[1]}}{RESET}{GREEN}{new}{RESET}")

print()
