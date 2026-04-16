"""回忆录分段器单测（无 LLM）：覆盖时间边界、元数据、校验报告。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.memoir_segmenter import (
    segment_memoir,
    extract_years,
    validate_segmentation,
    SegmentMeta,
)


# ---------------------------------------------------------------------------
# 原有测试（保持兼容）
# ---------------------------------------------------------------------------

def test_empty():
    assert segment_memoir("") == []
    assert segment_memoir("   ") == []


def test_short_single_segment():
    t = "这是一段很短的回忆。" * 5
    segs = segment_memoir(t, target_min_chars=300, target_max_chars=800)
    assert len(segs) == 1
    assert segs[0].index == 0


def test_paragraph_split():
    a = "第一章\n" + ("那是1988年的夏天。" * 40)
    b = "第二章\n" + ("后来我来到深圳。" * 40)
    text = a + "\n\n" + b
    segs = segment_memoir(text, target_min_chars=200, target_max_chars=600)
    assert len(segs) >= 1
    assert all(s.text.strip() for s in segs)


def test_indices_sequential():
    text = "\n\n".join([f"段落{i}。" + "内容" * 120 for i in range(5)])
    segs = segment_memoir(text, target_min_chars=150, target_max_chars=400)
    for j, s in enumerate(segs):
        assert s.index == j


# ---------------------------------------------------------------------------
# 新增：时间边界感知
# ---------------------------------------------------------------------------

def test_temporal_boundary_split():
    """以不同年份开头的段落不应被合并。"""
    p1 = "一九七二年秋天，我来到陕北插队。" + "黄土高坡" * 30
    p2 = "一九七七年十月，广播里传来恢复高考的消息。" + "复习准备" * 30
    text = p1 + "\n\n" + p2
    segs = segment_memoir(text, target_min_chars=200, target_max_chars=2000)
    # 两段虽然各自可能比 target_min 短（取决于乘数），但因为有时间边界，不应合并
    assert len(segs) >= 2
    # 第一段包含 1972，第二段包含 1977
    years_0 = segs[0].meta.detected_years if segs[0].meta else ()
    years_1 = segs[1].meta.detected_years if segs[1].meta else ()
    assert "1972" in years_0
    assert "1977" in years_1


def test_year_extraction_chinese():
    """中文大写年份正确转为阿拉伯数字。"""
    years = extract_years("一九八八年夏天我到了深圳，后来二零零八年又回去了")
    assert "1988" in years
    assert "2008" in years


def test_year_extraction_arabic():
    """阿拉伯数字年份正确提取。"""
    years = extract_years("1992年春天邓小平南巡，到了2008年奥运会")
    assert "1992" in years
    assert "2008" in years


# ---------------------------------------------------------------------------
# 新增：元数据
# ---------------------------------------------------------------------------

def test_segment_meta_populated():
    """分段后每个 segment 都应有 meta。"""
    text = "一九七二年秋天，我来到陕北。" * 50 + "\n\n" + "一九八八年夏天，我到了深圳。" * 50
    segs = segment_memoir(text, target_min_chars=100, target_max_chars=600)
    for s in segs:
        assert s.meta is not None
        assert isinstance(s.meta, SegmentMeta)
        assert s.meta.split_reason != ""


def test_segment_meta_locations():
    """元数据应能提取地名。"""
    text = "一九七二年我来到陕北张家塬。" * 60
    segs = segment_memoir(text, target_min_chars=100, target_max_chars=800)
    all_locations = []
    for s in segs:
        if s.meta:
            all_locations.extend(s.meta.detected_locations)
    assert "陕北" in all_locations


# ---------------------------------------------------------------------------
# 新增：校验报告
# ---------------------------------------------------------------------------

def test_validate_segmentation_pass():
    """正常分段应通过校验。"""
    text = "\n\n".join(["段落内容。" * 50 for _ in range(3)])
    segs = segment_memoir(text, target_min_chars=100, target_max_chars=600)
    report = validate_segmentation(segs, target_min_chars=100, target_max_chars=600)
    assert report.passed
    assert report.segment_count == len(segs)
    assert report.total_chars > 0


def test_validate_segmentation_detects_issues():
    """过长段应被报告为问题。"""
    # 构造一个超长段
    from src.generation.memoir_segmenter import MemoirSegment
    fake = [MemoirSegment(0, "x" * 5000)]
    report = validate_segmentation(fake, target_min_chars=300, target_max_chars=800)
    assert any(i.severity == "error" for i in report.issues)
    assert not report.passed


# ---------------------------------------------------------------------------
# 新增：真实样本
# ---------------------------------------------------------------------------

def test_real_sample_segmentation():
    """对真实测试样本进行分段，验证段数和元数据合理性。"""
    sample = project_root / "tests" / "fixtures" / "long_memoir_sample.txt"
    if not sample.exists():
        return  # skip if fixture not found
    text = sample.read_text("utf-8")
    segs = segment_memoir(text)
    # 样本有 6 个自然段，每段以不同年份开头
    assert len(segs) >= 4  # 至少分成 4 段
    assert len(segs) <= 10  # 不应过度切分
    # 每段都有 meta
    for s in segs:
        assert s.meta is not None
    # 至少有一些段有时间标签
    labeled = [s for s in segs if s.meta and s.meta.temporal_label]
    assert len(labeled) >= 3
