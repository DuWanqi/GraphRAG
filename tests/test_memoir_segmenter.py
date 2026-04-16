"""回忆录分段器单测（无 LLM）。"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generation.memoir_segmenter import segment_memoir


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
