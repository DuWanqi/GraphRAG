"""Convert memoir PDFs to text using PyMuPDF (workaround for Claude Code Read tool bug)."""
import fitz
import sys
from pathlib import Path

SOURCES = Path(r"D:\BaiduSyncdisk\CUHKSZ\Y4T2\DDA4080\GraphRAG\data\memoirs\sources")
OUT_DIR = Path(r"D:\BaiduSyncdisk\CUHKSZ\Y4T2\DDA4080\GraphRAG\data\memoirs\sources_text")
OUT_DIR.mkdir(exist_ok=True)

PDFS = [
    "01_wusheng_qunluo.pdf",
    "02_hongweibing_xiaobao.pdf",
    "03_heiwulei_yijiu.pdf",
    "04_gaige_licheng.pdf",
    "05_cangsang_suiyue.pdf",
    "06_songrenqiong.pdf",
]

for name in PDFS:
    pdf_path = SOURCES / name
    out_path = OUT_DIR / (pdf_path.stem + ".txt")
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"SKIP {name} (already converted, {out_path.stat().st_size} bytes)")
        continue
    try:
        doc = fitz.open(str(pdf_path))
        with out_path.open("w", encoding="utf-8") as f:
            for i, page in enumerate(doc):
                f.write(f"\n===== PAGE {i+1} =====\n")
                f.write(page.get_text())
        print(f"OK   {name}: {len(doc)} pages -> {out_path.name} ({out_path.stat().st_size} bytes)")
        doc.close()
    except Exception as e:
        print(f"FAIL {name}: {e}")
