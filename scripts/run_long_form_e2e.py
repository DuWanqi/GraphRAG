#!/usr/bin/env python3
"""
长文分章 + 评估 pipeline本地验收脚本。

用法（在 GraphRAG 项目根目录）：
  python scripts/run_long_form_e2e.py

依赖：已配置 LLM 与 GraphRAG 索引；否则检索/生成可能失败。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm import create_llm_adapter
from src.retrieval import MemoirRetriever
from src.generation import LiteraryGenerator, run_long_form_generation
from src.evaluation import evaluate_long_form, long_form_eval_to_json


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "tests/fixtures/long_memoir_sample.txt",
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--length-bucket", default="400-800")
    parser.add_argument("--retrieval-mode", default="keyword")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    print(f"chars={len(text)} path={args.input}")

    llm = create_llm_adapter(provider=args.provider)
    retriever = MemoirRetriever(llm_adapter=llm)
    generator = LiteraryGenerator(llm_adapter=llm)

    t0 = time.perf_counter()
    lf = await run_long_form_generation(
        text,
        retriever,
        generator,
        length_bucket=args.length_bucket,
        retrieval_mode=args.retrieval_mode,
        use_llm_parsing=False,
    )
    print(f"chapters={len(lf.chapters)} generate_s={time.perf_counter()-t0:.2f}")

    ev = await evaluate_long_form(
        lf,
        llm_adapter=llm,
        use_llm_eval=False,
        enable_fact_check=True,
        max_atomic_facts_per_segment=12,
        fact_check_timeout_per_segment=45.0,
        use_rule_decompose=True,
    )
    print(ev.summary_text)
    out = ROOT / "data/long_form_e2e_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(long_form_eval_to_json(ev), encoding="utf-8")
    print(f"wrote {out} total_s={time.perf_counter()-t0:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
