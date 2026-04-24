#!/usr/bin/env python3
"""
长文分章 + 评估 + 质量门控重试 pipeline 本地验收脚本。

用法（在 GraphRAG 项目根目录）：
  python scripts/run_long_form_e2e.py \
      --input tests/fixtures/long_memoir_sample.txt \
      --provider openai \
      --length-bucket "400-800" \
      --retrieval-mode keyword \
      --max-gate-retries 2

输出:
  data/long_form_e2e_output.txt     — 生成的完整文本（各章以 --- 分隔）
  data/long_form_e2e_report.json    — 评估报告（含生成内容、质量门控、修复记录）
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
from src.generation import LiteraryGenerator, run_long_form_generation, regenerate_chapters
from src.evaluation import evaluate_long_form, long_form_eval_to_json


def _save_generated_text(lf, out_path: Path) -> None:
    """将生成的各章文本写入文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ch in enumerate(lf.chapters):
        lines.append(f"===== 第 {i + 1} 章 =====\n")
        lines.append(ch.generation.content.strip())
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _enrich_report_with_content(raw_json: dict, lf) -> dict:
    """在报告 JSON 中嵌入各章生成的文本。"""
    for i, seg_data in enumerate(raw_json.get("segments", [])):
        if i < len(lf.chapters):
            seg_data["generated_text"] = lf.chapters[i].generation.content
    raw_json["merged_content"] = lf.merged_content
    return raw_json


async def main():
    parser = argparse.ArgumentParser(description="长文分章 E2E 验收（含质量门控重试）")
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "tests/fixtures/long_memoir_sample.txt",
    )
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--length-bucket", default="400-800")
    parser.add_argument("--retrieval-mode", default="keyword")
    parser.add_argument(
        "--max-gate-retries", type=int, default=2,
        help="质量门控未通过时对失败章节的最大重试次数 (0=不重试)",
    )
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    print(f"[输入] chars={len(text)} path={args.input}")

    llm = create_llm_adapter(provider=args.provider)
    retriever = MemoirRetriever(llm_adapter=llm)
    generator = LiteraryGenerator(llm_adapter=llm)

    t0 = time.perf_counter()

    # ---- 阶段 A: 初次生成 ----
    lf = await run_long_form_generation(
        text, retriever, generator,
        length_bucket=args.length_bucket,
        retrieval_mode=args.retrieval_mode,
        use_llm_parsing=False,
    )
    t_gen = time.perf_counter() - t0
    print(f"[生成] chapters={len(lf.chapters)} elapsed={t_gen:.1f}s")

    # ---- 阶段 B: 评估 + 质量门控（各章并发） ----
    t_eval0 = time.perf_counter()
    ev = await evaluate_long_form(
        lf, llm_adapter=llm,
        use_llm_eval=False,
        enable_fact_check=True,
        max_atomic_facts_per_segment=12,
        fact_check_timeout_per_segment=45.0,
        use_rule_decompose=True,
    )
    t_eval = time.perf_counter() - t_eval0
    print(f"[评估] elapsed={t_eval:.1f}s")
    print(ev.summary_text)

    # ---- 阶段 C: 质量门控重试 ----
    gate_retry_log: list[dict] = []
    retry_round = 0

    while (
        ev.quality_gate is not None
        and not ev.quality_gate.passed
        and ev.quality_gate.remediation is not None
        and retry_round < args.max_gate_retries
    ):
        retry_round += 1
        remed = ev.quality_gate.remediation
        to_regen = remed.chapters_to_regenerate
        print(f"\n[门控重试 {retry_round}/{args.max_gate_retries}] "
              f"未通过，重新生成第 {', '.join(str(c+1) for c in to_regen)} 章")

        for ch_idx in to_regen:
            reasons = remed.reasons.get(ch_idx, [])
            for r in reasons:
                print(f"  第{ch_idx+1}章失败原因: {r}")

        gate_retry_log.append({
            "round": retry_round,
            "chapters_regenerated": to_regen,
            "reasons": {str(k): v for k, v in remed.reasons.items()},
        })

        lf = await regenerate_chapters(
            lf, retriever, generator,
            chapters_to_regenerate=to_regen,
            prompt_adjustments=remed.prompt_adjustments,
            retrieval_mode=args.retrieval_mode,
        )

        ev = await evaluate_long_form(
            lf, llm_adapter=llm,
            use_llm_eval=False,
            enable_fact_check=True,
            max_atomic_facts_per_segment=12,
            fact_check_timeout_per_segment=45.0,
            use_rule_decompose=True,
        )
        print(f"[门控重试 {retry_round}] 重新评估完成")
        print(ev.summary_text)

    if ev.quality_gate and not ev.quality_gate.passed:
        print(f"\n[警告] 经过 {retry_round} 轮重试后仍未通过质量门控")
    elif ev.quality_gate and ev.quality_gate.passed:
        status = "初次通过" if retry_round == 0 else f"第 {retry_round} 轮重试后通过"
        print(f"\n[质量门控] 通过 ({status})")

    # ---- 阶段 D: 保存生成结果 ----
    out_text = ROOT / "data/long_form_e2e_output.txt"
    _save_generated_text(lf, out_text)
    print(f"\n[输出] 生成文本 -> {out_text}")

    # ---- 阶段 E: 保存报告 ----
    report_data = ev.raw_json_ready
    _enrich_report_with_content(report_data, lf)
    if gate_retry_log:
        report_data["gate_retry_log"] = gate_retry_log
    report_data["gate_retry_rounds"] = retry_round

    out_report = ROOT / "data/long_form_e2e_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_s = time.perf_counter() - t0
    print(f"[输出] 评估报告 -> {out_report}")
    print(f"[完成] total={total_s:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
