"""
Chunk sweep report formatting and configuration definitions.

Extracted from chunk_sweep.py to keep each module under 300 lines.
"""
import json
import os
from datetime import datetime

# ================================================================
# Candidate configurations
# ================================================================

SWEEP_CONFIGS_FULL = [
    # (label, parent_chunk_size, child_chunk_size, chunk_overlap, strategy)
    ("baseline",         1200, 300, 50,  "recursive"),
    ("finer",            800,  200, 50,  "recursive"),
    ("coarser",          1600, 400, 50,  "recursive"),
    ("more-overlap",     1200, 300, 100, "recursive"),
    ("less-overlap",     1200, 300, 20,  "recursive"),
    ("large-parent",     2000, 500, 50,  "recursive"),
    ("semantic-strategy", 1200, 300, 50,  "semantic"),
]

SWEEP_CONFIGS_FAST = [
    ("baseline",    1200, 300, 50, "recursive"),
    ("finer",       800,  200, 50, "recursive"),
    ("coarser",     1600, 400, 50, "recursive"),
]


# ================================================================
# Report formatting
# ================================================================

def print_report(results, project_root=None):
    """Output a Markdown comparison report of sweep results."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    completed = [r for r in results if r["status"] == "completed"]
    failed = [r for r in results if r["status"] != "completed"]

    print("\n" + "=" * 80)
    print("Chunk 参数扫描报告")
    print("=" * 80)

    if not completed:
        print("\n无成功完成的评估运行。")
        if failed:
            print("\n失败项:")
            for r in failed:
                print(f"  - {r['config']['label']}: {r.get('error', 'unknown')}")
        return

    # Table header
    header = (
        f"{'配置':<22} {'faithfulness':>13} {'answer_rel':>11} "
        f"{'ctx_precision':>13} {'ctx_recall':>11} {'耗时(s)':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    # Sort by faithfulness descending
    completed.sort(
        key=lambda r: r["metrics"].get("faithfulness") or 0,
        reverse=True,
    )

    best = completed[0]
    baseline = next((r for r in completed if r["config"]["label"] == "baseline"), None)

    for r in completed:
        m = r["metrics"]
        label = r["config"]["label"]
        marker = " <-- BEST" if r is best else ""
        if r is baseline and r is not best:
            marker = " (baseline)"

        print(
            f"{label:<22} "
            f"{m.get('faithfulness'):>13.4f} "
            f"{m.get('answer_relevancy'):>11.4f} "
            f"{m.get('context_precision'):>13.4f} "
            f"{m.get('context_recall'):>11.4f} "
            f"{r['elapsed_seconds']:>7.1f}s"
            f"{marker}"
        )

    # Delta vs baseline
    if baseline and baseline is not best:
        print("\n--- 相对 baseline 改善 ---")
        bm = baseline["metrics"]
        for r in completed:
            if r is baseline:
                continue
            m = r["metrics"]
            deltas = []
            for metric in ["faithfulness", "answer_relevancy",
                           "context_precision", "context_recall"]:
                if bm.get(metric) and m.get(metric):
                    delta = m[metric] - bm[metric]
                    sign = "+" if delta >= 0 else ""
                    deltas.append(f"{metric}: {sign}{delta:.4f}")
            if deltas:
                print(f"  {r['config']['label']:<20} {', '.join(deltas)}")

    if failed:
        print(f"\n--- 失败 ({len(failed)}) ---")
        for r in failed:
            print(f"  {r['config']['label']}: {r.get('error', 'unknown')}")

    # Config details
    print("\n--- 配置详情 ---")
    for r in completed:
        c = r["config"]
        print(
            f"  {c['label']:<20} "
            f"parent={c['parent_chunk_size']} "
            f"child={c['child_chunk_size']} "
            f"overlap={c['chunk_overlap']} "
            f"strategy={c['strategy']}"
        )

    print("\n" + "=" * 80)

    # Save JSON report
    report_path = os.path.join(project_root, "logs", "chunk_sweep_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.now().isoformat(),
                "results": results,
            },
            f, ensure_ascii=False, indent=2,
        )
    print(f"JSON 报告已保存: {report_path}")
