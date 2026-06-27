"""
离线 Chunk 参数扫描工具

对多组 chunk 配置分别运行完整 RAGAS 评估管道，输出对比报告。
用于数据驱动地调优 parent_chunk_size / child_chunk_size / chunk_overlap / strategy。

用法:
    uv run python scripts/chunk_sweep.py
    uv run python scripts/chunk_sweep.py --dry-run          # 仅打印配置，不实际运行
    uv run python scripts/chunk_sweep.py --configs fast      # 快速扫描（3 组配置）
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime

# ---- 路径推导 ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "rag_qa"))
sys.path.insert(0, os.path.join(_project_root, "rag_qa", "core"))

from base import logger, Config
from base.chunk_config import ChunkConfigManager


# ================================================================
# 候选配置定义
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


def build_chunk_snapshot(label, parent, child, overlap, strategy):
    """构建 chunk_config_snapshot 字典，写入 eval_runs 表。"""
    return {
        "label": label,
        "parent_chunk_size": parent,
        "child_chunk_size": child,
        "chunk_overlap": overlap,
        "strategy": strategy,
        "swept_at": datetime.now().isoformat(),
    }


def update_chunk_config(parent, child, overlap, strategy):
    """通过 ChunkConfigManager 在运行时更新 chunk 参数。"""
    mgr = ChunkConfigManager()
    mgr.update_config(
        parent_chunk_size=parent,
        child_chunk_size=child,
        chunk_overlap=overlap,
        default_strategy=strategy,
    )
    logger.info(
        "ChunkConfig 已更新: parent=%s child=%s overlap=%s strategy=%s",
        parent, child, overlap, strategy,
    )


def create_temp_collection_name(label):
    suffix = uuid.uuid4().hex[:8]
    return f"edurag_sweep_{label}_{suffix}"


def drop_collection(client, collection_name):
    try:
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            logger.info("临时集合已删除: %s", collection_name)
    except Exception as e:
        logger.warning("删除临时集合失败 (%s): %s", collection_name, e)


def reindex_documents(data_dir, parent, child, overlap):
    """用当前 chunk 参数重新处理文档，返回子块列表。"""
    from llamaindex_processor import process_documents
    chunks = process_documents(data_dir, parent, child, overlap)
    logger.info("重新索引完成: %s 个子块", len(chunks))
    return chunks


def run_single_sweep(config_label, parent, child, overlap, strategy,
                     data_dir, conf, sweep_collection_name):
    """运行单组配置的完整评估流程。"""
    from rag_qa.core.vector_store import VectorStore
    from rag_qa.core.rag_system import RAGSystem
    from repositories.eval_repo import EvalRepository
    from rag_qa.eval.eval_service import EvalService
    from openai import OpenAI

    # 1. 更新 runtime chunk 配置
    update_chunk_config(parent, child, overlap, strategy)

    # 2. 创建临时 VectorStore
    logger.info("[%s] 创建临时集合: %s", config_label, sweep_collection_name)
    vector_store = VectorStore(collection_name=sweep_collection_name)

    # 3. 重新索引文档
    chunks = reindex_documents(data_dir, parent, child, overlap)
    if chunks:
        vector_store.add_documents(chunks)

    # 4. 创建 LLM 客户端 + RAGSystem
    llm_client = OpenAI(
        api_key=conf.DASHSCOPE_API_KEY,
        base_url=conf.DASHSCOPE_BASE_URL,
    )

    def call_dashscope(prompt):
        if llm_client is None:
            yield "错误：LLM服务不可用"
            return
        for attempt in range(conf.LLM_MAX_RETRIES):
            try:
                completion = llm_client.chat.completions.create(
                    model=conf.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个有用的助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=30,
                    stream=True,
                )
                for chunk in completion:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                if attempt < conf.LLM_MAX_RETRIES - 1:
                    import time as _time
                    delay = min(conf.LLM_RETRY_BASE_DELAY * (2 ** attempt),
                                conf.LLM_RETRY_MAX_DELAY)
                    logger.warning("LLM 重试 %s/%s: %s", attempt + 1, conf.LLM_MAX_RETRIES, e)
                    _time.sleep(delay)
                else:
                    logger.error("LLM 调用失败: %s", e)
                    yield f"错误: 调用LLM失败 - {e}"
                    return

    rag_system = RAGSystem(vector_store, call_dashscope)

    # 5. 创建 EvalService 并运行评估
    from db_models.base import SessionLocal
    repo = EvalRepository(SessionLocal)
    eval_service = EvalService(
        config=conf, repo=repo,
        rag_system=rag_system,
        llm_client=llm_client,
        vector_store=vector_store,
    )

    snapshot = build_chunk_snapshot(config_label, parent, child, overlap, strategy)

    result = eval_service.run_evaluation(
        triggered_by="chunk_sweep",
        chunk_config_snapshot=snapshot,
    )

    # 6. 清理
    drop_collection(vector_store.client, sweep_collection_name)

    return {
        "config": {
            "label": config_label,
            "parent_chunk_size": parent,
            "child_chunk_size": child,
            "chunk_overlap": overlap,
            "strategy": strategy,
        },
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "metrics": result.get("metrics", {}),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "error": result.get("error"),
    }


def print_report(results):
    """输出 Markdown 格式的对比报告。"""
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

    # 表头
    header = (
        f"{'配置':<22} {'faithfulness':>13} {'answer_rel':>11} "
        f"{'ctx_precision':>13} {'ctx_recall':>11} {'耗时(s)':>8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    # 按 faithfulness 降序排列
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

    # 与 baseline 的对比
    if baseline and baseline is not best:
        print(f"\n--- 相对 baseline 改善 ---")
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

    # 配置详情
    print(f"\n--- 配置详情 ---")
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

    # 保存 JSON 报告
    report_path = os.path.join(_project_root, "logs", "chunk_sweep_report.json")
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


def main():
    parser = argparse.ArgumentParser(description="离线 Chunk 参数扫描工具")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅打印候选配置，不实际运行",
    )
    parser.add_argument(
        "--configs", choices=["fast", "full"], default="full",
        help="扫描配置集: fast (3组) 或 full (7组)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="数据目录路径（默认使用 config.ini 中 VALID_SOURCES 对应的 data/ 子目录）",
    )
    args = parser.parse_args()

    conf = Config()
    configs = SWEEP_CONFIGS_FAST if args.configs == "fast" else SWEEP_CONFIGS_FULL

    if args.dry_run:
        print("\n候选配置 (dry-run):")
        for label, parent, child, overlap, strategy in configs:
            print(
                f"  {label:<22} "
                f"parent={parent} child={child} overlap={overlap} strategy={strategy}"
            )
        print(f"\n共 {len(configs)} 组配置")
        return

    # 确定数据目录
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(_project_root, "rag_qa", "data", "ai_data")

    if not os.path.isdir(data_dir):
        logger.error("数据目录不存在: %s", data_dir)
        print(f"错误：数据目录不存在: {data_dir}")
        print("请使用 --data-dir 指定有效路径，或确保 rag_qa/data/ai_data 存在文档。")
        sys.exit(1)

    print(f"\n数据目录: {data_dir}")
    print(f"配置集: {args.configs} ({len(configs)} 组)")
    print(f"开始时间: {datetime.now().isoformat()}\n")

    results = []
    for i, (label, parent, child, overlap, strategy) in enumerate(configs):
        collection_name = create_temp_collection_name(label)
        print(f"[{i+1}/{len(configs)}] 运行: {label} "
              f"(parent={parent}, child={child}, overlap={overlap}, strategy={strategy})")

        try:
            result = run_single_sweep(
                label, parent, child, overlap, strategy,
                data_dir, conf, collection_name,
            )
        except Exception as e:
            logger.exception("[%s] 扫描失败", label)
            result = {
                "config": {
                    "label": label,
                    "parent_chunk_size": parent,
                    "child_chunk_size": child,
                    "chunk_overlap": overlap,
                    "strategy": strategy,
                },
                "run_id": None,
                "status": "failed",
                "metrics": {},
                "elapsed_seconds": 0,
                "error": str(e),
            }
            # 尝试清理
            try:
                from pymilvus import MilvusClient
                client = MilvusClient(
                    uri=f"http://{conf.MILVUS_HOST}:{conf.MILVUS_PORT}",
                    db_name=conf.MILVUS_DATABASE_NAME,
                )
                drop_collection(client, collection_name)
            except Exception:
                pass

        results.append(result)
        status_icon = "OK" if result["status"] == "completed" else "FAIL"
        print(f"  -> {status_icon} (run_id={result['run_id']}, "
              f"faithfulness={result['metrics'].get('faithfulness')}, "
              f"{result['elapsed_seconds']:.1f}s)\n")

    print_report(results)


if __name__ == "__main__":
    main()
