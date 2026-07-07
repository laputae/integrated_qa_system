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
import os
import sys
import uuid
from datetime import datetime

# ---- 路径推导 ----
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from base import Config, logger  # noqa: E402
from base.chunk_config import ChunkConfigManager  # noqa: E402
from scripts.chunk_sweep_report import SWEEP_CONFIGS_FAST, SWEEP_CONFIGS_FULL, print_report  # noqa: E402


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
    from rag_qa.core.llamaindex_processor import process_documents
    chunks = process_documents(data_dir, parent, child, overlap)
    logger.info("重新索引完成: %s 个子块", len(chunks))
    return chunks


def run_single_sweep(config_label, parent, child, overlap, strategy,
                     data_dir, conf, sweep_collection_name):
    """运行单组配置的完整评估流程。"""
    from openai import OpenAI

    from rag_qa.core.rag_system import RAGSystem
    from rag_qa.core.vector_store import VectorStore
    from rag_qa.eval.eval_service import EvalService
    from repositories.eval_repo import EvalRepository

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

    print_report(results, project_root=_project_root)


if __name__ == "__main__":
    main()
