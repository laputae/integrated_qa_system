"""MCP Server -- 将RAG能力以MCP Tool 形式对外暴露。"""

import uuid
import json
from typing import Literal

from fastmcp import FastMCP


mcp_server = FastMCP("rag-mcp")


def _get_qa_system():
    import app as app_module

    return app_module.qa_system


def _get_semaphore():
    import app as app_module

    return app_module._llm_semaphore


@mcp_server.tool()
async def rag_query(
    query: str,
    source_filter: Literal["ai_data", "jpkb"] | None = None,
    external_context: str | None = None,
) -> str:
    """企业级智能问答：基于 BM25 + RAG 双级检索，支持多数据源过滤。

    Args:
        query: 用户输入的具体问题文本。
        source_filter: 数据源过滤选项。支持的值为 'ai_data' 或 'jpkb'（可选）。
        external_context: 额外补充的上下文信息，将直接拼接至检索提示词中（可选）。

    Returns:
        str: 检索并生成的回答文本。若未查到答案返回“未找到相关答案”；若判定存在幻觉，尾部附加【幻觉提示】后缀。
    """
    qa_system = _get_qa_system()
    semaphore = _get_semaphore()
    session_id = str(uuid.uuid4())

    collected = ""
    async for token, is_complete in qa_system.aquery(
        query,
        semaphore,
        user_id=0,
        tenant_id=0,
        source_filter=source_filter,
        session_id=session_id,
        external_context=external_context,
    ):
        collected += token
        if is_complete:
            break

    if not collected:
        collected = "未找到相关答案"

    guard = getattr(qa_system, "_last_guard_result", None)
    if guard and guard.is_hallucinated:
        collected += "\n\n【幻觉提示】部分内容可能缺乏文档依据，建议核实后使用。"

    return collected


@mcp_server.tool()
async def check_system_health() -> str:
    """检查问答系统各组件健康状态（MySQL/Redis/Milvus/LLM等）。

    Returns:
        str: 格式化的 JSON 字符串，包含系统及核心组件的状态。结构如下：
            {
              "status": "healthy" | "unhealthy",
              "components": {
                "mysql": true | false,
                "redis": true | false,
                "milvus": true | false,
                "llm": true | false
              }
            }
    """
    qa_system = _get_qa_system()
    status = qa_system.health.get_status_response()
    return json.dumps(status, ensure_ascii=False, indent=2)


mcp_app = mcp_server.http_app(transport="streamable-http", path="/")
