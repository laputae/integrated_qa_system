# -*- coding: utf-8 -*-
# 导入 FastAPI 核心类：用于创建 API 应用、处理请求、返回错误
from fastapi import FastAPI, HTTPException, Request
# 导入 StreamingResponse，用于支持流式响应（逐字返回答案）
from fastapi.responses import StreamingResponse
# 导入 json 模块，用于将字典转换为 JSON 字符串（如返回给前端的数据）
import json
# 导入 uuid 模块，用于生成唯一的会话 ID（session_id）
import uuid
from new_main import IntegratedQASystem

# 创建一个 FastAPI 应用实例
app = FastAPI(title="集成问答系统 API", description="基于 RAG + MySQL + Redis 的问答系统 FastAPI 接口")

# 全局初始化一个问答系统实例
qa_system = IntegratedQASystem()

# 使用 @app.post 装饰器，将下面的函数注册为 POST 请求接口，路径为 /query
@app.post("/query")
async def handle_query(request: Request):
    """
    接收客户端发送的 JSON 请求，支持流式返回答案。
    请求体示例：
    {
        "query": "什么是人工智能？",
        "source_filter": "ai",  // 可选，用于学科过滤
        "session_id": "a1b2c3d4-..."   // 可选，用于维护对话历史
    }
    响应为 SSE（Server-Sent Events）流式格式，前端可实时接收每个 token。
    """

    # 尝试解析请求体中的 JSON 数据
    try:
        body = await request.json()
    # 如果 JSON 格式不合法（如缺少引号、语法错误），抛出 400 错误
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 数据")

    # 从 JSON 中获取用户问题，去除首尾空格，若无则为空字符串
    query = body.get("query", "").strip()
    # 获取学科过滤条件（可选），若未提供则为 None
    source_filter = body.get("source_filter", None)
    # 获取会话 ID（可选），用于维护多轮对话历史
    session_id = body.get("session_id", None)

    # 如果用户没有输入问题，返回 400 错误
    if not query:
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    # 如果客户端未提供 session_id，则自动生成一个全局唯一 ID
    if not session_id:
        session_id = str(uuid.uuid4())

    # 从配置中获取支持的学科类别列表（如 ['ai', 'java']）
    valid_sources = qa_system.config.VALID_SOURCES
    # 如果提供了 source_filter 但不在合法范围内，返回 400 错误
    if source_filter and source_filter not in valid_sources:
        raise HTTPException(
            status_code=400,
            detail=f"无效的学科类别。支持: {valid_sources}"
        )

    # 定义一个生成器函数，用于流式返回答案（逐 token 输出）
    def generate_response():
        try:
            # 调用问答系统的核心 query 方法，返回生成器（每次产出一个 token）
            for token, is_complete in qa_system.query(
                query=query,
                source_filter=source_filter,
                session_id=session_id
            ):
                # 构造要返回的 JSON 消息，包含当前文本片段和状态
                message = {
                    "token": token,           # 当前生成的文本（如一个字）
                    "is_complete": is_complete,     # 是否是最后一个 token
                    "session_id": session_id        # 返回会话 ID，便于前端维护
                }
                # 使用 SSE 格式：data: {json}\n\n
                # ensure_ascii=False 确保中文不被转义为 \uXXXX
                yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"

        # 捕获问答系统内部任何异常（如数据库错误、LLM 调用失败）
        except Exception as e:
            # 记录错误日志
            error_msg = f"处理查询时发生错误: {str(e)}"
            qa_system.logger.error(error_msg)
            # 构造错误消息，标记流结束
            message = {
                "error": error_msg,          # 错误信息
                "is_complete": True          # 表示流已结束
            }
            # 同样以 SSE 格式返回错误
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"

    # 返回流式响应，媒体类型为 text/event-stream（SSE 标准）
    return StreamingResponse(
        generate_response(),           # 传入生成器函数
        media_type="text/event-stream" # 告诉浏览器这是流式数据
    )

# 当直接运行此脚本时（python main.py），启动 Uvicorn 服务器
if __name__ == '__main__':
    # 导入 Uvicorn（用于运行 FastAPI）
    import uvicorn
    # 启动服务器，监听所有 IP 地址的 8000端口
    # 可通过 http://localhost:8003 访问 API
    uvicorn.run(app, host="0.0.0.0", port=8000)
