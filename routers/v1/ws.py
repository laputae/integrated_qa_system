"""WebSocket streaming endpoint (/api/stream)."""
import asyncio
import json
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from gateway.auth import decode_access_token
from mysql_qa import get_redis_client
from routers.v1.auth import check_greeting

router = APIRouter()


@router.websocket("/api/stream")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    user_id = 0
    username = "anonymous"
    tenant_id = 0

    if token:
        try:
            payload = decode_access_token(token)
            redis_client = get_redis_client()
            jti = payload.get("jti")
            if not jti or not redis_client.is_token_blacklisted(jti):
                user_id = payload["user_id"]
                username = payload["username"]
                tenant_id = payload.get("tenant_id", 0)
            else:
                await websocket.close(code=4001, reason="Token已失效")
                return
            from gateway.rate_limiter import RateLimiter
            limiter = RateLimiter()
            if not limiter.check_stream_limit(user_id, tenant_id):
                await websocket.close(code=4429, reason="请求过于频繁，请稍后再试")
                return
        except Exception:
            await websocket.close(code=4001, reason="令牌无效或已过期")
            return
    else:
        await websocket.close(code=4001, reason="未提供认证令牌")
        return

    await websocket.accept()

    # Lazy import to avoid circular dependency
    import app as app_module
    qa_system = app_module.qa_system
    _llm_semaphore = app_module._llm_semaphore

    # Per-connection message rate limiter
    ws_msg_count = 0
    ws_window_start = time.time()
    ws_max_messages = qa_system.config.WS_MAX_MESSAGES_PER_CONNECTION
    ws_window_secs = qa_system.config.WS_MESSAGE_WINDOW_SECONDS

    from base import logger

    try:
        while True:
            data = await websocket.receive_text()

            if ws_max_messages > 0:
                now = time.time()
                if now - ws_window_start > ws_window_secs:
                    ws_msg_count = 0
                    ws_window_start = now
                ws_msg_count += 1
                if ws_msg_count > ws_max_messages:
                    await websocket.close(code=4429, reason="单连接消息数超限")
                    break

            request_data = json.loads(data)
            query_text = request_data.get("query")
            source_filter = request_data.get("source_filter")
            session_id = request_data.get("session_id", str(uuid.uuid4()))
            external_context = request_data.get("external_context")
            start_time = time.time()

            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.send_json({"type": "start", "session_id": session_id})

            greeting_response = check_greeting(query_text)
            if greeting_response:
                if websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token", "token": greeting_response,
                        "session_id": session_id,
                    })
                    await websocket.send_json({
                        "type": "end", "session_id": session_id,
                        "is_complete": True,
                        "processing_time": time.time() - start_time,
                    })
                break

            collected_answer = ""
            async for token_val, is_complete in qa_system.aquery(
                query_text, _llm_semaphore,
                user_id=user_id, tenant_id=tenant_id,
                source_filter=source_filter, session_id=session_id,
                external_context=external_context,
            ):
                collected_answer += token_val
                if is_complete and not collected_answer:
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                if token_val and websocket.client_state == websocket.client_state.CONNECTED:
                    await websocket.send_json({
                        "type": "token", "token": token_val,
                        "session_id": session_id,
                    })
                if is_complete:
                    guard_result = getattr(qa_system, '_last_guard_result', None)
                    if guard_result is not None and guard_result.is_hallucinated:
                        if websocket.client_state == websocket.client_state.CONNECTED:
                            await websocket.send_json({
                                "type": "hallucination_warning",
                                "message": "部分回答内容可能缺乏文档依据，建议核实后使用。",
                                "details": guard_result.details,
                                "score": guard_result.score,
                                "session_id": session_id,
                            })
                    if websocket.client_state == websocket.client_state.CONNECTED:
                        await websocket.send_json({
                            "type": "end", "session_id": session_id,
                            "is_complete": True,
                            "processing_time": time.time() - start_time,
                        })
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket.client_state == websocket.client_state.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        try:
            if websocket.client_state == websocket.client_state.CONNECTED:
                await websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {str(e)}")
