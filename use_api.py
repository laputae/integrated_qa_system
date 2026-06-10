# -*- coding: utf-8 -*-
# 导入 requests 模块，用于发送 HTTP 请求（如调用 FastAPI 接口）
import requests
# 导入 json 模块，用于解析接口返回的 JSON 数据
import json
# 导入 uuid 模块，用于生成唯一的会话 ID（session_id）
import uuid
# 配置 FastAPI 服务的接口地址
API_URL = "http://localhost:8000/query"

def stream_query(query: str, source_filter: str = None, session_id: str = None):
    """
    向 FastAPI 问答接口发送流式查询请求，并实时打印返回的答案。
    支持：
      - 自定义问题（query）
      - 学科过滤（source_filter，可选）
      - 会话 ID（session_id，可选，不传则自动生成）
    使用流式读取（stream=True），逐行接收服务器发送的 SSE 数据。
    """

    # 如果未提供会话 ID，则生成一个全局唯一的 UUID
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"生成的会话ID: {session_id}")

    # 构造要发送给 API 的请求体数据
    data = {
        "query": query,                # 用户的问题
        "source_filter": source_filter, # 学科过滤条件（可选）
        "session_id": session_id       # 会话 ID，用于维护对话历史
    }

    # 打印正在发送的查询内容
    print(f"发送查询: {query}")
    # 如果设置了学科过滤，也打印出来
    if source_filter:
        print(f"学科过滤: {source_filter}")
    # 准备打印答案，不换行，实时刷新
    print("答案: ", end="", flush=True)

    try:
        # 使用 requests 发起 POST 请求，发送 JSON 数据，并启用流式读取
        with requests.post(API_URL, json=data, stream=True) as response:
            # 检查 HTTP 状态码是否为 200（成功）
            if response.status_code != 200:
                # 如果失败，打印错误状态码和响应内容
                print(f"\n请求失败: {response.status_code} - {response.text}")
                return  # 提前退出函数

            # 初始化一个变量，用于累积完整答案（用于统计长度等）
            full_answer = ""

            # 逐行读取服务器返回的流式响应（每行是一个 SSE 消息）
            for line in response.iter_lines(decode_unicode=True):
                # 去除首尾空白字符
                line = line.strip()
                # 跳过空行
                if not line:
                    continue
                # 只处理以 "data:" 开头的行（SSE 标准格式）
                if line.startswith("data:"):
                    try:
                        # 提取 "data:" 之后的 JSON 字符串（去掉前 5 个字符）
                        json_str = line[5:].strip()
                        # 如果提取后为空，跳过
                        if not json_str:
                            continue

                        # 将 JSON 字符串解析为 Python 字典
                        data = json.loads(json_str)

                        # 检查返回内容中是否包含错误信息
                        if "error" in data:
                            # 如果有错误，打印错误信息并退出
                            print(f"\n 错误: {data['error']}")
                            return

                        # 从返回数据中获取当前生成的文本片段（token）
                        token = data.get("token", "")
                        # 获取是否是最后一个 token（流是否结束）
                        is_complete = data.get("is_complete", False)

                        # 打印当前 token，不换行
                        # flush=True 表示立即输出到终端，实现“逐字打印”效果
                        print(token, end="", flush=True)
                        # 累积到完整答案中
                        full_answer += token

                        # 如果 is_complete 为 True，表示回答已完成
                        if is_complete:
                            # 打印完成提示和答案总长度
                            print(f"\n 回答完成（总长度: {len(full_answer)} 字符）")
                            # 打印本次会话的 session_id（来自返回数据或本地生成）
                            print(f"会话ID: {data.get('session_id', session_id)}")
                            # 跳出循环，结束处理
                            break

                    # 如果 JSON 解析失败（如格式错误），打印警告并继续
                    except json.JSONDecodeError as e:
                        print(f"\n JSON 解析失败: {line}")
                        continue

    # 如果无法连接到 FastAPI 服务（如服务未启动），捕获连接错误
    except requests.exceptions.ConnectionError:
        print(f"\n 连接失败，请确认 FastAPI 服务已运行在 {API_URL}")

    # 捕获其他所有未预期的异常（如网络超时、代码错误等）
    except Exception as e:
        print(f"\n 发生异常: {e}")

if __name__ == "__main__":
    print("开始测试 FastAPI 问答接口\n")

    # 示例 1: 基础查询
    # stream_query("什么是人工智能？")

    # print("\n" + "-" * 50 + "\n")
    #
    # # 示例 2: 带学科过滤的查询
    stream_query("AI学科的课程内容有什么", source_filter="ai")
    #
    # print("\n" + "-" * 50 + "\n")
    #
    # # 示例 3: 使用固定会话 ID（模拟多轮对话）
    # session = str(uuid.uuid4())
    # print("🔄 模拟多轮对话（同一会话ID）")
    # stream_query("机器学习的基本流程是什么？", session_id=session)
    # # print("\n" + "-" * 30)
    # stream_query("那深度学习呢？", session_id=session, source_filter="ai")