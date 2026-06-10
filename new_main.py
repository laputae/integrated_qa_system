# -*- coidng:utf-8 -*-
# 导入 MySQL 和 Redis 客户端，管理数据库和缓存
from mysql_qa import MySQLClient, RedisClient, BM25Search
# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa import VectorStore, RAGSystem
# 导入配置和日志工具，用于系统配置和日志记录
from base import logger, Config
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 导入时间库，用于记录处理时间
import time
# 导入 UUID 库，生成唯一会话 ID
import uuid
# 导入 pymysql 错误处理，用于数据库操作的异常捕获
import pymysql
class IntegratedQASystem:
    def __init__(self):
        # 初始化日志工具，用于记录系统运行信息
        self.logger = logger
        # 初始化配置对象，加载系统参数
        self.config = Config()
        # 初始化 MySQL 客户端，用于数据库操作
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端，用于缓存管理
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索模块，结合 MySQL 和 Redis
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)
        try:
            # 初始化 OpenAI 客户端，连接 DashScope API
            self.client = OpenAI(api_key=self.config.DASHSCOPE_API_KEY,
                                 base_url=self.config.DASHSCOPE_BASE_URL)
        except Exception as e:
            # 记录 OpenAI 初始化失败的错误日志
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            # 抛出异常，终止初始化
            raise
        # 初始化向量存储，用于 RAG 系统的知识库管理
        self.vector_store = VectorStore()
        # 初始化 RAG 系统，传入向量存储和 DashScope API 调用函数
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope)
        # 初始化对话历史表，用于存储会话记录
        self.init_conversation_table()

    def init_conversation_table(self):
        """初始化MySQL中的conversations表，用于存储对话历史"""
        try:
            # 创建 conversations 表，包含会话 ID、问题、答案和时间戳
            self.mysql_client.cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(36) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    INDEX idx_session_id (session_id)
                )
            """)
            # 提交数据库事务
            self.mysql_client.connection.commit()
            # 记录表初始化成功的日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 记录表初始化失败的错误日志
            self.logger.error(f"初始化对话历史表失败: {e}")
            # 抛出异常，终止初始化
            raise

    def call_dashscope(self, prompt):
        """调用DashScope API生成答案（流式输出）"""
        try:
            # 创建聊天完成请求，启用流式输出
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,  # 使用配置中的语言模型
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},  # 系统提示
                    {"role": "user", "content": prompt},  # 用户输入的提示
                ],
                timeout=30,  # 设置 30 秒超时
                stream=True  # 启用流式输出
            )
            # 遍历流式输出的每个 chunk
            for chunk in completion:
                # print(f'chunk--》{chunk}')
                # print("*"*80)
                if chunk.choices and chunk.choices[0].delta.content:
            #         # 获取当前 chunk 的内容
                    content = chunk.choices[0].delta.content
                    yield content
        except Exception as e:
            # 记录 API 调用失败的错误日志
            self.logger.error(f"LLM调用失败: {e}")
            # 返回错误信息
            return f"错误：LLM调用失败 - {e}"

    def _fetch_recent_history(self, session_id):
        """获取最近5轮对话历史"""
        try:
            # 执行 SQL 查询，获取最近 5 轮对话
            self.mysql_client.cursor.execute("""
                      SELECT question, answer
                      FROM conversations
                      WHERE session_id = %s
                      ORDER BY timestamp DESC
                      LIMIT %s
                  """, (session_id, 5))
            # print(f'self.mysql_client.cursor.fetchall()---》{self.mysql_client.cursor.fetchall()}')
            # 将查询结果转换为字典列表
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 反转结果，按时间正序返回
            return history[::-1]

        except pymysql.MySQLError as e:
            # 记录查询失败的错误日志
            self.logger.error(f"获取对话历史失败: {e}")
            # 返回空列表
            return []

    def get_session_history(self, session_id ):
        """从MySQL获取会话历史"""
        # 调用 _fetch_recent_history 获取对话历史
        return self._fetch_recent_history(session_id)

    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        """更新会话历史到MySQL，保留最近5轮对话"""
        try:
            # 插入新的对话记录
            self.mysql_client.cursor.execute("""
                INSERT INTO conversations (session_id, question, answer, timestamp)
                VALUES (%s, %s, %s, NOW())
            """, (session_id, question, answer))
            # 获取更新后的对话历史
            history = self._fetch_recent_history(session_id)
            # 删除超出 5 轮的旧记录
            self.mysql_client.cursor.execute("""
                DELETE FROM conversations
                WHERE session_id = %s AND id NOT IN (
                    SELECT id FROM (
                        SELECT id
                        FROM conversations
                        WHERE session_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ) AS sub
                )
            """, (session_id, session_id, 5))
            # 提交事务
            self.mysql_client.connection.commit()
            # 记录更新成功的日志
            self.logger.info(f"会话 {session_id} 历史更新成功")
            # 返回更新后的历史
            return history
        except pymysql.MySQLError as e:
            # 记录数据库操作失败的错误日志
            self.logger.error(f"更新会话历史失败: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise
        except Exception as e:
            # 记录意外错误的日志
            self.logger.error(f"更新会话历史意外错误: {e}")
            # 回滚事务
            self.mysql_client.connection.rollback()
            # 抛出异常
            raise

    def query(self, query, source_filter=None, session_id=None):
        # print(f'你好')
        """查询集成系统，支持对话历史和流式输出"""
        start_time = time.time()  # 记录查询开始时间
        # 记录查询信息到日志
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id})")
        # 获取对话历史，若无 session_id 则返回空列表
        history = self.get_session_history(session_id) if session_id else []
        # print(f'history--->{history}')
        # 执行 BM25 搜索，获取答案和是否需要 RAG 的标志
        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        # print(f'answer-——》{answer}')
        # print(f'need_rag-——》{need_rag}')
        if answer:
            # 如果找到可靠答案，记录答案到日志
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                # 更新对话历史
                self.update_session_history(session_id, query, answer)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回答案，标记为完整
            yield answer, True
        elif need_rag:
            self.logger.info("无可靠MySQL答案，回退到RAG")
            # 初始化收集完整答案的字符串
            collected_answer = ""
            # 从 RAG 系统获取流式输出
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                # 累积答案
                collected_answer += token
                # 逐 token 返回，标记为部分答案
                yield token, False
            if session_id:
                # 更新对话历史，存储完整答案
                self.update_session_history(session_id, query, collected_answer)
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 返回空字符串，标记流结束
            yield "", True
        else:
            # 如果无答案，记录信息到日志
            self.logger.info("未找到答案")
            # 计算处理时间
            processing_time = time.time() - start_time
            # 记录处理时间到日志
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            # 一次性返回默认答案，标记为完整
            yield "未找到答案", True



if __name__ == "__main__":
    new_qa_system = IntegratedQASystem()
    # answer = new_qa_system.query(query='什么是AI', session_id="603db0cf-cfa0-4433-9078-f37f3b29fd7c")
    # for value in answer:
    #     print(value)
    results = new_qa_system._fetch_recent_history(session_id="603db0cf-cfa0-4433-9078-f37f3b29fd7c")
    print(results)