# -*- coding:utf-8 -*-
from mysql_qa import MySQLClient, RedisClient, BM25Search
from rag_qa import VectorStore, RAGSystem
from base import logger, Config
from openai import (
    OpenAI, APITimeoutError, APIConnectionError,
    InternalServerError, RateLimitError,
)
import time
import uuid


class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()

        from db_models.base import init_db, SessionLocal, Base, engine
        self.engine = engine
        self.SessionLocal = SessionLocal

        self.redis_client = RedisClient()
        self.mysql_client = MySQLClient(engine=self.engine)
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

        try:
            self.client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL,
            )
        except Exception as e:
            self.logger.error(f"OpenAI 客户端初始化失败: {e}")
            raise

        self.vector_store = VectorStore()
        self.rag_system = RAGSystem(self.vector_store, self.call_dashscope, redis_client=self.redis_client)

        Base.metadata.create_all(self.engine)

    def call_dashscope(self, prompt):
        max_retries = self.config.LLM_MAX_RETRIES
        base_delay = self.config.LLM_RETRY_BASE_DELAY
        max_delay = self.config.LLM_RETRY_MAX_DELAY

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config.LLM_MODEL,
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
            except (APITimeoutError, APIConnectionError,
                    InternalServerError, RateLimitError,
                    ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    self.logger.warning(
                        f"LLM调用失败 (attempt {attempt+1}/{max_retries}): {e}，"
                        f"{delay:.1f}s 后重试..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"LLM调用失败，已达最大重试次数 {max_retries}: {e}")
                    yield f"错误：LLM调用失败 - {e}"
            except Exception as e:
                self.logger.error(f"LLM调用失败（不可重试）: {e}")
                yield f"错误：LLM调用失败 - {e}"
                return

    def _get_conversation_repo(self):
        from repositories.conversation_repo import ConversationRepository
        return ConversationRepository(self.SessionLocal)

    def _fetch_recent_history(self, session_id: str, user_id: int, tenant_id: int):
        repo = self._get_conversation_repo()
        return repo.get_recent_history(session_id, user_id, tenant_id, limit=5)

    def get_session_history(self, session_id: str, user_id: int = 0, tenant_id: int = 0):
        repo = self._get_conversation_repo()
        return repo.get_session_history(session_id, user_id, tenant_id)

    def update_session_history(self, session_id: str, user_id: int, tenant_id: int,
                                question: str, answer: str) -> list:
        repo = self._get_conversation_repo()
        repo.insert(session_id, user_id, tenant_id, question, answer)
        repo.prune_old_records(session_id, user_id, tenant_id, keep=5)
        self.logger.info(f"会话 {session_id} 历史更新成功")
        return repo.get_recent_history(session_id, user_id, tenant_id, limit=5)

    def clear_session_history(self, session_id: str, user_id: int = 0,
                               tenant_id: int = 0) -> bool:
        repo = self._get_conversation_repo()
        return repo.soft_delete_sessions([session_id], user_id, tenant_id) > 0

    def query(self, query, user_id: int = 0, tenant_id: int = 0,
              source_filter=None, session_id=None):
        start_time = time.time()
        self.logger.info(f"处理查询: '{query}' (会话ID: {session_id}, 用户ID: {user_id}, 租户ID: {tenant_id})")
        history = self.get_session_history(session_id, user_id, tenant_id) if session_id else []

        answer, need_rag = self.bm25_search.search(query, threshold=0.85)
        if answer:
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield answer, True
        elif need_rag:
            self.logger.info("无可靠MySQL答案，回退到RAG")
            collected_answer = ""
            for token in self.rag_system.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False
            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, collected_answer)
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "", True
        else:
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True


if __name__ == "__main__":
    new_qa_system = IntegratedQASystem()
    results = new_qa_system._fetch_recent_history(
        session_id="603db0cf-cfa0-4433-9078-f37f3b29fd7c", user_id=1, tenant_id=1
    )
    print(results)
