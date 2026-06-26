# -*- coding:utf-8 -*-
from mysql_qa import MySQLClient, RedisClient, BM25Search
from rag_qa import VectorStore, RAGSystem
from base import logger, Config
from base.health import SystemHealth, DegradationLevel
from base.metrics import (
    qa_query_total, qa_query_latency_seconds,
    qa_llm_call_total, qa_bm25_hit_total,
)
from openai import (
    OpenAI, APITimeoutError, APIConnectionError,
    InternalServerError, RateLimitError,
)
import time
import uuid
import asyncio


class IntegratedQASystem:
    def __init__(self):
        self.logger = logger
        self.config = Config()
        self._startup_time = time.time()

        # ---- Phase 1: Core infrastructure (must succeed) ----
        from db_models.base import init_db, SessionLocal, Base, engine
        self.engine = engine
        self.SessionLocal = SessionLocal

        # ---- Phase 2: Optional components (graceful on failure) ----
        self.redis_client = self._init_redis()
        self.mysql_client = self._init_mysql()
        self.bm25_search = self._init_bm25()
        self.llm_client = self._init_llm()
        self.vector_store = self._init_vector_store()
        self.rag_system = self._init_rag_system()
        self.eval_service = self._init_eval_service()

        # ---- Phase 3: DB schema (best-effort) ----
        self._init_db_schema()

        # ---- Phase 4: Register health checks ----
        self.health = SystemHealth(self.config, self.logger)
        self._register_health_checks()

        # Report initial degradation level
        level = self.health.get_degradation_level()
        if level == DegradationLevel.LEVEL0_FULL:
            self.logger.info("系统初始化完成，所有组件健康")
        else:
            self.logger.warning(f"系统初始化完成，当前降级等级: {level.name}")

    # ========== Phase 2: Component Initializers ==========

    def _init_redis(self):
        try:
            client = RedisClient()
            self.logger.info("Redis 连接成功")
            return client
        except Exception as e:
            self.logger.warning(f"Redis 初始化失败 (系统将降级运行): {e}")
            return None

    def _init_mysql(self):
        try:
            client = MySQLClient(engine=self.engine)
            self.logger.info("MySQL 客户端初始化成功")
            return client
        except Exception as e:
            self.logger.warning(f"MySQL 客户端初始化失败 (系统将无法提供服务): {e}")
            return None

    def _init_bm25(self):
        if not self.redis_client or not self.mysql_client:
            self.logger.warning("BM25Search 初始化跳过: Redis 或 MySQL 不可用")
            return None
        try:
            bm25 = BM25Search(self.redis_client, self.mysql_client)
            self.logger.info("BM25Search 初始化成功")
            return bm25
        except Exception as e:
            self.logger.warning(f"BM25Search 初始化失败: {e}")
            return None

    def _init_llm(self):
        try:
            client = OpenAI(
                api_key=self.config.DASHSCOPE_API_KEY,
                base_url=self.config.DASHSCOPE_BASE_URL,
            )
            self.logger.info("LLM 客户端初始化成功")
            return client
        except Exception as e:
            self.logger.warning(f"LLM 客户端初始化失败: {e}")
            return None

    def _init_vector_store(self):
        try:
            vs = VectorStore()
            self.logger.info("VectorStore 初始化成功")
            return vs
        except Exception as e:
            self.logger.warning(f"VectorStore 初始化失败 (RAG 降级): {e}")
            return None

    def _init_rag_system(self):
        if not self.vector_store or not self.llm_client:
            self.logger.warning("RAGSystem 初始化跳过: VectorStore 或 LLM 不可用")
            return None
        try:
            rag = RAGSystem(self.vector_store, self.call_dashscope, redis_client=self.redis_client)
            self.logger.info("RAGSystem 初始化成功")
            return rag
        except Exception as e:
            self.logger.warning(f"RAGSystem 初始化失败: {e}")
            return None

    def _init_eval_service(self):
        if not self.rag_system:
            self.logger.warning("EvalService 初始化跳过: RAGSystem 不可用")
            return None
        try:
            from repositories.eval_repo import EvalRepository
            from rag_qa.eval.eval_service import EvalService
            repo = EvalRepository(self.SessionLocal)
            service = EvalService(
                config=self.config, repo=repo,
                rag_system=self.rag_system,
                llm_client=self.llm_client,
                vector_store=self.vector_store,
            )
            self.logger.info("EvalService 初始化成功")
            return service
        except Exception as e:
            self.logger.warning(f"EvalService 初始化失败: {e}")
            return None

    def _init_db_schema(self):
        try:
            from db_models.base import Base
            Base.metadata.create_all(self.engine)
            self.logger.info("数据库 Schema 创建/验证完成")
        except Exception as e:
            self.logger.warning(f"数据库 Schema 创建失败: {e}")

    # ========== Health Check Registration ==========

    def _register_health_checks(self):
        from base.health import HealthChecker
        checker = HealthChecker(self.config, self.logger)

        self.health.register_component("mysql",
            lambda: checker.check_mysql(self.engine))
        self.health.register_component("redis",
            lambda: checker.check_redis(self.redis_client))
        self.health.register_component("milvus",
            lambda: checker.check_milvus(self.vector_store))
        self.health.register_component("llm",
            lambda: checker.check_llm(self.llm_client, self.config))
        self.health.register_component("embedding",
            lambda: checker.check_embedding(self.vector_store))
        self.health.register_component("reranker",
            lambda: checker.check_reranker(self.vector_store))
        self.health.register_component("classifier",
            lambda: checker.check_classifier(self.rag_system))
        self.health.register_component("llm_reranker",
            lambda: checker.check_llm_reranker(self.config))
        self.health.register_component("hallucination_guard",
            lambda: checker.check_hallucination_guard(self.rag_system))
        self.health.register_component("eval_quality",
            lambda: checker.check_eval_quality(self.eval_service))

    # ========== LLM Call ==========

    def call_dashscope(self, prompt):
        if self.llm_client is None:
            self.logger.error("LLM 客户端未初始化")
            yield "错误：LLM服务不可用"
            return

        max_retries = self.config.LLM_MAX_RETRIES
        base_delay = self.config.LLM_RETRY_BASE_DELAY
        max_delay = self.config.LLM_RETRY_MAX_DELAY

        for attempt in range(max_retries):
            try:
                completion = self.llm_client.chat.completions.create(
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
                qa_llm_call_total.labels(status="success").inc()
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
                    qa_llm_call_total.labels(status="retry_exhausted").inc()
                    yield f"错误：LLM调用失败 - {e}"
            except Exception as e:
                self.logger.error(f"LLM调用失败（不可重试）: {e}")
                qa_llm_call_total.labels(status="failure").inc()
                yield f"错误：LLM调用失败 - {e}"
                return

    # ========== Conversation Helpers ==========

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

    # ========== Main Query Pipeline ==========

    def query(self, query, user_id: int = 0, tenant_id: int = 0,
              source_filter=None, session_id=None, external_context=None):
        start_time = time.time()

        # --- Degradation check ---
        level = self.health.get_degradation_level()
        self.logger.info(
            f"处理查询: '{query}' (降级等级: {level.name}, "
            f"会话ID: {session_id}, 用户ID: {user_id}, 租户ID: {tenant_id})"
        )

        if level == DegradationLevel.LEVEL4_NO_MYSQL:
            self.logger.error("MySQL 不可用，拒绝查询")
            yield "系统维护中，暂无法处理查询，请联系管理员。", True
            return

        history = self.get_session_history(session_id, user_id, tenant_id) if session_id else []

        # --- Phase 1: BM25 search ---
        answer = None
        need_rag = False
        if self.bm25_search:
            try:
                answer, need_rag = self.bm25_search.search(query, threshold=0.85)
            except Exception as e:
                self.logger.error(f"BM25 搜索失败: {e}")
                answer, need_rag = None, False
        else:
            # No BM25 at all — go straight to RAG if available
            need_rag = True

        if answer:
            self.logger.info(f"MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, answer)
            qa_bm25_hit_total.inc()
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield answer, True
            return

        # --- Phase 2: BM25 missed. Try RAG if available ---
        if need_rag and self.rag_system and level < DegradationLevel.LEVEL2_NO_MILVUS:
            if level == DegradationLevel.LEVEL3_NO_LLM:
                self.logger.info("LLM 降级中，返回检索到的原始上下文")
                collected_answer = self._degraded_rag_retrieve(query, source_filter)
                processing_time = time.time() - start_time
                source = source_filter or "all"
                qa_query_total.labels(degradation_level=level.name, source=source).inc()
                qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
                yield collected_answer, True
                return

            # Full RAG pipeline (Level 0 or 1)
            self.logger.info("无可靠MySQL答案，回退到RAG")
            collected_answer = ""
            for token in self.rag_system.generate_answer(
                query, source_filter=source_filter, history=history,
                external_context=external_context
            ):
                collected_answer += token
                yield token, False
            # 读取 HallucinationGuard 最近一次检测结果
            self._last_guard_result = getattr(
                self.rag_system, '_last_guard_result', None
            )
            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, collected_answer)
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "", True
        elif need_rag and level >= DegradationLevel.LEVEL2_NO_MILVUS:
            self.logger.info(f"RAG 不可用 (降级等级: {level.name})")
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True
        else:
            self.logger.info("未找到答案")
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
            yield "未找到答案", True

    # ========== Async Query Pipeline (asyncio.Semaphore + to_thread) ==========

    async def aquery(self, query, semaphore: asyncio.Semaphore,
                     user_id: int = 0, tenant_id: int = 0,
                     source_filter=None, session_id=None, external_context=None):
        """Async variant of query() for high-concurrency WebSocket/SSE endpoints.

        - BM25 phase runs inline (fast, non-blocking).
        - Degraded RAG (Level 3, no LLM) runs via asyncio.to_thread.
        - Full RAG acquires semaphore, then streams tokens from a sync thread
          through an asyncio.Queue so the event loop stays free.
        """
        start_time = time.time()

        level = self.health.get_degradation_level()
        self.logger.info(
            f"[async] 处理查询: '{query}' (降级等级: {level.name}, "
            f"会话ID: {session_id}, 用户ID: {user_id}, 租户ID: {tenant_id})"
        )

        if level == DegradationLevel.LEVEL4_NO_MYSQL:
            self.logger.error("[async] MySQL 不可用，拒绝查询")
            yield "系统维护中，暂无法处理查询，请联系管理员。", True
            return

        history = self.get_session_history(session_id, user_id, tenant_id) if session_id else []

        # --- Phase 1: BM25 (inline — lightweight) ---
        answer = None
        need_rag = False
        if self.bm25_search:
            try:
                answer, need_rag = self.bm25_search.search(query, threshold=0.85)
            except Exception as e:
                self.logger.error(f"[async] BM25 搜索失败: {e}")
                answer, need_rag = None, False
        else:
            need_rag = True

        if answer:
            self.logger.info(f"[async] MySQL答案: {answer}")
            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, answer)
            qa_bm25_hit_total.inc()
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            yield answer, True
            return

        # --- Phase 2: RAG (offload to thread) ---
        if need_rag and self.rag_system and level < DegradationLevel.LEVEL2_NO_MILVUS:
            if level == DegradationLevel.LEVEL3_NO_LLM:
                self.logger.info("[async] LLM 降级中，返回检索到的原始上下文")
                loop = asyncio.get_running_loop()
                collected_answer = await loop.run_in_executor(
                    None, self._degraded_rag_retrieve, query, source_filter
                )
                processing_time = time.time() - start_time
                source = source_filter or "all"
                qa_query_total.labels(degradation_level=level.name, source=source).inc()
                qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
                yield collected_answer, True
                return

            # Full RAG pipeline — acquire semaphore, then stream via Queue
            self.logger.info("[async] 无可靠MySQL答案，回退到RAG (async path)")
            queue: asyncio.Queue = asyncio.Queue()

            async with semaphore:
                def _run_rag():
                    try:
                        collected = ""
                        for token in self.rag_system.generate_answer(
                            query, source_filter=source_filter, history=history,
                            external_context=external_context
                        ):
                            collected += token
                            queue.put_nowait(("token", token))
                        self._last_guard_result = getattr(
                            self.rag_system, '_last_guard_result', None
                        )
                        queue.put_nowait(("done", collected))
                    except Exception as e:
                        self.logger.error(f"[async] RAG 线程异常: {e}")
                        queue.put_nowait(("error", str(e)))

                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, _run_rag)

                collected_answer = ""
                while True:
                    msg_type, data = await queue.get()
                    if msg_type == "token":
                        collected_answer += data
                        yield data, False
                    elif msg_type == "done":
                        collected_answer = data  # final collected string
                        break
                    elif msg_type == "error":
                        yield f"抱歉，处理问题时出错，请联系人工客服：{self.config.CUSTOMER_SERVICE_PHONE}", True
                        return

            if session_id:
                self.update_session_history(session_id, user_id, tenant_id, query, collected_answer)
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            self.logger.info(f"[async] 查询处理耗时 {processing_time:.2f}秒")
            yield "", True

        elif need_rag and level >= DegradationLevel.LEVEL2_NO_MILVUS:
            self.logger.info(f"[async] RAG 不可用 (降级等级: {level.name})")
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            yield "未找到答案", True
        else:
            self.logger.info("[async] 未找到答案")
            processing_time = time.time() - start_time
            source = source_filter or "all"
            qa_query_total.labels(degradation_level=level.name, source=source).inc()
            qa_query_latency_seconds.labels(degradation_level=level.name).observe(processing_time)
            yield "未找到答案", True

    def _degraded_rag_retrieve(self, query, source_filter=None) -> str:
        """Level 3 degraded retrieval: return raw context without LLM summarization."""
        try:
            strategy = self.rag_system.strategy_selector.select_strategy(query)
            docs = self.rag_system.retrieve_and_merge(
                query, source_filter=source_filter, strategy=strategy
            )
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
                return (
                    "【系统提示】大语言模型暂不可用，以下为检索到的相关资料：\n\n"
                    f"{context}"
                )
            return "未找到相关答案"
        except Exception as e:
            self.logger.error(f"降级检索失败: {e}")
            return "未找到相关答案"


if __name__ == "__main__":
    new_qa_system = IntegratedQASystem()
    results = new_qa_system._fetch_recent_history(
        session_id="603db0cf-cfa0-4433-9078-f37f3b29fd7c", user_id=1, tenant_id=1
    )
    print(results)
