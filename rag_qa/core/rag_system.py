"""
todo: 和之前的rag_system不一样的地方是：生成答案时，考虑了历史对话记录，以及我们大模型输出结果时stream流式输出结果
"""

# -*-coding:utf-8-*-
# core/rag_system.py 源码
import os
import time

from base import Config, logger

from .llm_reranker import (
    check_force_rag_keywords,
    is_critical_query,
    rerank_with_llm,
)
from .prompts import RAGPrompts
from .query_classifier import QueryClassifier
from .retrieval_strategies import (
    retrieve_with_backtracking,
    retrieve_with_hyde,
    retrieve_with_subqueries,
)
from .strategy_selector import StrategySelector
from .vector_store import VectorStore

current_dir = os.path.dirname(os.path.abspath(__file__))
rag_qa_path = os.path.dirname(current_dir)

conf = Config()


class RAGSystem:
    def __init__(self, vector_store, llm, redis_client=None, llm_client=None):
        self.vector_store = vector_store
        self.llm = llm
        self.rag_prompt = RAGPrompts.rag_prompt()
        classifier_path = os.path.join(rag_qa_path, "core", "bert_query_classifier")
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        self.strategy_selector = StrategySelector(redis_client=redis_client, llm_client=llm_client)
        self.redis_client = redis_client
        self.hallucination_guard = None
        self._last_guard_result = None
        if conf.HALLUCINATION_GUARD_ENABLED:
            try:
                from .nli_guard import HallucinationGuard

                self.hallucination_guard = HallucinationGuard(
                    model_name=conf.HALLUCINATION_GUARD_MODEL,
                )
            except Exception as e:
                logger.warning(f"HallucinationGuard 初始化失败 (继续运行): {e}")

    # ---- Delegated retrieval strategies ----

    def _retrieve_with_backtracking(self, query, source_filter, top_k=None):
        return retrieve_with_backtracking(
            self.vector_store,
            self.llm,
            self.redis_client,
            conf,
            query,
            source_filter=source_filter,
            top_k=top_k,
        )

    def _retrieve_with_subqueries(self, query, source_filter, top_k=None):
        return retrieve_with_subqueries(
            self.vector_store,
            self.llm,
            self.redis_client,
            conf,
            query,
            source_filter=source_filter,
            top_k=top_k,
        )

    def _retrieve_with_hyde(self, query, source_filter, top_k=None):
        return retrieve_with_hyde(
            self.vector_store,
            self.llm,
            self.redis_client,
            conf,
            query,
            source_filter=source_filter,
            top_k=top_k,
        )

    # ---- Delegated LLM reranker ----

    def _rerank_with_llm(self, query, docs):
        return rerank_with_llm(self.llm, query, docs, conf)

    def _is_critical_query(self, query, strategy):
        return is_critical_query(query, strategy, conf)

    def _check_force_rag_keywords(self, query):
        return check_force_rag_keywords(query)

    # ---- Core pipeline ----

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        critical = self._is_critical_query(query, strategy)
        effective_top_k = conf.LLM_RERANKER_LISTWISE_K if critical else None

        ranked_chunks = []
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter, top_k=effective_top_k)
        elif strategy == "子查询检索":
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter, top_k=effective_top_k)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query, source_filter, top_k=effective_top_k)
        else:
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                query,
                k=conf.RETRIEVAL_K,
                source_filter=source_filter,
                top_k=effective_top_k,
            )

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_chunks)} 个候选文档")

        if critical and len(ranked_chunks) >= 2:
            ranked_chunks = self._rerank_with_llm(query, ranked_chunks)

        final_context_docs = ranked_chunks[: conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    def _is_context_sufficient(self, context_docs, context_str) -> bool:
        """检查检索到的上下文质量是否足以回答用户问题。"""
        if not context_docs or len(context_docs) == 0:
            logger.warning("上下文质量检查：未检索到任何文档")
            return False

        if not context_str or not context_str.strip():
            logger.warning("上下文质量检查：上下文内容为空")
            return False

        threshold = conf.RERANKER_SCORE_THRESHOLD
        if threshold > 0.0:
            scores = [doc.metadata.get("rerank_score", 0.0) for doc in context_docs if "rerank_score" in doc.metadata]
            if scores and all(s < threshold for s in scores):
                logger.warning(f"上下文质量检查：所有{len(scores)}个文档的reranker分数均低于阈值({threshold})")
                return False

        return True

    def _build_fallback_message(self, query, reason="insufficient_context"):
        """构建信息不足时的兜底回复。"""
        phone = conf.CUSTOMER_SERVICE_PHONE
        messages = {
            "insufficient_context": (
                f"抱歉，根据现有知识库中的资料，我无法准确回答您的问题「{query}」。"
                f"这可能是因为知识库中缺少相关领域的文档。\n"
                f"建议您：\n"
                f"1. 精简或重新表述您的问题。\n"
                f"2. 联系人工客服获取进一步帮助，电话：{phone}。"
            ),
        }
        return messages.get(reason, messages["insufficient_context"])

    def generate_answer(self, query, source_filter=None, history=None, external_context=None):
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
        if history is not None and not isinstance(history, list):
            logger.warning(f"无效的历史格式：{type(history)},忽略历史")
            history = []
        elif history:
            history = history[-5:]
        history_context = ""
        if history:
            history_context = "\n".join([f"Q:{h['question']}\nA:{h['answer']}" for h in history])
            logger.info(f"使用对话历史：{history_context[:50]}")

        query_category, confidence = self.query_classifier.predict_with_confidence(query)
        threshold = conf.CLASSIFIER_CONFIDENCE_THRESHOLD
        force_rag = self._check_force_rag_keywords(query) or bool(source_filter)
        logger.info(f"查询分类结果：{query_category} (置信度: {confidence:.4f}, 阈值: {threshold}) (查询: '{query}')")

        skip_rag = query_category == "通用知识" and confidence >= threshold and not force_rag
        if skip_rag:
            logger.info(f"查询为通用知识（置信度 {confidence:.4f} >= {threshold}），直接调用 LLM")
            prompt_input = RAGPrompts.general_knowledge_prompt().format(question=query)
            try:
                for token in self.llm(prompt_input):
                    yield token
                process_time = time.time() - start_time
                logger.info(f"LLM通用知识查询处理完成（耗时：{process_time:.2f}s, 查询：{query})")
            except Exception as e:
                logger.error(f"调用LLM失败:{e}")
                yield f"抱歉，处理问题时出错，请你联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"
            return
        else:
            if source_filter:
                logger.info(f"指定了学科过滤 source_filter={source_filter}，强制执行 RAG 流程")
            elif force_rag:
                logger.info("查询命中领域关键词，强制执行 RAG 流程")
            elif query_category == "通用知识" and confidence < threshold:
                logger.info(f"通用知识置信度 {confidence:.4f} 低于阈值 {threshold}，降级为 RAG 流程")
            else:
                logger.info("查询为专业咨询，执行 RAG 流程")
            strategy = self.strategy_selector.select_strategy(query)
            context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)
            if context_docs:
                context = "\n\n".join([doc.page_content for doc in context_docs])
                logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            else:
                context = ""
                logger.info("未检索到相关文档，上下文为空")

            if not self._is_context_sufficient(context_docs, context):
                logger.info(f"上下文质量不足，触发fallback回应 (查询: '{query}')")
                yield self._build_fallback_message(query)
                process_time = time.time() - start_time
                logger.info(f"Fallback处理完成（耗时：{process_time:.2f}s, 查询：{query})")
                return

        prompt_input = self.rag_prompt.format(
            context=context,
            question=query,
            history=history_context,
            phone=conf.CUSTOMER_SERVICE_PHONE,
            external_context=external_context or "无",
        )
        try:
            collected_tokens = []
            for token in self.llm(prompt_input):
                collected_tokens.append(token)
                yield token
            full_answer = "".join(collected_tokens)

            if self.hallucination_guard and context:
                try:
                    self._last_guard_result = self.hallucination_guard.verify(full_answer, context)
                    if self._last_guard_result.is_hallucinated:
                        logger.warning(
                            f"HallucinationGuard 标记 (查询: '{query[:50]}...'): {self._last_guard_result.details}"
                        )
                except Exception as e:
                    self._last_guard_result = None
                    logger.warning(f"HallucinationGuard 验证异常: {e}")
            else:
                self._last_guard_result = None

            process_time = time.time() - start_time
            logger.info(f"LLM查询处理完成（耗时：{process_time:.2f}s, 查询：{query})")
        except Exception as e:
            logger.error(f"调用LLM失败:{e}")
            yield f"抱歉，处理问题时出错，请你联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}"


if __name__ == "__main__":
    vector_store = VectorStore()

    def call_dashscope(prompt):
        from openai import OpenAI

        client = OpenAI(api_key=Config().DASHSCOPE_API_KEY, base_url=Config().DASHSCOPE_BASE_URL)
        try:
            completion = client.chat.completions.create(
                model=Config().LLM_MODEL,
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
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return f"错误：LLM调用失败 - {e}"

    rag_system = RAGSystem(vector_store, call_dashscope)
    answer = rag_system.generate_answer(query="AI学科的课程大纲内容有什么", source_filter="ai")
    for vlaue in answer:
        print(vlaue)
