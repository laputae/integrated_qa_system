'''
todo: 和之前的rag_system不一样的地方是：生成答案时，考虑了历史对话记录，以及我们大模型输出结果时stream流式输出结果
'''
# -*-coding:utf-8-*-
# core/rag_system.py 源码
import sys, os, hashlib, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
# 获取core文件所在的目录的绝对路径
rag_qa_path = os.path.dirname(current_dir)
# print(f'rag_qa_path--》{rag_qa_path}')
sys.path.insert(0, rag_qa_path)
# 获取根目录文件所在的绝对位置
project_root = os.path.dirname(rag_qa_path)
sys.path.insert(0, project_root)
from prompts import RAGPrompts
#   导入 time 模块，用于计算时间
import time
from base import logger, Config
from base.metrics import qa_llm_rerank_total, qa_llm_rerank_latency_seconds
from query_classifier import QueryClassifier  # 导入查询分类器
from strategy_selector import StrategySelector  # 导入策略选择器
from vector_store import VectorStore  # 导入向量数据库对象

conf = Config()


#   定义 RAGSystem 类，封装 RAG 系统的核心逻辑
class RAGSystem:
    #   初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm, redis_client=None, llm_client=None):
        #   设置向量数据库对象
        self.vector_store = vector_store
        #   设置大语言模型调用函数
        self.llm = llm
        #   获取 RAG 提示模板
        self.rag_prompt = RAGPrompts.rag_prompt()
        #   初始化查询分类器
        classifier_path = os.path.join(rag_qa_path, 'core', 'bert_query_classifier')
        self.query_classifier = QueryClassifier(model_path=classifier_path)
        #   初始化策略选择器
        self.strategy_selector = StrategySelector(redis_client=redis_client, llm_client=llm_client)
        #   Redis 缓存客户端（用于缓存 LLM 生成的中间结果）
        self.redis_client = redis_client
        #   HallucinationGuard — 按需初始化
        self.hallucination_guard = None
        self._last_guard_result = None  # 供 API 层读取最近一次检测结果
        if conf.HALLUCINATION_GUARD_ENABLED:
            try:
                from nli_guard import HallucinationGuard
                self.hallucination_guard = HallucinationGuard(
                    model_name=conf.HALLUCINATION_GUARD_MODEL,
                )
            except Exception as e:
                logger.warning(f"HallucinationGuard 初始化失败 (继续运行): {e}")
        #   定义方法，生成答案

    #   定义类似私有方法，使用回溯问题进行检索 （注意讲义中没有加source_filter参数，这里补齐了）
    def _retrieve_with_backtracking(self, query, source_filter, top_k=None):
        logger.info(f"使用回溯问题策略进行检索 (查询: '{query}')")
        #   获取回溯问题生成的 Prompt 模板
        backtrack_prompt_template = RAGPrompts.backtracking_prompt()

        # 检查 Redis 缓存，避免同一原始查询反复调用 LLM 得出不同回溯问题
        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        cache_key = f"bt:{query_hash}"
        simplified_query = None

        if self.redis_client:
            try:
                simplified_query = self.redis_client.get_data(cache_key)
                if simplified_query:
                    logger.info(f"回溯问题缓存命中: '{simplified_query}' (原始查询: '{query}')")
            except Exception as e:
                logger.warning(f"读取回溯问题缓存失败: {e}")

        try:
            if not simplified_query:
                #   调用大语言模型生成回溯问题
                simplified_query = ''.join(self.llm(backtrack_prompt_template.format(query=query))).strip()
                logger.info(f"生成的回溯问题: '{simplified_query}'")
                # 写入 Redis 缓存
                if self.redis_client:
                    try:
                        self.redis_client.set_data(cache_key, simplified_query, ttl=conf.EMBEDDING_CACHE_TTL)
                    except Exception as e:
                        logger.warning(f"写入回溯问题缓存失败: {e}")

            #   使用回溯问题进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                simplified_query, k=conf.RETRIEVAL_K, source_filter=source_filter,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"回溯问题策略执行失败: {e}")
            return []

    #   定义类似私有方法，使用子查询进行检索（注意讲义中没有加source_filter参数，这里补齐了）
    def _retrieve_with_subqueries(self, query, source_filter, top_k=None):
        logger.info(f"使用子查询策略进行检索 (查询: '{query}')")
        #   获取子查询生成的 Prompt 模板
        subquery_prompt_template = RAGPrompts.subquery_prompt()  # 使用 template 后缀区分

        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        cache_key = f"sq:{query_hash}"
        subqueries = None

        if self.redis_client:
            try:
                subqueries = self.redis_client.get_data(cache_key)
                if subqueries:
                    logger.info(f"子查询缓存命中: {subqueries} (原始查询: '{query}')")
            except Exception as e:
                logger.warning(f"读取子查询缓存失败: {e}")

        try:
            if not subqueries:
                #   调用大语言模型生成子查询列表
                subqueries_text = ''.join(self.llm(subquery_prompt_template.format(query=query))).strip()
                subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]
                logger.info(f"生成的子查询: {subqueries}")
                if self.redis_client:
                    try:
                        self.redis_client.set_data(cache_key, subqueries, ttl=conf.EMBEDDING_CACHE_TTL)
                    except Exception as e:
                        logger.warning(f"写入子查询缓存失败: {e}")
            if not subqueries:
                logger.warning("未能生成有效的子查询")
                return []
            #   初始化空列表，用于存储所有子查询的检索结果
            all_docs = []
            max_workers = min(len(subqueries), conf.RETRIEVAL_MAX_WORKERS)
            sub_k = top_k if top_k is not None else conf.CANDIDATE_M // 2
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_subq = {
                    executor.submit(
                        self.vector_store.hybrid_search_with_rerank,
                        sub_q, sub_k, source_filter
                    ): sub_q for sub_q in subqueries
                }
                for future in as_completed(future_to_subq):
                    sub_q = future_to_subq[future]
                    try:
                        docs = future.result()
                        all_docs.extend(docs)
                        logger.info(f"子查询 '{sub_q}' 检索到 {len(docs)} 个文档")
                    except Exception as e:
                        logger.error(f"子查询 '{sub_q}' 检索失败: {e}")
            # print(f'all_docs-->{len(all_docs)}')
            # print(f'all_docs[0]-->{all_docs[0]}')
            #   对所有检索结果进行去重 (基于对象内存地址，如果 Document 内容相同但对象不同则无法去重)
            #   更可靠的去重方式是基于文档内容或 ID
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}  # 基于内容去重
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"所有子查询共检索到 {len(all_docs)} 个文档, 去重后剩 {len(unique_docs)} 个")
            return unique_docs  # 返回所有唯一文档，让 retrieve_and_merge 处理数量
        except Exception as e:
            logger.error(f'子查询存在错误：{e}')
            return []

    #   定义私有方法，使用假设文档进行检索（HyDE）
    def _retrieve_with_hyde(self, query, source_filter, top_k=None):
        logger.info(f"使用 HyDE 策略进行检索 (查询: '{query}')")
        #   获取假设问题生成的 Prompt 模板
        hyde_prompt_template = RAGPrompts.hyde_prompt()  # 使用 template 后缀区分

        query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()
        cache_key = f"hyde:{query_hash}"
        hypo_answer = None

        if self.redis_client:
            try:
                hypo_answer = self.redis_client.get_data(cache_key)
                if hypo_answer:
                    logger.info(f"HyDE 假设答案缓存命中 (原始查询: '{query}')")
            except Exception as e:
                logger.warning(f"读取 HyDE 缓存失败: {e}")

        #   调用大语言模型生成假设答案
        try:
            if not hypo_answer:
                hypo_answer = ''.join(self.llm(hyde_prompt_template.format(query=query))).strip()
                logger.info(f"HyDE 生成的假设答案: '{hypo_answer}'")
                if self.redis_client:
                    try:
                        self.redis_client.set_data(cache_key, hypo_answer, ttl=conf.EMBEDDING_CACHE_TTL)
                    except Exception as e:
                        logger.warning(f"写入 HyDE 缓存失败: {e}")
            #   使用假设答案进行检索，并返回检索结果
            return self.vector_store.hybrid_search_with_rerank(
                hypo_answer, k=conf.RETRIEVAL_K, source_filter=source_filter,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"HyDE 策略执行失败: {e}")
            return []

    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        #   如果未指定检索策略，则使用策略选择器选择
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 关键查询检测：决定是否启用 LLM 重排序及扩展候选池
        critical = self._is_critical_query(query, strategy)
        effective_top_k = conf.LLM_RERANKER_LISTWISE_K if critical else None

        # 根据检索策略选择不同的检索方式
        ranked_chunks = []  # 初始化
        if strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query, source_filter, top_k=effective_top_k)
        elif strategy == '子查询检索':
            ranked_chunks = self._retrieve_with_subqueries(query, source_filter, top_k=effective_top_k)
        elif strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query, source_filter, top_k=effective_top_k)
        else:
            # 直接检索：
            logger.info(f"使用直接检索策略 (查询: '{query}')")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                query, k=conf.RETRIEVAL_K, source_filter=source_filter,
                top_k=effective_top_k
            )  # 注意 hybrid_search_with_rerank 返回的是 rerank 后的父文档

        logger.info(f"策略 '{strategy}' 检索到 {len(ranked_chunks)} 个候选文档 (可能已是父文档)")

        # LLM 重排序 (仅关键查询 + 候选文档 >= 2)
        if critical and len(ranked_chunks) >= 2:
            ranked_chunks = self._rerank_with_llm(query, ranked_chunks)

        final_context_docs = ranked_chunks[:conf.CANDIDATE_M]
        logger.info(f"最终选取 {len(final_context_docs)} 个文档作为上下文")
        return final_context_docs

    def _check_force_rag_keywords(self, query):
        """领域关键词预检 — 命中则强制走 RAG，不受分类器结果影响"""
        force_rag_patterns = [
            "课程", "大纲", "教案", "讲义", "课件", "实训", "实验",
            "项目", "案例", "作业", "考试", "考核", "认证",
            "培训", "教学", "师资", "老师", "教师", "讲师",
            "架构", "框架", "算法", "模型", "原理",
            "实现", "部署", "优化", "调优",
        ]
        for pattern in force_rag_patterns:
            if pattern in query:
                return True
        return False

    def _is_critical_query(self, query: str, strategy: str) -> bool:
        """判断查询是否需要 LLM 重排序。

        同时满足以下条件时视为"关键查询"：
        1. LLM 重排序功能已在配置中启用
        2. 策略属于配置的复杂策略列表（HyDE / 回溯 / 子查询）
        3. 查询长度达到配置的最低阈值
        """
        if not conf.LLM_RERANKER_ENABLED:
            return False

        if strategy not in conf.LLM_RERANKER_CRITICAL_STRATEGIES:
            return False

        if len(query.strip()) < conf.LLM_RERANKER_CRITICAL_MIN_LENGTH:
            return False

        return True

    def _rerank_with_llm(self, query: str, docs: list) -> list:
        """使用 LLM 对文档列表做 listwise 重排序。

        一次 LLM 调用对所有候选文档排序，失败时静默回退到原顺序。
        """
        if len(docs) < 2:
            return docs

        start_time = time.time()

        # 格式化文档为编号列表，每篇截断 800 字符防止上下文溢出
        doc_lines = []
        for i, doc in enumerate(docs, start=1):
            content_preview = doc.page_content[:800]
            doc_lines.append(f"[{i}] {content_preview}")
        documents_str = "\n---\n".join(doc_lines)

        prompt = RAGPrompts.llm_reranker_prompt().format(
            query=query,
            documents=documents_str,
        )

        try:
            response = ''.join(self.llm(prompt)).strip()

            # 提取最后一个 JSON 数组
            json_match = re.search(r'\[[\d,\s]+\]', response)
            if not json_match:
                logger.warning(
                    f"LLM reranker: 响应中无 JSON 数组，回退。"
                    f"Response: {response[:200]}"
                )
                qa_llm_rerank_total.labels(status="parse_failure").inc()
                return docs

            indices = json.loads(json_match.group())

            # 验证索引有效性
            if not isinstance(indices, list) or len(indices) != len(docs):
                logger.warning(
                    f"LLM reranker: 索引数量不匹配 "
                    f"(got {len(indices)}, expected {len(docs)})，回退"
                )
                qa_llm_rerank_total.labels(status="invalid_indices").inc()
                return docs

            seen = set()
            reordered = []
            for idx in indices:
                zero_idx = int(idx) - 1
                if zero_idx < 0 or zero_idx >= len(docs):
                    logger.warning(
                        f"LLM reranker: 索引 {idx} 越界 [1, {len(docs)}]，回退"
                    )
                    qa_llm_rerank_total.labels(status="out_of_range").inc()
                    return docs
                if zero_idx in seen:
                    logger.warning(
                        f"LLM reranker: 重复索引 {idx}，回退"
                    )
                    qa_llm_rerank_total.labels(status="duplicate_index").inc()
                    return docs
                seen.add(zero_idx)
                reordered.append(docs[zero_idx])

            # 标注 LLM 重排序位置
            for rank, doc in enumerate(reordered):
                doc.metadata["llm_rerank_position"] = rank
                doc.metadata["llm_rerank_source"] = "llm"

            qa_llm_rerank_total.labels(status="success").inc()
            latency = time.time() - start_time
            qa_llm_rerank_latency_seconds.observe(latency)
            logger.info(
                f"LLM reranker 完成: {len(docs)} 文档重排序, "
                f"新顺序: {indices}, 耗时 {latency:.2f}s"
            )
            return reordered

        except json.JSONDecodeError as e:
            logger.warning(f"LLM reranker: JSON 解析错误 ({e})，回退")
            qa_llm_rerank_total.labels(status="json_error").inc()
            return docs
        except Exception as e:
            logger.warning(f"LLM reranker: 异常 ({e})，回退")
            qa_llm_rerank_total.labels(status="error").inc()
            return docs

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
            scores = [
                doc.metadata.get("rerank_score", 0.0)
                for doc in context_docs
                if "rerank_score" in doc.metadata
            ]
            if scores and all(s < threshold for s in scores):
                logger.warning(
                    f"上下文质量检查：所有{len(scores)}个文档的reranker分数均低于阈值({threshold})"
                )
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
        #   记录查询开始时间
        start_time = time.time()
        logger.info(f"开始处理查询: '{query}', 学科过滤: {source_filter}")
        # 验证历史
        if history is not None and not isinstance(history, list):
            logger.warning(f'无效的历史格式：{type(history)},忽略历史')
            history = []
        elif history:
            history = history[-5:] # 严格只取出最近5轮对话
        # 构造历史的上下文：
        history_context = ''
        if history:
            history_context ="\n".join([f"Q:{h['question']}\nA:{h['answer']}" for h in history])
            logger.info(f'使用对话历史：{history_context[:50]}')

        #   判断查询类型（带置信度阈值）
        query_category, confidence = self.query_classifier.predict_with_confidence(query)
        threshold = conf.CLASSIFIER_CONFIDENCE_THRESHOLD
        force_rag = self._check_force_rag_keywords(query) or bool(source_filter)
        logger.info(f"查询分类结果：{query_category} (置信度: {confidence:.4f}, 阈值: {threshold}) (查询: '{query}')")

        skip_rag = (query_category == "通用知识" and confidence >= threshold and not force_rag)
        if skip_rag:
            logger.info(f"查询为通用知识（置信度 {confidence:.4f} >= {threshold}），直接调用 LLM")
            context = ''
            prompt_input = RAGPrompts.general_knowledge_prompt().format(question=query)
            try:
                for token in self.llm(prompt_input):
                    yield token
                process_time = time.time() - start_time
                logger.info(f'LLM通用知识查询处理完成（耗时：{process_time:.2f}s, 查询：{query})')
            except Exception as e:
                logger.error(f'调用LLM失败:{e}')
                yield f'抱歉，处理问题时出错，请你联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}'
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
            #   选择检索策略
            strategy = self.strategy_selector.select_strategy(query)
            context_docs = self.retrieve_and_merge(query, source_filter=source_filter, strategy=strategy)
            if context_docs:
                context = "\n\n".join([doc.page_content for doc in context_docs])
                logger.info(f"构建上下文完成，包含 {len(context_docs)} 个文档块")
            else:
                context = ""
                logger.info("未检索到相关文档，上下文为空")

            # 上下文质量检查：质量不足时直接返回兜底回复，不调用LLM
            if not self._is_context_sufficient(context_docs, context):
                logger.info(f"上下文质量不足，触发fallback回应 (查询: '{query}')")
                yield self._build_fallback_message(query)
                process_time = time.time() - start_time
                logger.info(f'Fallback处理完成（耗时：{process_time:.2f}s, 查询：{query})')
                return

        prompt_input = self.rag_prompt.format(context=context,
                                              question=query,
                                              history=history_context,
                                              phone=conf.CUSTOMER_SERVICE_PHONE,
                                              external_context=external_context or "无")
        try:
            # 流式输出 + 收集完整回答用于 NLI 验证
            collected_tokens = []
            for token in self.llm(prompt_input):
                collected_tokens.append(token)
                yield token
            full_answer = "".join(collected_tokens)

            # 生成后 NLI 幻觉检测（SoftGate：仅记录，不阻断）
            if self.hallucination_guard and context:
                try:
                    self._last_guard_result = self.hallucination_guard.verify(
                        full_answer, context
                    )
                    if self._last_guard_result.is_hallucinated:
                        logger.warning(
                            f"HallucinationGuard 标记 (查询: '{query[:50]}...'): "
                            f"{self._last_guard_result.details}"
                        )
                except Exception as e:
                    self._last_guard_result = None
                    logger.warning(f"HallucinationGuard 验证异常: {e}")
            else:
                self._last_guard_result = None

            process_time = time.time() - start_time
            logger.info(f'LLM查询处理完成（耗时：{process_time:.2f}s, 查询：{query})')
        except Exception as e:
            logger.error(f'调用LLM失败:{e}')
            yield f'抱歉，处理问题时出错，请你联系人工客服：{conf.CUSTOMER_SERVICE_PHONE}'


if __name__ == '__main__':
    vector_store = VectorStore()
    def call_dashscope(prompt):
        client = OpenAI(api_key= Config().DASHSCOPE_API_KEY,
                        base_url=Config().DASHSCOPE_BASE_URL)
        """调用DashScope API生成答案（流式输出）"""
        try:
            # 创建聊天完成请求，启用流式输出
            completion = client.chat.completions.create(
                model= Config().LLM_MODEL,  # 使用配置中的语言模型
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
            logger.error(f"LLM调用失败: {e}")
            # 返回错误信息
            return f"错误：LLM调用失败 - {e}"
    # print(llm(prompt="什么是AI"))
    rag_system = RAGSystem(vector_store, call_dashscope)
    answer = rag_system.generate_answer(query="AI学科的课程大纲内容有什么", source_filter="ai")
    for vlaue in answer:
        print(vlaue)
    # rag_system._retrieve_with_subqueries(query="AI和JAVA的区别是什么？", source_filter="ai")
    # result = rag_system._retrieve_with_hyde(query="AI课程的NLP的技术有哪些?",source_filter="ai")
    # print(result)
    # print(len(result))