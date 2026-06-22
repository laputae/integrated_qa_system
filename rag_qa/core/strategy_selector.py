# -*-coding:utf-8-*-
# core/strategy_selector.py 源码
import sys, os
import re
import hashlib
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
# 导入 LangChain 提示模板
from langchain_core.prompts import PromptTemplate
# 导入日志和配置
from base import logger, Config
# 导入 OpenAI
from openai import (
    OpenAI, APITimeoutError, APIConnectionError,
    InternalServerError, RateLimitError,
)
import time

conf = Config()


class RulePreJudge:
    """规则预判器 — 用本地规则快速判断检索策略，仅在把握大时返回策略名，否则返回 None"""

    DIRECT_ENTITIES = re.compile(
        r"学费|多少钱|价格|费用|大纲|课程|学科|课时|讲师|老师|教师|"
        r"AI|Java|Python|大数据|人工智能|前端|后端|测试|运维|"
        r"MySQL|Redis|Milvus|Docker|K8s|Spring|Vue|React"
    )

    # 不含"是什么"——太宽泛，"JAVA的课程大纲是什么"其实是具体实体查询
    ABSTRACT_PATTERNS = re.compile(
        r"什么是|的定义|有哪些种类|有哪些类型|有哪些应用|应用有哪些|"
        r"的分类|的应用场景|的发展趋势|的优缺点|有哪些优势|有哪些特点"
    )

    COMPLEX_HOWTO = re.compile(r"如何|怎么|怎样")

    COMPLEX_TECH_TERMS = re.compile(
        r"部署|实现|优化|调优|配置|搭建|架构|设计|开发|集成|迁移|监控|调试"
    )

    def prejudge(self, query: str) -> str | None:
        q = query.strip()

        # 1. 多问号/多编号 → 子查询检索
        if q.count("？") >= 2 or q.count("?") >= 2:
            logger.info(f"规则预判命中(多问号) → 子查询检索 (查询: '{query}')")
            return "子查询检索"
        if re.search(r"(?:^|\n)\s*\d+[.、）\)]\s*\S", q):
            logger.info(f"规则预判命中(编号分段) → 子查询检索 (查询: '{query}')")
            return "子查询检索"

        # 2. 极短查询 → 直接检索
        if len(q) <= 8:
            logger.info(f"规则预判命中(极短查询) → 直接检索 (查询: '{query}')")
            return "直接检索"

        # 3. 复杂操作提问（如何 + 技术词）→ 回溯问题检索
        if self.COMPLEX_HOWTO.search(q) and self.COMPLEX_TECH_TERMS.search(q):
            logger.info(f"规则预判命中(复杂操作) → 回溯问题检索 (查询: '{query}')")
            return "回溯问题检索"

        # 4. 具体实体查询 → 直接检索（先于抽象判断，避免含实体+抽象词时误判为HyDE）
        if self.DIRECT_ENTITIES.search(q):
            logger.info(f"规则预判命中(具体实体) → 直接检索 (查询: '{query}')")
            return "直接检索"

        # 5. 抽象概念提问 → 假设问题检索
        if self.ABSTRACT_PATTERNS.search(q):
            logger.info(f"规则预判命中(抽象概念) → 假设问题检索 (查询: '{query}')")
            return "假设问题检索"

        return None


class StrategySelector:
    def __init__(self, redis_client=None):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=Config().DASHSCOPE_API_KEY,
                             base_url=Config().DASHSCOPE_BASE_URL)
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()
        # 规则预判器
        self.rule_prejudge = RulePreJudge()
        # Redis 缓存客户端（可选）
        self.redis_client = redis_client

    @staticmethod
    def _hash_query(query: str) -> str:
        return hashlib.md5(query.encode("utf-8")).hexdigest()

    def _cache_get(self, query_hash: str) -> str | None:
        if not self.redis_client:
            return None
        return self.redis_client.get_data(f"strategy:{query_hash}")

    def _cache_set(self, query_hash: str, strategy: str):
        if not self.redis_client:
            return
        self.redis_client.set_data(
            f"strategy:{query_hash}", strategy, ttl=conf.STRATEGY_CACHE_TTL
        )

    def call_dashscope(self, prompt):
        max_retries = conf.LLM_MAX_RETRIES
        base_delay = conf.LLM_RETRY_BASE_DELAY
        max_delay = conf.LLM_RETRY_MAX_DELAY

        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=conf.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个有用的助手。"},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                return completion.choices[0].message.content if completion.choices else "直接检索"
            except (APITimeoutError, APIConnectionError,
                    InternalServerError, RateLimitError,
                    ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"DashScope API 调用失败 (attempt {attempt+1}/{max_retries}): {e}，"
                        f"{delay:.1f}s 后重试..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"DashScope API 调用失败，已达最大重试次数 {max_retries}: {e}")
                    return "直接检索"
            except Exception as e:
                logger.error(f"DashScope API 调用失败（不可重试）: {e}")
                return "直接检索"


    def _get_strategy_prompt(self):
        #   定义类似私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 示例：
                        * 查询：AI 学科学费是多少？
                        * 策略：直接检索
                    * 查询：JAVA的课程大纲是什么？
                        * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象，直接检索效果不佳的问题，例如：
                    * 示例：
                        * 查询：人工智能在教育领域的应用有哪些？
                        * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：比较 Milvus 和 Zilliz Cloud 的优缺点。
                        * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询。可以吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    def select_strategy(self, query):
        # 1. 规则预判
        strategy = self.rule_prejudge.prejudge(query)
        if strategy:
            return strategy

        # 2. Redis 缓存查询
        query_hash = self._hash_query(query)
        cached = self._cache_get(query_hash)
        if cached:
            logger.info(f"策略缓存命中 → {cached} (查询: '{query}')")
            return cached

        # 3. LLM 策略选择
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"LLM 为查询 '{query}' 选择的检索策略：{strategy}")

        # 写入缓存
        self._cache_set(query_hash, strategy)

        return strategy


if __name__ == '__main__':
    ss = StrategySelector()
    # print(f'ss.clinet--->{ss.client}')
    # result = ss.call_dashscope(prompt="你是谁")
    # print(f'result--》{result}')
    ss.select_strategy(query="Mysql数据库能不能支持100w个样本的插入")
