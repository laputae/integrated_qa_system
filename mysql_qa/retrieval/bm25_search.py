# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
# 导入配置和日志
import sys, os
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
module_dir = os.path.dirname(current_dir)
# print(f'module_dir--》{module_dir}')
sys.path.insert(0, module_dir)
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)

from utils.preprocess import preprocess_text
from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
# 导入日志
from base import logger


class BM25Search:
    def __init__(self, redis_client, mysql_client):
        # 初始化日志
        self.logger = logger
        # 初始化 Redis 客户端
        self.redis_client = redis_client
        # 初始化 MySQL 客户端
        self.mysql_client = mysql_client
        # 初始化 BM25 模型
        self.bm25 = None
        # 初始化问题列表
        self.questions = None
        # 初始化原始问题
        self.original_questions = None
        # 加载数据
        self._load_data()

    def _load_data(self):
        # 加载数据
        original_key = "qa_original_questions"
        tokenized_key = "qa_tokenized_questions"
        # 从redis中获取原始问题（快）
        self.original_questions = self.redis_client.get_data(original_key)
        # print(f'self.original_questions --》{self.original_questions }')
        # 从redis中获取分词后的问题（快）
        tokenized_questions = self.redis_client.get_data(tokenized_key)
        # print(f'self.tokenized_questions --》{tokenized_questions}')
        # 如果 Redis 中没有数据，从 MySQL 加载
        if not self.original_questions or not tokenized_questions:
            # 从Mysql中获取问题
            self.original_questions = self.mysql_client.fetch_questions()
            # print(f'self.original_questions--》{self.original_questions}')
            # 如果mysql中未获得问题，那么给出警告
            if not self.original_questions:
                self.logger.warning("未加载问题")
                return
            # 对问题进行分词
            tokenized_questions = [preprocess_text(q[0])for q in self.original_questions]
            # print(f'tokenized_questions--》{tokenized_questions}')
            # 把原始的问题存储到redis
            self.redis_client.set_data(original_key,  [(q[0]) for q in self.original_questions])
            # 把分词之后的问题存储到redis
            self.redis_client.set_data(tokenized_key, tokenized_questions)

        # 设置问题列表
        self.questions = tokenized_questions
        # 初始化 BM25 模型
        self.bm25 = BM25Okapi(self.questions)
        # 记录 BM25 初始化成功
        self.logger.info("BM25 模型初始化完成")

    def _softmax(self, scores):
        # 计算softmax分数：但是我们对每个score都减去一个最大值，为了防止数据过大，内存爆炸
        exp_scores = np.exp(scores - np.max(scores))
        # 返回归一化分数
        return exp_scores / exp_scores.sum()

    def search(self, query, threshold=0.85):
        # 搜索查询
        if not query or not isinstance(query, str):
            # 记录无效查询
            self.logger.error("无效查询")
            return None, False
        # 检查Redis缓存
        cached_answer = self.redis_client.get_answer(query)
        if cached_answer:
            return cached_answer, False
        try:
            # 分词
            query_tokens = preprocess_text(query)
            # 计算🧑bm25的分数
            scores = self.bm25.get_scores(query_tokens)
            # print(f'scores--》{scores}')
            # 进行分数的归一化
            softmax_score = self._softmax(scores)
            # print(f'softmax_score--》{softmax_score}')
            # 获取最高分对应的索引
            best_idx = softmax_score.argmax()
            # 根据上述的索引获取最高分值
            best_score = softmax_score[best_idx]
            # print(f'best_score--》{best_score}')
            # 检查分数是否超过阈值
            if best_score >= threshold:
                # 获取原始的问题
                original_question = self.original_questions[best_idx]
                # print(f'original_question--》{original_question}')
                # 查数据库获得问题对应的答案
                answer = self.mysql_client.fetch_answer(original_question)
                if answer:
                    # 缓存qa
                    self.redis_client.set_data(f'answer:{query}', answer)
                    # 记录搜索成功
                    self.logger.info(f'搜索成功，Softamx相似度：{best_score:.3f}')
                    return answer, False
            # 记录无可靠答案
            self.logger.info(f"未找到可靠答案，最高 Softmax 相似度: {best_score:.3f}")
            # 返回 None 和 True
            return None, True
        except Exception as e:
            self.logger.error(f'搜索查询失败：{e}')
            return None, True

if __name__ == "__main__":
    redis_client = RedisClient()
    mysql_client = MySQLClient()
    bm25_search = BM25Search(redis_client, mysql_client)
    bm25_search.search(query="VMware安装VMware时显示灰色如何解决")


