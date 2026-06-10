# -*-coding:utf-8-*-
# 导入 MySQL 客户端
from db.mysql_client import MySQLClient
# 导入 Redis 客户端
from cache.redis_client import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search
# 导入日志
from base import logger
# 导入时间库
import time

class MySQLQASystem:
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MySQLClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    def query(self, query):
        # 查询MYSQL系统
        start_time = time.time()
        # 记录查询信息
        self.logger.info(f'处理查询：{query}')
        # 执行BM25的搜索
        answer,  _ = self.bm25_search.search(query, threshold=0.85)
        if answer:
            self.logger.info(f'Mysql答案：{answer}')
        else:
            self.logger.info('SQL中未找到答案，需要调用RAG系统')
            # 设置默认答案
            answer = 'SQL中未找到答案'
        # 计算处理的时间
        process_time = time.time() - start_time
        self.logger.info(f'查询处理的耗时：{process_time:.2f}秒')
        return answer

def main():
    mysql_qa = MySQLQASystem()
    try:
        print('\n欢迎使用 MYSQL 问答系统')
        print("输入查询进行回答，输入 'exit' 退出。")
        while True:
            # 获取用户的输入
            query = input("\n请输入查询：").strip()
            if query.lower() == 'exit':
                logger.info("退出Mysql系统")
                print("再见")
                break
            # 执行查询
            answer = mysql_qa.query(query)
            print(f'\n答案：{answer}')
    except Exception as e:
        logger.error(f'系统错误：{e}')
    finally:
        mysql_qa.mysql_client.close()
if __name__ == '__main__':
    main()

