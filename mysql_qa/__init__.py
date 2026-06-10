import os, sys
current_dir = os.path.abspath(__file__)
mysql_qa_path = os.path.dirname(current_dir)
# print(f'mysql_qa_path--》{mysql_qa_path}')
sys.path.insert(0, mysql_qa_path)
from db.mysql_client import MySQLClient
from cache.redis_client import RedisClient
from retrieval.bm25_search import BM25Search