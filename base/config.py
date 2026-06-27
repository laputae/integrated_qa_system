# -*- coding:utf-8 -*-
# 导入配置ini文件的解析库
import configparser
# 导入路径操作
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# print(f'current_file_path--》{current_file_path}')
# 获取当前文件所在目录的绝对路径
current_dir_path = os.path.dirname(current_file_path)
# print(f'current_dir_path--》{current_dir_path}')
# 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir_path)

config_file_path = os.path.join(project_root, 'config.ini')
# print(f'config_file_path--》{config_file_path}')

class Config():
    def __init__(self, config_file=config_file_path):
        # config_file代表配置文件ini的路径
        # 1.创建配置文件解析器
        self.config = configparser.ConfigParser()
        # 2. 读取配置文件
        # self.config.read(config_file)
        with open(config_file, 'r', encoding='utf-8') as fp:
            self.config.read_file(fp)
        # 3. 获取相关的配置
        # 3.1 获取Mysql数据库的配置
        self.MYSQL_HOST = os.environ.get("MYSQL_HOST") or self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = os.environ.get("MYSQL_USER") or self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD") or self.config.get('mysql', 'password', fallback='')
        self.MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE") or self.config.get('mysql', 'database', fallback='subjects_kg')

        # Redis 配置
        self.REDIS_HOST = os.environ.get("REDIS_HOST") or self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = int(os.environ.get("REDIS_PORT") or self.config.get('redis', 'port', fallback='6379'))
        self.REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD") or self.config.get('redis', 'password', fallback='')
        self.REDIS_DB = int(os.environ.get("REDIS_DB") or self.config.get('redis', 'db', fallback='0'))

        # Milvus 配置
        self.MILVUS_HOST = os.environ.get("MILVUS_HOST") or self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = os.environ.get("MILVUS_PORT") or self.config.get('milvus', 'port', fallback='19530')
        self.MILVUS_DATABASE_NAME = os.environ.get("MILVUS_DATABASE_NAME") or self.config.get('milvus', 'database_name', fallback='itcast')
        self.MILVUS_COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION_NAME") or self.config.get('milvus', 'collection_name', fallback='edurag_final')
        self.MILVUS_TIMEOUT = int(os.environ.get("MILVUS_TIMEOUT") or self.config.get('milvus', 'timeout', fallback='10'))

        # LLM 配置
        # LLM 模型名
        self.LLM_MODEL =os.environ.get("DEEPSEEK_MODEL") or self.config.get('llm', 'model', fallback='deepseek-v4-pro')
        # DashScope API 密钥
        self.DASHSCOPE_API_KEY =os.environ.get("DEEPSEEK_API_KEY") or self.config.get('llm', 'dashscope_api_key',fallback='')
        # DashScope API 地址
        self.DASHSCOPE_BASE_URL =os.environ.get("DEEPSEEK_BASE_URL") or self.config.get('llm', 'dashscope_base_url',
                                                  fallback='https://api.deepseek.com')

        # Chunking 策略配置
        self.CHUNK_DEFAULT_STRATEGY = self.config.get('chunking', 'default_strategy', fallback='recursive')
        self.CHUNK_DOC_TYPE_STRATEGIES = self.config.get('chunking', 'doc_type_strategies', fallback='{}')
        self.CHUNK_SEMANTIC_MODEL_PATH = self.config.get('chunking', 'semantic_model_path', fallback='')
        self.CHUNK_SEMANTIC_DEVICE = self.config.get('chunking', 'semantic_device', fallback='cpu')
        self.CHUNK_SEMANTIC_FALLBACK_STRATEGY = self.config.get('chunking', 'semantic_fallback_strategy', fallback='recursive')

        # 检索参数
        # 父块大小
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        # 子块大小
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        # 块重叠大小
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        # 检索返回数量
        self.RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K") or self.config.get('retrieval', 'retrieval_k', fallback='5'))
        # 最终候选数量
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        # 子查询并行检索最大线程数
        self.RETRIEVAL_MAX_WORKERS = self.config.getint('retrieval', 'max_workers', fallback=3)

        # Reranker 分数阈值（低于该分数的文档将被过滤）
        self.RERANKER_SCORE_THRESHOLD = float(
            os.environ.get("RERANKER_SCORE_THRESHOLD")
            or self.config.get('retrieval', 'reranker_score_threshold', fallback='0.3')
        )

        # LLM Reranker 配置
        self.LLM_RERANKER_ENABLED = self.config.getboolean(
            'llm_reranker', 'enabled', fallback=False
        )
        self.LLM_RERANKER_CRITICAL_MIN_LENGTH = self.config.getint(
            'llm_reranker', 'critical_min_length', fallback=20
        )
        self.LLM_RERANKER_CRITICAL_STRATEGIES = [
            s.strip() for s in self.config.get(
                'llm_reranker', 'critical_strategies',
                fallback='假设问题检索,回溯问题检索,子查询检索'
            ).split(',') if s.strip()
        ]
        self.LLM_RERANKER_LISTWISE_K = self.config.getint(
            'llm_reranker', 'listwise_k', fallback=3
        )

        # 查询分类器配置
        self.CLASSIFIER_CONFIDENCE_THRESHOLD = self.config.getfloat(
            'classifier', 'confidence_threshold', fallback=0.8
        )

        # 策略选择配置
        self.STRATEGY_CACHE_TTL = self.config.getint(
            'strategy', 'cache_ttl', fallback=604800
        )

        # LLM 重试配置
        self.LLM_MAX_RETRIES = self.config.getint('retry', 'max_retries', fallback=3)
        self.LLM_RETRY_BASE_DELAY = self.config.getfloat('retry', 'base_delay', fallback=1.0)
        self.LLM_RETRY_MAX_DELAY = self.config.getfloat('retry', 'max_delay', fallback=30.0)

        # Embedding 配置
        self.EMBEDDING_MODEL = self.config.get('embedding', 'model', fallback='bge-m3')
        self.EMBEDDING_BATCH_SIZE = self.config.getint('embedding', 'batch_size', fallback=32)
        self.EMBEDDING_CHECKPOINT_DIR = self.config.get('embedding', 'checkpoint_dir', fallback='checkpoints/embedding')
        self.EMBEDDING_CACHE_TTL = self.config.getint('embedding', 'cache_ttl', fallback=86400)

        # 应用配置
        self.CUSTOMER_SERVICE_PHONE = os.environ.get("CUSTOMER_SERVICE_PHONE") or self.config.get('app', 'customer_service_phone', fallback='')
        self.VALID_SOURCES = eval(self.config.get('app', 'valid_sources', fallback=["ai", "java", "test", "ops", "bigdata"]))
        # 日志配置
        self.LOG_FILE = os.environ.get("LOG_FILE") or self.config.get('logger', 'log_file', fallback='logs/app.log')
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL") or self.config.get('logger', 'log_level', fallback='INFO')
        self.LOG_FORMAT = os.environ.get("LOG_FORMAT") or self.config.get('logger', 'log_format', fallback='json')
        self.LOG_MAX_BYTES = self.config.getint('logger', 'log_max_bytes', fallback=10485760)
        self.LOG_BACKUP_COUNT = self.config.getint('logger', 'log_backup_count', fallback=5)

        # Auth 配置
        self.JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or self.config.get(
            'auth', 'jwt_secret_key', fallback=''
        )
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(
            os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES")
            or self.config.get('auth', 'access_token_expire_minutes', fallback='30')
        )
        self.REFRESH_TOKEN_EXPIRE_DAYS = int(
            os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS")
            or self.config.get('auth', 'refresh_token_expire_days', fallback='7')
        )
        self.BCRYPT_COST_FACTOR = int(
            os.environ.get("BCRYPT_COST_FACTOR")
            or self.config.get('auth', 'bcrypt_cost_factor', fallback='12')
        )

        # Tenant 配置
        self.DEFAULT_TENANT_NAME = self.config.get(
            'tenant', 'default_tenant_name', fallback='default'
        )

        # Eval 配置
        self.EVAL_LLM_MODEL = os.environ.get("EVAL_LLM_MODEL") or self.config.get('eval', 'eval_llm_model', fallback='') or None
        self.EVAL_LLM_BASE_URL = os.environ.get("EVAL_LLM_BASE_URL") or self.config.get('eval', 'eval_llm_base_url', fallback='') or None
        self.EVAL_EMBEDDING_MODEL = os.environ.get("EVAL_EMBEDDING_MODEL") or self.config.get('eval', 'eval_embedding_model', fallback='mxbai-embed-large')
        self.EVAL_EMBEDDING_BASE_URL = os.environ.get("EVAL_EMBEDDING_BASE_URL") or self.config.get('eval', 'eval_embedding_base_url', fallback='http://localhost:11434')
        self.EVAL_INTERVAL_SECONDS = self.config.getint('eval', 'eval_interval_seconds', fallback=86400)
        self.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = self.config.getfloat('eval', 'regression_faithfulness_threshold', fallback=0.6)
        self.EVAL_REGRESSION_CONSECUTIVE_RUNS = self.config.getint('eval', 'regression_consecutive_runs', fallback=3)
        self.EVAL_QUALITY_WARNING_THRESHOLD = self.config.getfloat('eval', 'quality_warning_threshold', fallback=0.6)
        self.EVAL_QUALITY_CRITICAL_THRESHOLD = self.config.getfloat('eval', 'quality_critical_threshold', fallback=0.4)
        self.EVAL_DEFAULT_DATASET_PATH = self.config.get('eval', 'default_dataset_path', fallback='rag_qa/rag_assesment/rag_evaluate_data.json')

        # Health check 配置
        self.HEALTH_CHECK_TIMEOUT = float(
            os.environ.get("HEALTH_CHECK_TIMEOUT")
            or self.config.get('health', 'check_timeout', fallback='5.0')
        )
        self.HEALTH_CACHE_TTL = self.config.getint(
            'health', 'cache_ttl', fallback=30
        )
        self.HEALTH_RECOVERY_INTERVAL = self.config.getint(
            'health', 'recovery_interval', fallback=60
        )
        self.HEALTH_CIRCUIT_BREAKER_THRESHOLD = self.config.getint(
            'health', 'circuit_breaker_threshold', fallback=3
        )
        self.HEALTH_CIRCUIT_BREAKER_COOLDOWN = self.config.getint(
            'health', 'circuit_breaker_cooldown', fallback=30
        )

        # HallucinationGuard 配置
        self.HALLUCINATION_GUARD_ENABLED = self.config.getboolean(
            'hallucination_guard', 'enabled', fallback=False
        )
        self.HALLUCINATION_GUARD_MODEL = self.config.get(
            'hallucination_guard', 'model', fallback='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
        )
        self.HALLUCINATION_GUARD_ENTAILMENT_THRESHOLD = self.config.getfloat(
            'hallucination_guard', 'entailment_threshold', fallback=0.5
        )
        self.HALLUCINATION_GUARD_CONTRADICTION_THRESHOLD = self.config.getfloat(
            'hallucination_guard', 'contradiction_threshold', fallback=0.5
        )

        # 并发控制配置
        self.MAX_CONCURRENT_LLM_CALLS = int(
            os.environ.get("MAX_CONCURRENT_LLM_CALLS")
            or self.config.get('concurrency', 'max_concurrent_llm_calls', fallback='10')
        )
        self.THREAD_POOL_WORKERS = int(
            os.environ.get("THREAD_POOL_WORKERS")
            or self.config.get('concurrency', 'thread_pool_workers', fallback='20')
        )


if __name__ == '__main__':
    conf = Config()
    print(conf.CHUNK_OVERLAP)
    print(conf.VALID_SOURCES)
    print(type(conf.VALID_SOURCES))