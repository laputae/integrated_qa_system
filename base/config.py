# -*- coding:utf-8 -*-
# 导入配置ini文件的解析库
import configparser
import json
# 导入路径操作
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录的绝对路径
current_dir_path = os.path.dirname(current_file_path)
# 获取项目根目录的绝对路径
project_root = os.path.dirname(current_dir_path)

config_file_path = os.path.join(project_root, 'config.ini')

_config_singleton = None


def get_config():
    global _config_singleton
    if _config_singleton is None:
        _config_singleton = Config()
    return _config_singleton


def reset_config():
    """Reset the config singleton (for tests that need fresh config)."""
    global _config_singleton
    _config_singleton = None


class Config():
    """统一配置类，单一来源：config.ini"""

    _initialized = False

    def __new__(cls, *args, **kwargs):
        global _config_singleton
        if _config_singleton is None:
            _config_singleton = super().__new__(cls)
        return _config_singleton

    def __init__(self, config_file=config_file_path):
        if self._initialized:
            return
        self._initialized = True

        # 1. 创建配置文件解析器
        self.config = configparser.ConfigParser()
        # 2. 读取配置文件
        with open(config_file, 'r', encoding='utf-8') as fp:
            self.config.read_file(fp)

        # 3. 获取相关的配置（仅从 config.ini 读取，不再通过环境变量覆盖）

        # 3.1 MySQL 数据库配置
        self.MYSQL_HOST = self.config.get('mysql', 'host', fallback='localhost')
        self.MYSQL_USER = self.config.get('mysql', 'user', fallback='root')
        self.MYSQL_PASSWORD = self.config.get('mysql', 'password', fallback='')
        self.MYSQL_DATABASE = self.config.get('mysql', 'database', fallback='subjects_kg')

        # Redis 配置
        self.REDIS_HOST = self.config.get('redis', 'host', fallback='localhost')
        self.REDIS_PORT = self.config.getint('redis', 'port', fallback=6379)
        self.REDIS_PASSWORD = self.config.get('redis', 'password', fallback='')
        self.REDIS_DB = self.config.getint('redis', 'db', fallback=0)

        # Milvus 配置
        self.MILVUS_HOST = self.config.get('milvus', 'host', fallback='localhost')
        self.MILVUS_PORT = self.config.get('milvus', 'port', fallback='19530')
        self.MILVUS_DATABASE_NAME = self.config.get('milvus', 'database_name', fallback='itcast')
        self.MILVUS_COLLECTION_NAME = self.config.get('milvus', 'collection_name', fallback='edurag_final')
        self.MILVUS_TIMEOUT = self.config.getint('milvus', 'timeout', fallback=10)

        # LLM 配置
        self.LLM_MODEL = self.config.get('llm', 'model', fallback='deepseek-v4-pro')
        self.DASHSCOPE_API_KEY = self.config.get('llm', 'dashscope_api_key', fallback='')
        self.DASHSCOPE_BASE_URL = self.config.get('llm', 'dashscope_base_url', fallback='https://api.deepseek.com')

        # Chunking 策略配置
        self.CHUNK_DEFAULT_STRATEGY = self.config.get('chunking', 'default_strategy', fallback='recursive')
        self.CHUNK_DOC_TYPE_STRATEGIES = self.config.get('chunking', 'doc_type_strategies', fallback='{}')
        self.CHUNK_SEMANTIC_MODEL_PATH = self.config.get('chunking', 'semantic_model_path', fallback='')
        self.CHUNK_SEMANTIC_DEVICE = self.config.get('chunking', 'semantic_device', fallback='cpu')
        self.CHUNK_SEMANTIC_FALLBACK_STRATEGY = self.config.get('chunking', 'semantic_fallback_strategy', fallback='recursive')

        # 检索参数
        self.PARENT_CHUNK_SIZE = self.config.getint('retrieval', 'parent_chunk_size', fallback=1200)
        self.CHILD_CHUNK_SIZE = self.config.getint('retrieval', 'child_chunk_size', fallback=300)
        self.CHUNK_OVERLAP = self.config.getint('retrieval', 'chunk_overlap', fallback=50)
        self.RETRIEVAL_K = self.config.getint('retrieval', 'retrieval_k', fallback=5)
        self.CANDIDATE_M = self.config.getint('retrieval', 'candidate_m', fallback=2)
        self.RETRIEVAL_MAX_WORKERS = self.config.getint('retrieval', 'max_workers', fallback=3)

        # Reranker 分数阈值
        self.RERANKER_SCORE_THRESHOLD = self.config.getfloat('retrieval', 'reranker_score_threshold', fallback=0.3)

        # LLM Reranker 配置
        self.LLM_RERANKER_ENABLED = self.config.getboolean('llm_reranker', 'enabled', fallback=False)
        self.LLM_RERANKER_CRITICAL_MIN_LENGTH = self.config.getint('llm_reranker', 'critical_min_length', fallback=20)
        self.LLM_RERANKER_CRITICAL_STRATEGIES = [
            s.strip() for s in self.config.get(
                'llm_reranker', 'critical_strategies',
                fallback='假设问题检索,回溯问题检索,子查询检索'
            ).split(',') if s.strip()
        ]
        self.LLM_RERANKER_LISTWISE_K = self.config.getint('llm_reranker', 'listwise_k', fallback=3)

        # 查询分类器配置
        self.CLASSIFIER_CONFIDENCE_THRESHOLD = self.config.getfloat('classifier', 'confidence_threshold', fallback=0.8)

        # 策略选择配置
        self.STRATEGY_CACHE_TTL = self.config.getint('strategy', 'cache_ttl', fallback=604800)

        # LLM 重试配置
        self.LLM_MAX_RETRIES = self.config.getint('retry', 'max_retries', fallback=3)
        self.LLM_RETRY_BASE_DELAY = self.config.getfloat('retry', 'base_delay', fallback=1.0)
        self.LLM_RETRY_MAX_DELAY = self.config.getfloat('retry', 'max_delay', fallback=30.0)

        # Embedding 配置
        self.EMBEDDING_MODEL = self.config.get('embedding', 'model', fallback='bge-m3')
        self.EMBEDDING_BATCH_SIZE = self.config.getint('embedding', 'batch_size', fallback=32)
        self.EMBEDDING_CHECKPOINT_DIR = self.config.get('embedding', 'checkpoint_dir', fallback='checkpoints/embedding')
        self.EMBEDDING_CACHE_TTL = self.config.getint('embedding', 'cache_ttl', fallback=86400)

        # /metrics 端点认证配置
        self.METRICS_AUTH_USER = self.config.get('metrics', 'metrics_auth_user', fallback='')
        self.METRICS_AUTH_PASSWORD = self.config.get('metrics', 'metrics_auth_password', fallback='')

        # 安全响应头配置
        self.SECURE_HEADERS_ENABLED = self.config.getboolean('security_headers', 'enabled', fallback=True)

        # 应用配置
        self.CUSTOMER_SERVICE_PHONE = self.config.get('app', 'customer_service_phone', fallback='')
        cors_origins_raw = self.config.get('app', 'cors_origins', fallback='http://localhost:3000,http://127.0.0.1:8000')
        self.CORS_ORIGINS = [o.strip() for o in cors_origins_raw.split(',') if o.strip()]
        valid_sources_raw = self.config.get('app', 'valid_sources', fallback='["ai", "java", "test", "ops", "bigdata"]')
        try:
            self.VALID_SOURCES = json.loads(valid_sources_raw)
        except json.JSONDecodeError:
            self.VALID_SOURCES = ["ai", "java", "test", "ops", "bigdata"]

        # 日志配置
        self.LOG_FILE = self.config.get('logger', 'log_file', fallback='logs/app.log')
        self.LOG_LEVEL = self.config.get('logger', 'log_level', fallback='INFO')
        self.LOG_FORMAT = self.config.get('logger', 'log_format', fallback='json')
        self.LOG_MAX_BYTES = self.config.getint('logger', 'log_max_bytes', fallback=10485760)
        self.LOG_BACKUP_COUNT = self.config.getint('logger', 'log_backup_count', fallback=5)

        # Auth 配置
        self.JWT_SECRET_KEY = self.config.get('auth', 'jwt_secret_key', fallback='')
        self.ACCESS_TOKEN_EXPIRE_MINUTES = self.config.getint('auth', 'access_token_expire_minutes', fallback=30)
        self.REFRESH_TOKEN_EXPIRE_DAYS = self.config.getint('auth', 'refresh_token_expire_days', fallback=7)
        self.BCRYPT_COST_FACTOR = self.config.getint('auth', 'bcrypt_cost_factor', fallback=12)

        # Tenant 配置
        self.DEFAULT_TENANT_NAME = self.config.get('tenant', 'default_tenant_name', fallback='default')

        # Eval 配置
        self.EVAL_LLM_MODEL = self.config.get('eval', 'eval_llm_model', fallback='') or None
        self.EVAL_LLM_BASE_URL = self.config.get('eval', 'eval_llm_base_url', fallback='') or None
        self.EVAL_EMBEDDING_MODEL = self.config.get('eval', 'eval_embedding_model', fallback='mxbai-embed-large')
        self.EVAL_EMBEDDING_BASE_URL = self.config.get('eval', 'eval_embedding_base_url', fallback='http://localhost:11434')
        self.EVAL_INTERVAL_SECONDS = self.config.getint('eval', 'eval_interval_seconds', fallback=86400)
        self.EVAL_REGRESSION_FAITHFULNESS_THRESHOLD = self.config.getfloat('eval', 'regression_faithfulness_threshold', fallback=0.6)
        self.EVAL_REGRESSION_CONSECUTIVE_RUNS = self.config.getint('eval', 'regression_consecutive_runs', fallback=3)
        self.EVAL_QUALITY_WARNING_THRESHOLD = self.config.getfloat('eval', 'quality_warning_threshold', fallback=0.6)
        self.EVAL_QUALITY_CRITICAL_THRESHOLD = self.config.getfloat('eval', 'quality_critical_threshold', fallback=0.4)
        self.EVAL_DEFAULT_DATASET_PATH = self.config.get('eval', 'default_dataset_path', fallback='rag_qa/rag_assesment/rag_evaluate_data.json')

        # Health check 配置
        self.HEALTH_CHECK_TIMEOUT = self.config.getfloat('health', 'check_timeout', fallback=5.0)
        self.HEALTH_CACHE_TTL = self.config.getint('health', 'cache_ttl', fallback=30)
        self.HEALTH_RECOVERY_INTERVAL = self.config.getint('health', 'recovery_interval', fallback=60)
        self.HEALTH_CIRCUIT_BREAKER_THRESHOLD = self.config.getint('health', 'circuit_breaker_threshold', fallback=3)
        self.HEALTH_CIRCUIT_BREAKER_COOLDOWN = self.config.getint('health', 'circuit_breaker_cooldown', fallback=30)

        # HallucinationGuard 配置
        self.HALLUCINATION_GUARD_ENABLED = self.config.getboolean('hallucination_guard', 'enabled', fallback=False)
        self.HALLUCINATION_GUARD_MODEL = self.config.get('hallucination_guard', 'model', fallback='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
        self.HALLUCINATION_GUARD_ENTAILMENT_THRESHOLD = self.config.getfloat('hallucination_guard', 'entailment_threshold', fallback=0.5)
        self.HALLUCINATION_GUARD_CONTRADICTION_THRESHOLD = self.config.getfloat('hallucination_guard', 'contradiction_threshold', fallback=0.5)

        # 并发控制配置
        self.MAX_CONCURRENT_LLM_CALLS = self.config.getint('concurrency', 'max_concurrent_llm_calls', fallback=10)
        self.THREAD_POOL_WORKERS = self.config.getint('concurrency', 'thread_pool_workers', fallback=20)

        # ========== 关键字段校验 ==========
        missing = []
        if not self.JWT_SECRET_KEY:
            missing.append('[auth] jwt_secret_key')
        if not self.DASHSCOPE_API_KEY:
            missing.append('[llm] dashscope_api_key')

        if missing:
            raise ValueError(
                f"config.ini 中以下关键配置项为空，请填写后重试：\n  "
                + "\n  ".join(missing)
                + "\n\n提示：可参考 config.ini.example 模板文件。"
            )


if __name__ == '__main__':
    conf = Config()
    print(f"CHUNK_OVERLAP = {conf.CHUNK_OVERLAP}")
    print(f"VALID_SOURCES = {conf.VALID_SOURCES}")
    print(f"LLM_MODEL = {conf.LLM_MODEL}")
