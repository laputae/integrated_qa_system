# EduRAG — 企业级智能问答系统

基于 **BM25 + RAG + LLM** 双级检索架构的企业级智能教育问答系统，面向生产环境设计，具备多租户隔离、JWT 认证鉴权、多级降级熔断、安全审计、速率限制等完整的企业级基础设施。支持 MySQL 精确匹配与 Milvus 向量语义检索的自动切换，提供流式 WebSocket 和 SSE 接口。

## 企业级特性

| 特性 | 实现 |
|------|------|
| **多租户架构** | `tenant_id` FK 级联隔离，数据完全按租户划分，支持租户启用/禁用 |
| **认证鉴权** | JWT Access Token + Refresh Token 双令牌机制，Redis 黑名单即时失效，bcrypt(cost=12) 密码哈希 |
| **安全防护** | 网关三层中间件：SQL注入/XSS过滤 → 分级速率限制 → JWT校验 + 黑名单检查 |
| **审计追踪** | 全事件审计日志（登录/登出/Token刷新/攻击拦截/限流触发/历史清除），持久化 MySQL |
| **会话管理** | 多轮对话历史持久化，session_id + user_id + tenant_id 三维隔离，自动裁剪保留最近 5 轮 |
| **健康检查** | 7 组件独立健康探测（MySQL/Redis/Milvus/LLM/Embedding/Reranker/Classifier），`/health` `/ready` `/status` 端点 |
| **多级降级** | 5 级自动降级（Level 0~4），熔断器 + 后台自动恢复，依赖故障时优雅降级而非崩溃 |
| **生产韧性** | 数据库连接池（pool_size=10, max_overflow=20, pool_pre_ping, pool_recycle），LLM 指数退避重试，批量嵌入 Checkpoint/Resume 断点续传 |
| **GPU 加速** | PyTorch cu126 (CUDA 12.6) 原生支持；BGE-M3 默认 CPU 运行避免 fp16 类型不匹配，OCR/训练可选用 GPU |

## 架构概览

```
用户提问
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI 网关层                            │
│   WebSocket / SSE 流式  │  REST API  │  JWT认证  │  静态前端    │
│   中间件: 安全过滤 → 速率限制 → JWT校验 → 审计日志              │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                 IntegratedQASystem 调度层                      │
│    会话历史管理  │  问候语检测  │  降级等级判断  │  双路路由    │
│    健康检查编排  │  熔断器管理  │  后台自动恢复                 │
└──────────────────────────────────────────────────────────────┘
  │
  ├─── Tier 1: BM25 精确匹配 ──────────────────────────────────┐
  │    MySQL (jpkb 问答表)  │  jieba 分词  │  Redis 缓存         │
  │    相似度 ≥ 0.85 → 直接返回答案                              │
  └─────────────────────────────────────────────────────────────┤
  │
  └─── Tier 2: RAG 语义检索 ───────────────────────────────────┐
       查询分类(BERT+置信度) → 策略选择(LLM+Redis缓存)           │
       → 子查询并行检索(ThreadPool) → 混合检索(Milvus/BGE-M3)    │
       → 重排序(BGE-Reranker) → 分数阈值过滤 → Few-shot Prompt   │
       → LLM 流式生成(指数退避重试) → 上下文不足时兜底回复        │
       ┌───────────────────────────────────────────────────────┐
       │  嵌入模型注册表 (BGE-M3 / BGE-Large-ZH / Text2Vec)     │
       │  增量文档加载 (SQLite哈希追踪 + LlamaIndex ref_doc_id)  │
       │  批量嵌入 + Checkpoint/Resume                          │
       └───────────────────────────────────────────────────────┘
  └─────────────────────────────────────────────────────────────┘
```

### 双级检索流程

1. **BM25 精确匹配**：用户问题经 jieba 分词后，与 MySQL 中预存问答对进行 BM25 评分，经 softmax 归一化后若最高分 ≥ 0.85，直接返回缓存答案
2. **RAG 语义检索**：BM25 置信度不足时自动回落，由 BERT 分类器判断查询类型并输出置信度——低于阈值或命中领域关键词时强制执行 RAG，LLM 选择检索策略（直接检索 / HyDE / 子查询 / 回溯），子查询以 ThreadPoolExecutor 并行执行，Milvus 混合检索（稠密 + 稀疏向量）后经 CrossEncoder 重排序，低于分数阈值的文档被过滤，最终由 LLM 流式生成答案（指数退避重试）。当检索上下文质量不足时，直接返回兜底回复而不浪费 LLM 调用

### 多级降级体系

系统注册 7 个组件健康检查，按故障影响范围自动计算降级等级：

| 等级 | 名称 | 触发条件 | 行为 |
|------|------|---------|------|
| Level 0 | 全功能 | 全部健康 | 正常运行 |
| Level 1 | 无 Redis | Redis 不可用 | 停用查询缓存、限流计数、Token 黑名单 |
| Level 2 | 无 Milvus | Milvus / Embedding / Reranker / Classifier 不可用 | 仅 BM25，不执行 RAG |
| Level 3 | 无 LLM | LLM 不可用 | BM25 优先，Fallback 时返回原始检索上下文 |
| Level 4 | 无 MySQL | MySQL 不可用 | 返回 503，拒绝所有查询 |

每个组件配备独立熔断器（3 次连续失败 → OPEN → 30s cooldown → HALF_OPEN 探测），后台 asyncio 任务按可配置间隔自动探测故障组件并恢复。

## 项目结构

```
integrated_qa_system/
├── base/                          # 基础设施
│   ├── config.py                  # 配置管理（读取 config.ini）
│   ├── logger.py                  # 日志系统
│   └── health.py                  # 健康检查 + 多级降级 + 熔断器 + 自动恢复
│
├── db_models/                     # SQLAlchemy 数据模型（5 张 ORM 表）
│   ├── base.py                    # 引擎、Session、Base 基类
│   ├── tenant.py                  # Tenant 模型（多租户）
│   ├── user.py                    # User 模型（用户认证）
│   ├── conversation.py            # Conversation 模型（对话历史）
│   ├── refresh_token.py           # RefreshToken 模型（JWT 刷新令牌）
│   └── audit_log.py               # AuditLog 模型（审计日志）
│
├── repositories/                  # 数据访问层
│   ├── tenant_repo.py             # 租户 CRUD
│   ├── user_repo.py               # 用户 CRUD
│   ├── conversation_repo.py       # 会话历史 CRUD
│   └── audit_repo.py              # 审计日志写入
│
├── gateway/                       # API 网关层
│   ├── auth.py                    # JWT 签发/校验（Access + Refresh Token）、密码哈希
│   ├── deps.py                    # FastAPI 依赖注入（get_current_user）
│   ├── middleware.py               # 三层中间件（安全过滤 → 速率限制 → JWT校验）
│   ├── security.py                # SQL注入/XSS 输入安全过滤
│   ├── rate_limiter.py            # 分级速率限制（登录/注册/查询/流式）
│   └── audit.py                   # 审计日志器（事件类型枚举 + 日志写入）
│
├── mysql_qa/                      # Tier 1: BM25 精确匹配
│   ├── main.py                    # MySQLQASystem 独立 CLI
│   ├── db/mysql_client.py         # MySQL 查询（jpkb 表，raw SQL）
│   ├── cache/redis_client.py      # Redis 缓存 + Token 黑名单 + 限流计数
│   ├── retrieval/bm25_search.py   # BM25Okapi + Softmax 归一化
│   └── utils/preprocess.py        # jieba 中文分词
│
├── rag_qa/                        # Tier 2: RAG 语义检索
│   ├── rag_main.py                # RAG CLI（数据预处理 / 交互查询）
│   ├── core/
│   │   ├── rag_system.py          # RAGSystem（流式 + 对话历史 + 上下文质量检查 + 兜底回复）
│   │   ├── vector_store.py        # Milvus 向量库 + BGE-M3 混合检索 + 重排序 + 分数阈值过滤
│   │   ├── embedding_registry.py  # 嵌入模型注册表（多模型 A/B 切换）+ 批量嵌入 + Checkpoint
│   │   ├── llamaindex_processor.py # LlamaIndex 文档处理器（OCR加载 → 切分 → 批量索引）
│   │   ├── ingestion_tracker.py   # SQLite 文件哈希追踪（增量加载：NEW/MODIFIED/UNCHANGED/DELETED）
│   │   ├── document_processor.py  # 文档加载 + 父子块分割（传统管线）
│   │   ├── query_classifier.py    # BERT 查询分类器（通用/专业 + 置信度评分）
│   │   ├── strategy_selector.py   # LLM 检索策略选择器（Redis 缓存，7 天 TTL）
│   │   └── prompts.py             # LangChain Prompt 模板（Few-shot + external_context）
│   ├── edu_document_loaders/      # 自定义文档加载器（含 OCR）
│   ├── edu_text_spliter/          # 中文感知文本分割器
│   ├── rag_assesment/             # RAGAS 质量评估
│   ├── classify_data/             # 分类器训练数据
│   └── models/                    # 本地模型文件
│       ├── bge-m3/                # BGE-M3 嵌入模型（稠密1024维 + 稀疏）
│       ├── bge-reranker-large/    # BGE-Reranker 交叉编码器
│       ├── bert-base-chinese/     # BERT 中文基础模型
│       ├── bge-large-zh/          # BGE-Large-ZH 嵌入模型（1024维，可选）
│       └── text2vec-large-chinese/ # Text2Vec 嵌入模型（768维，可选）
│
├── scripts/                       # 运维脚本
│   ├── seed_default_tenant.py     # 一键建表 + 写入默认租户
│   └── migrate_add_is_deleted.py  # 迁移脚本：conversations 表新增 is_deleted 字段
│
├── static/                        # Web 前端（React JSX）
├── tests/                         # 测试
│   ├── test_document_quality.py   # 文档质量评估冒烟测试
│   └── test_strategy_cache.py     # 策略选择缓存测试
├── main.py                        # 主调度器（IntegratedQASystem + 降级编排）
├── app.py                         # FastAPI 主入口（WebSocket + REST + 静态服务 + 健康端点）
├── api.py                         # FastAPI SSE 流式接口
├── config.ini                     # 全局配置文件
├── pyproject.toml                 # 项目元数据与依赖（uv 管理）
└── logs/                          # 运行日志
```

## 环境要求

- **Python** ≥ 3.11, < 3.13
- **uv**（Python 包管理器，推荐）
- **MySQL** 5.7+（建议 8.0）
- **Redis** 6.0+
- **Milvus** 2.4+（建议使用 Milvus Lite 或 Standalone）
- **GPU**（可选，CUDA 12.6+，加速 BERT 分类器、BGE 嵌入、CrossEncoder 推理）
- **CUDA**：PyTorch 使用 cu126 索引（`https://download.pytorch.org/whl/cu126`），内置 CUDA 12.6 运行时

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd integrated_qa_system

# 2. 安装依赖（uv 自动创建虚拟环境，自动从 cu126 索引安装 torch）
uv sync

# 3. 下载本地模型（放到 rag_qa/models/ 目录）
#   - BAAI/bge-m3                → rag_qa/models/bge-m3/
#   - BAAI/bge-reranker-large    → rag_qa/models/bge-reranker-large/
#   - google-bert/bert-base-chinese → rag_qa/models/bert-base-chinese/
#   - (可选) BAAI/bge-large-zh    → rag_qa/models/bge-large-zh/
#   - (可选) shibing624/text2vec-large-chinese → rag_qa/models/text2vec-large-chinese/
```

## 配置

编辑项目根目录下 `config.ini`：

```ini
[mysql]
host = 127.0.0.1
user = root
password = 123456
database = subjects_kg

[redis]
host = 127.0.0.1
port = 6379
password = 1234
db = 0

[milvus]
host = 127.0.0.1
port = 19530
database_name = itcast
collection_name = edurag_final

[llm]
model = deepseek-v4-pro
dashscope_api_key =           # 你的 API Key（也支持环境变量 DEEPSEEK_API_KEY）
dashscope_base_url = https://api.deepseek.com

[embedding]
model = bge-m3                # 可选: bge-m3 | bge-large-zh | text2vec-large-chinese
batch_size = 32
checkpoint_dir = checkpoints/embedding
cache_ttl = 86400

[classifier]
confidence_threshold = 0.8         # BERT 分类置信度低于此值时强制走 RAG

[strategy]
cache_ttl = 604800                 # 检索策略缓存 TTL（秒），默认 7 天

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 5
candidate_m = 2
max_workers = 3                    # 子查询并行检索最大线程数
reranker_score_threshold = 0.3     # Reranker 分数阈值，低于此分数的文档被过滤

[retry]
max_retries = 3                    # LLM 调用最大重试次数
base_delay = 1.0                   # 指数退避基础延迟（秒）
max_delay = 30.0                   # 指数退避最大延迟（秒）

[auth]
jwt_secret_key = <your-random-64-char-hex-key>
access_token_expire_minutes = 30
refresh_token_expire_days = 7
bcrypt_cost_factor = 12

[tenant]
default_tenant_name = default

[health]
check_timeout = 5.0                # 单次健康检查超时（秒）
cache_ttl = 30                     # 健康状态缓存 TTL（秒）
recovery_interval = 60             # 后台自动恢复探测间隔（秒）
circuit_breaker_threshold = 3      # 熔断器连续失败阈值
circuit_breaker_cooldown = 30      # 熔断器冷却时间（秒）

[app]
valid_sources = ["ai", "java", "test", "ops", "bigdata"]
customer_service_phone = 12345678

[logger]
log_file = logs/app.log
```

> **安全提示**：生产环境请通过环境变量注入敏感信息（API Key、密码），`config.py` 已支持 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`JWT_SECRET_KEY` 等环境变量覆盖。

## 数据库初始化

系统使用 **MySQL** 存储结构化数据（6 张表）和 **Milvus** 存储向量数据（1 个 Collection），外加 **SQLite** 追踪文件摄入状态。

| 存储 | 用途 | 管理方式 |
|------|------|----------|
| MySQL | 业务数据（租户/用户/会话/令牌/审计） + BM25 问答对 | 5 ORM + 1 raw SQL |
| Milvus | 文档向量（稠密 + 稀疏混合嵌入） | pymilvus / LlamaIndex |
| SQLite | 文件摄入追踪（哈希、状态、块计数） | IngestionTracker |

### 创建 MySQL 数据库

```sql
CREATE DATABASE IF NOT EXISTS subjects_kg DEFAULT CHARACTER SET utf8mb4;
USE subjects_kg;
```

### 方式一：自动建表（推荐）

启动应用时，`IntegratedQASystem` 初始化会自动调用 `Base.metadata.create_all(engine)` 创建所有 SQLAlchemy 管理的表（tenants / users / conversations / refresh_tokens / audit_logs），并运行 seed 脚本写入默认租户：

```bash
# 1. 先创建数据库（见上方 SQL）
# 2. 运行 seed 脚本，自动建表 + 写入默认租户
uv run python scripts/seed_default_tenant.py
```

`jpkb` 问答表不走 ORM，需要手动创建（见下方方式二）。

### 方式二：手动建表

完整的 SQL 建表语句如下：

```sql
-- ===================== 业务表 =====================

-- 1. 问答对表（BM25 精确匹配数据源）
CREATE TABLE IF NOT EXISTS jpkb (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    subject_name VARCHAR(20)  NOT NULL COMMENT '学科名称',
    question     VARCHAR(1000) NOT NULL COMMENT '问题文本',
    answer       VARCHAR(1000) NOT NULL COMMENT '答案文本'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ===================== 多租户 & 认证表（SQLAlchemy ORM 管理） =====================

-- 2. 租户表
CREATE TABLE IF NOT EXISTS tenants (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    name       VARCHAR(100) NOT NULL COMMENT '租户名称',
    is_active  BOOLEAN      NOT NULL DEFAULT TRUE COMMENT '是否启用',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_tenant_name (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3. 用户表
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id     INT          NOT NULL DEFAULT 1 COMMENT '所属租户',
    username      VARCHAR(50)  NOT NULL COMMENT '用户名',
    password_hash VARCHAR(255) NOT NULL COMMENT 'bcrypt 密码哈希',
    is_active     BOOLEAN      NOT NULL DEFAULT TRUE COMMENT '是否激活',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_user_tenant (username, tenant_id),
    INDEX idx_users_tenant (tenant_id),
    CONSTRAINT fk_users_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4. 会话历史表
CREATE TABLE IF NOT EXISTS conversations (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id  INT          NOT NULL DEFAULT 1 COMMENT '所属租户',
    user_id    INT          NOT NULL COMMENT '用户ID',
    session_id VARCHAR(36)  NOT NULL COMMENT '会话UUID',
    question   TEXT         NOT NULL COMMENT '用户问题',
    answer     TEXT         NOT NULL COMMENT '系统回答',
    is_deleted BOOLEAN      NOT NULL DEFAULT FALSE COMMENT '逻辑删除标记',
    timestamp  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_user_session (user_id, session_id),
    INDEX idx_tenant_session (tenant_id, session_id),
    INDEX idx_conv_tenant (tenant_id),
    INDEX idx_is_deleted (is_deleted),
    CONSTRAINT fk_conv_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT fk_conv_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5. Refresh Token 表
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id   INT          NOT NULL DEFAULT 1 COMMENT '所属租户',
    user_id     INT          NOT NULL COMMENT '用户ID',
    token_jti   VARCHAR(36)  NOT NULL COMMENT 'JWT JTI 唯一标识',
    device_info VARCHAR(255) NULL COMMENT '设备信息',
    expires_at  DATETIME     NOT NULL COMMENT '过期时间',
    revoked     BOOLEAN      NOT NULL DEFAULT FALSE COMMENT '是否已撤销',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_token_jti (token_jti),
    INDEX idx_rt_user (user_id),
    INDEX idx_rt_tenant (tenant_id),
    CONSTRAINT fk_rt_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT fk_rt_user FOREIGN KEY (user_id) REFERENCES users(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6. 审计日志表
CREATE TABLE IF NOT EXISTS audit_logs (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    tenant_id  INT          NULL COMMENT '所属租户',
    user_id    INT          NULL COMMENT '用户ID',
    event_type VARCHAR(50)  NOT NULL COMMENT '事件类型',
    ip_address VARCHAR(45)  NULL COMMENT '来源IP',
    user_agent VARCHAR(500) NULL COMMENT 'User-Agent',
    detail     TEXT         NULL COMMENT '事件详情JSON',
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_created (created_at),
    INDEX idx_audit_user (user_id),
    INDEX idx_tenant_event (tenant_id, event_type),
    INDEX idx_audit_tenant (tenant_id),
    CONSTRAINT fk_audit_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7. 写入默认租户（必须，否则用户注册/登录会失败）
INSERT INTO tenants (name) VALUES ('default')
ON DUPLICATE KEY UPDATE name = name;
```

### 各表用途说明

| 表名 | 用途 | 管理方式 |
|------|------|----------|
| `jpkb` | BM25 精确匹配问答对 | 手动 SQL / CSV 导入 |
| `tenants` | 多租户隔离 | SQLAlchemy ORM（自动建表） |
| `users` | 用户注册/登录（bcrypt 密码哈希） | SQLAlchemy ORM（自动建表） |
| `conversations` | 对话历史（按 session_id + user_id + tenant_id 隔离） | SQLAlchemy ORM（自动建表） |
| `refresh_tokens` | JWT Refresh Token 持久化 | SQLAlchemy ORM（自动建表） |
| `audit_logs` | 用户操作审计日志 | SQLAlchemy ORM（自动建表） |

### 导入 BM25 问答数据

将 CSV 数据导入 `jpkb` 表：

```sql
-- CSV 格式：subject_name, question, answer
LOAD DATA LOCAL INFILE 'mysql_qa/data/JP学科知识问答.csv'
INTO TABLE jpkb
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(subject_name, question, answer);
```

或者用 Python 脚本导入：

```python
import csv
from db_models.base import SessionLocal, engine
from sqlalchemy import text

with open('mysql_qa/data/JP学科知识问答.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = [(r['学科名称'], r['问题'], r['答案']) for r in reader]

with SessionLocal() as session:
    for subject, question, answer in rows:
        session.execute(
            text("INSERT INTO jpkb (subject_name, question, answer) VALUES (:s, :q, :a)"),
            {"s": subject, "q": question, "a": answer}
        )
    session.commit()
```

### 构建 RAG 向量库

**文档目录结构**：

处理脚本会遍历 `config.ini` 中 `valid_sources` 配置的每个学科（如 `ai`、`java`、`test`、`ops`、`bigdata`），在 `--data-dir` 指定的目录下查找 `<source>_data` 子目录。默认目录结构如下：

```
rag_qa/data/
├── ai_data/          # AI 学科文档
│   ├── xxx.pdf
│   ├── xxx.docx
│   └── xxx.md
├── java_data/        # Java 学科文档
│   └── xxx.pptx
├── test_data/        # 测试学科文档
├── ops_data/         # 运维学科文档
└── bigdata_data/     # 大数据学科文档
```

**方式一：传统全量构建**（使用 pymilvus 传统管线）：

```bash
uv run python rag_qa/rag_main.py --data-processing --data-dir rag_qa/data
```

此命令将：
- 加载各学科文档（MD / PDF / DOCX / PPTX / 图片）
- 通过 OCR 提取图片中的文字
- 执行父子块切分（父块 1200 字符，子块 300 字符）
- 子块写入 Milvus 向量库（BGE-M3 稠密 + 稀疏混合嵌入）

**方式二：增量构建**（使用 LlamaIndex + SQLite 哈希追踪，推荐重复使用）：

```python
from rag_qa.core.llamaindex_processor import incremental_process_and_index

# 首次运行 — 全部文件标记为 NEW，执行全量索引
result = incremental_process_and_index("rag_qa/data/ai_data")
# => {"new": 50, "modified": 0, "deleted": 0, "unchanged": 0, "total_chunks": 300}

# 第二次运行 — 未修改文件标记为 UNCHANGED，跳过处理
result = incremental_process_and_index("rag_qa/data/ai_data")
# => {"new": 0, "modified": 0, "deleted": 0, "unchanged": 50, "total_chunks": 0}
```

增量管线自动分类文件为 NEW / MODIFIED / UNCHANGED / DELETED：
- **NEW** → OCR + 切分 + 插入新向量
- **MODIFIED** → 先删除旧向量（通过 `ref_doc_id`），再重新处理
- **UNCHANGED** → 直接跳过，零开销
- **DELETED** → 从 Milvus 中删除对应向量

## 首次启动

```bash
# 1. 安装依赖
uv sync

# 2. 编辑 config.ini，填写 MySQL / Redis / Milvus 连接信息和 LLM API Key

# 3. 创建 MySQL 数据库
#    在 MySQL 中执行：CREATE DATABASE IF NOT EXISTS subjects_kg DEFAULT CHARACTER SET utf8mb4;

# 4. 一键建表 + 写入默认租户
uv run python scripts/seed_default_tenant.py

# 5. 手动创建 jpkb 表并导入问答数据（参考上方「数据库初始化」→「方式二」的 SQL）

# 6. 构建 RAG 向量库（参考上方「构建 RAG 向量库」）
uv run python rag_qa/rag_main.py --data-processing
```

## 启动服务

### 方式一：Web 全功能模式（推荐）

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

启动后访问 `http://localhost:8000` 使用 Web 聊天界面。

API 端点：

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/` | 聊天 Web 界面 | 无 |
| GET | `/health` | Liveness 探针（进程存活） | 无 |
| GET | `/ready` | Readiness 探针（可服务状态 + 降级等级） | 无 |
| GET | `/status` | 详细状态（7 组件健康 + 运行时长） | 无 |
| POST | `/api/auth/register` | 用户注册 | 无 |
| POST | `/api/auth/login` | 用户登录，返回 JWT | 无 |
| POST | `/api/auth/refresh` | 刷新 Access Token | Refresh Token |
| POST | `/api/auth/logout` | 登出（Token 加入黑名单） | 必须 |
| POST | `/api/create_session` | 创建会话 | 可选 |
| GET | `/api/sessions` | 获取用户会话列表 | 必须 |
| POST | `/api/query` | 非流式查询（BM25 快捷接口，含降级检查） | 可选 |
| WebSocket | `/api/stream` | 流式查询（支持 RAG 流式输出 + external_context） | Token 参数 |
| GET | `/api/history/{session_id}` | 获取对话历史 | 必须 |
| POST | `/api/history/delete` | 批量删除对话历史（逻辑删除） | 必须 |
| GET | `/api/sources` | 获取支持的学科类别 | 无 |

### 方式二：SSE 流式接口

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

POST `/query` 请求体：

```json
{
  "query": "用上下文管理器实现函数运行时间的计算？",
  "source_filter": "ai",
  "session_id": "a1b2c3d4-...",
  "external_context": "由上游编排层注入的 Function Calling 结果（可选）"
}
```

### 方式三：命令行交互

```bash
uv run python main.py               # 集成问答（BM25 + RAG + 对话历史）
uv run python mysql_qa/main.py     # MySQL BM25 独立问答
uv run python rag_qa/rag_main.py   # RAG 独立问答
```

## 核心技术说明

### 文档处理与 OCR

自定义文档加载器支持多种格式，对图片型文档自动调用 OCR：

| 加载器 | 支持格式 | 说明 |
|--------|---------|------|
| OCRPDFLoader | `.pdf` | PyMuPDF 提取文本 + 图片 OCR |
| OCRDOCLoader | `.docx` | 段落/表格提取 + 图片 OCR |
| OCRPPTLoader | `.ppt` `.pptx` | 文本框架/表格提取 + 图片 OCR |
| OCRIMGLoader | `.png` `.jpg` | 纯图片 OCR |

OCR 引擎：优先使用 RapidOCR Paddle（GPU 加速），fallback 为 RapidOCR ONNX Runtime（CPU）。

### 文档质量评估 — 数据治理

`estimate_document_quality()` 对 OCR 清洗后的文本进行三级评分（0-1），作为企业数据治理的质量门槛：

- **文本长度充足度**（权重 0.30）：评估文档段落是否达到有效检索的最小长度
- **有效字符占比**（权重 0.40）：中文字符 + 拉丁字母 + 数字的实际占比
- **OCR 噪音伪影**（权重 0.30）：检测连续重复字符、非标准符号、行结构异常

评分 < 0.3 的文档标记为 `is_low_quality`，可在后续管线中过滤或降权处理，避免低质量 OCR 文档污染检索结果。

### 文本分割

- **ChineseRecursiveTextSplitter**：中文感知的递归分割器，使用 `。！？；，` 等中文标点作为分隔符
- **MarkdownTextSplitter**：对 `.md` 文件自动切换 Markdown 感知分割
- **AliTextSplitter**：基于 ModelScope BERT 文档分割模型的语义分割

### 父子块策略

- **父块**（1200 字符）：保持文档段落完整性，作为 LLM 上下文
- **子块**（300 字符，50 重叠）：写入 Milvus 索引，提高检索精度
- 检索时命中的子块溯源到对应的父块，以完整父块作为 LLM 的输入上下文

### 嵌入模型注册表

通过 `embedding_registry.py` 支持多嵌入模型 A/B 切换，在 `config.ini` 的 `[embedding] model` 中指定：

| 模型 | 维度 | 稀疏向量 | 后端 |
|------|------|---------|------|
| `bge-m3` | 1024 | 支持 | milvus_model (BGEM3EmbeddingFunction) + LlamaIndex |
| `bge-large-zh` | 1024 | 不支持 | sentence-transformers + LlamaIndex |
| `text2vec-large-chinese` | 768 | 不支持 | sentence-transformers + LlamaIndex |

模型统一通过 `create_milvus_model()` / `create_llamaindex_model()` 工厂方法创建，调用方无需关心具体后端差异。

### 批量嵌入与 Checkpoint/Resume — 生产韧性

`batch_embed()` 提供企业级批量嵌入能力，应对大规模文档处理和运行中断：

- **tqdm 进度条**：实时显示 batch 处理进度，便于运维监控
- **Checkpoint 断点续传**：每完成一个 batch 原子写入 JSON checkpoint，进程崩溃或 OOM Kill 后设置 `resume=True` 从断点继续，已完成的 batch 跳过不重复计算
- **原子写入**：先写 `.tmp` 再 `os.replace`，杜绝 checkpoint 损坏导致的进度丢失
- **自动清理**：全部 batch 完成后自动删除 checkpoint，避免残留
- **异常隔离**：单个 batch 失败抛出明确错误并保留 checkpoint，修复后可精确续传

**生产场景**：百万级文档嵌入耗时数小时，进程意外退出后重启即可从断点恢复，零浪费。

### 增量文档加载（IngestionTracker）— 生产级效率

`IngestionTracker` 使用 SQLite 追踪每个文件的摄入状态，实现企业级增量处理——避免每次全量重建：

- **内容哈希**：SHA-256 流式计算（64KB 分块），精准检测文件变更
- **ref_doc_id**：从归一化路径派生的稳定 ID，支持 LlamaIndex 文档级原子删除/更新
- **状态管理**：active（已索引）/ deleted（已删除），软删除保留审计线索
- **WAL 模式**：SQLite WAL 日志 + NORMAL 同步，兼顾写入性能和崩溃恢复

增量管线（`incremental_process_and_index`）：
1. 扫描目录 → SHA-256 对比 SQLite → 四类分类
2. DELETED → 通过 `ref_doc_id` 从 Milvus 原子删除
3. MODIFIED → 先删旧块，避免僵尸向量
4. NEW + MODIFIED → OCR 加载 → 文本清洗 → 质量评估 → 父子块切分 → LlamaIndex 批量插入
5. UNCHANGED → 直接跳过，零计算开销
6. 更新 SQLite 追踪记录（单个事务）

**生产收益**：大型文档库（10K+ 文件）增量模式下仅处理变更文件，处理时间从小时级降至分钟级。

### 混合检索与重排序

1. **BGE-M3 嵌入**：同时生成稠密向量（1024 维）和稀疏向量（词权重）
2. **加权混合检索**：Milvus 中稠密权重 1.0，稀疏权重 0.7
3. **CrossEncoder 重排序**：BGE-Reranker-Large 对候选文档精排，取 Top-M 作为最终上下文
4. **分数阈值过滤**：低于 `reranker_score_threshold`（默认 0.3）的文档被丢弃，避免低质量上下文污染 LLM 生成

### 检索策略选择

LLM 根据查询特征自动从四种策略中选取（结果以 Redis 缓存 7 天，避免重复调用 LLM）：

| 策略 | 适用场景 | Redis 缓存键 |
|------|---------|-------------|
| 直接检索 | 查询明确、关键词清晰 | — |
| HyDE（假设答案） | 查询模糊，先生成假设答案再检索 | `hyde:{md5}` |
| 子查询检索 | 复杂多问，拆分为子问题并行检索 | `sq:{md5}` |
| 回溯检索 | 专业术语问题，扩展别名后检索 | `bt:{md5}` |

**子查询并行检索**：使用 `ThreadPoolExecutor`（最大线程数由 `max_workers` 控制，默认 3）并发执行各子查询的混合检索+重排序，所有结果汇总后基于内容去重。

### LLM 指数退避重试

LLM 调用失败时自动重试（`[retry]` 配置），延迟按指数增长：`base_delay * 2^attempt`，上限 `max_delay`。可重试异常包括：`APITimeoutError`、`APIConnectionError`、`InternalServerError`、`RateLimitError`、`ConnectionError`、`TimeoutError`。

### Few-shot Prompt 与兜底回复

Prompt 模板升级为 Few-shot 格式，包含「正常回答」和「无法回答」两个示例，引导 LLM 遵循以下规则：

- 严格基于提供的文档回答，必须标注来源
- 上下文为空或不相关时，**不编造答案**，回复兜底消息："信息不足，无法回答，请联系人工客服，电话：{phone}"
- 检索后额外执行上下文质量检查：文档数为 0、内容为空、所有 reranker 分数均低于阈值时，**直接返回兜底回复而不调用 LLM**，节省 API 成本

### external_context — 上游 Function Calling 注入

RAG 接口支持 `external_context` 参数，允许上游编排层（如 LangGraph Agent）将 Function Calling 的返回结果注入 Prompt 模板的 `{external_context}` 占位符。当编排层已通过工具调用获取到外部数据时，无需依赖文档检索即可丰富 LLM 上下文。

### 网关安全体系（企业级防护）

三层中间件（`gateway/middleware.py`）对所有 `/api/` 请求依次执行，构建纵深防御：

```
请求 → Layer 1: SecurityFilter → Layer 2: RateLimiter → Layer 3: AuthMiddleware → 业务层
               │                        │                       │
               ├─ SQL注入检测           ├─ 登录: IP限流         ├─ Bearer Token 提取
               ├─ XSS 攻击检测          ├─ 注册: IP限流         ├─ Token 黑名单检查 (Redis)
               ├─ 恶意请求体拦截        ├─ 查询: 用户+租户限流   ├─ JWT 签名校验
               └─ 审计日志记录          ├─ 流式: 用户+租户限流   └─ 用户信息注入 request.scope
                                        └─ 审计日志记录
```

**白名单路径**（跳过认证）：`/api/auth/login`、`/api/auth/register`、`/api/auth/refresh`、`/health`、`/ready`、`/status`、`/api/sources`

**审计事件覆盖**（`gateway/audit.py`）：

| 事件类型 | 触发场景 |
|---------|---------|
| `LOGIN_SUCCESS` / `LOGIN_FAILED` | 用户登录成功/失败 |
| `REGISTER_SUCCESS` | 用户注册成功 |
| `TOKEN_REFRESH` / `LOGOUT` | Token 刷新 / 用户登出 |
| `SQL_INJECTION_ATTEMPT` / `XSS_ATTEMPT` | 安全过滤器拦截 |
| `RATE_LIMIT_EXCEEDED` | 速率限制触发 |
| `UNAUTHORIZED_ACCESS` | 未认证访问受保护接口 |
| `HISTORY_DELETED` | 用户批量删除对话历史 |

每条审计日志携带 `tenant_id`、`user_id`、`ip_address`、`user_agent`、`event_type`、`detail`(JSON)，满足企业合规审计要求。

### 对话历史管理

- 每个会话最多保留最近 **5 轮** 对话，自动裁剪（`prune_old_records`），防止历史膨胀
- 历史存储于 MySQL `conversations` 表，通过 `session_id` + `user_id` + `tenant_id` 三维隔离，杜绝跨用户/跨租户数据泄露
- 历史以 `[{question, answer}, ...]` 形式注入 RAG 提示词模板，保持多轮上下文的连贯可追溯
- **逻辑删除**：`is_deleted` 字段标记删除，查询自动过滤；前端支持多选 + 全选批量删除，`POST /api/history/delete` 接收 `{session_ids: [...]}`

## 评估

```bash
cd rag_qa/rag_assesment
uv run python rag_as.py
```

使用 RAGAS 框架评估四个指标：**Faithfulness**、**Answer Relevancy**、**Context Precision**、**Context Recall**。

## License

待定

## 致谢

- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — 多语言混合嵌入模型
- [BGE-Reranker](https://huggingface.co/BAAI/bge-reranker-large) — 交叉编码器重排序
- [Milvus](https://milvus.io/) — 向量数据库
- [LlamaIndex](https://www.llamaindex.ai/) — 数据索引框架
- [LangChain](https://www.langchain.com/) — LLM 应用框架
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR 引擎
