# EduRAG — 集成智能问答系统

基于 **BM25 + RAG + LLM** 双级检索架构的智能教育问答系统，支持 MySQL 精确匹配与 Milvus 向量语义检索的自动切换，提供流式 WebSocket 和 SSE 接口。

## 架构概览

```
用户提问
  │
  ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI 网关层                      │
│   WebSocket / SSE 流式接口  │  REST API  │  静态前端   │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│               IntegratedQASystem 调度层               │
│         会话历史管理  │  问候语检测  │  双路路由        │
└─────────────────────────────────────────────────────┘
  │
  ├─── Tier 1: BM25 精确匹配 ──────────────────────────┐
  │    MySQL (jpkb 问答表)  │  jieba 分词  │  Redis 缓存 │
  │    相似度 ≥ 0.85 → 直接返回答案                     │
  └────────────────────────────────────────────────────┤
  │
  └─── Tier 2: RAG 语义检索 ──────────────────────────┐
       查询分类(BERT) → 策略选择(LLM) → 混合检索(Milvus)  │
       → 重排序(BGE-Reranker) → LLM 流式生成            │
  └────────────────────────────────────────────────────┘
```

### 双级检索流程

1. **BM25 精确匹配**：用户问题经 jieba 分词后，与 MySQL 中预存问答对进行 BM25 评分，经 softmax 归一化后若最高分 ≥ 0.85，直接返回缓存答案
2. **RAG 语义检索**：BM25 置信度不足时自动回落，由 BERT 分类器判断查询类型，LLM 选择检索策略，Milvus 混合检索（稠密 + 稀疏向量）后经 CrossEncoder 重排序，最终由 LLM 流式生成答案

## 项目结构

```
integrated_qa_system/
├── base/                          # 基础设施
│   ├── config.py                  # 配置管理（读取 config.ini）
│   └── logger.py                  # 日志系统
│
├── db_models/                     # SQLAlchemy 数据模型
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
│   ├── auth.py                    # JWT 签发/校验、密码哈希
│   ├── deps.py                    # FastAPI 依赖注入（get_current_user）
│   ├── middleware.py              # 网关中间件
│   ├── security.py                # 输入安全过滤
│   ├── rate_limiter.py            # 速率限制
│   └── audit.py                   # 审计日志器
│
├── mysql_qa/                      # MySQL QA 模块（Tier 1）
│   ├── main.py                    # MySQLQASystem 独立 CLI
│   ├── db/mysql_client.py         # MySQL 查询（jpkb 表）
│   ├── cache/redis_client.py      # Redis 缓存 + Token 黑名单 + 限流
│   ├── retrieval/bm25_search.py   # BM25Okapi + Softmax 搜索
│   ├── utils/preprocess.py        # jieba 中文分词
│   └── data/                      # 问答对 CSV 数据
│
├── rag_qa/                        # RAG 模块（Tier 2）
│   ├── rag_main.py                # RAG CLI（数据预处理 / 交互查询）
│   ├── core/
│   │   ├── new_rag_system.py      # RAGSystem v2（流式 + 对话历史）
│   │   ├── vector_store.py        # Milvus 向量库 + BGE-M3 混合检索 + 重排序
│   │   ├── query_classifier.py    # BERT 查询分类器（通用/专业）
│   │   ├── strategy_selector.py   # LLM 检索策略选择器
│   │   ├── prompts.py             # LangChain Prompt 模板
│   │   ├── document_processor.py  # 文档加载 + 父子块分割
│   │   └── llamaindex_processor.py # LlamaIndex 文档处理后端
│   ├── edu_document_loaders/      # 自定义文档加载器（含 OCR）
│   ├── edu_text_spliter/          # 中文感知文本分割器
│   ├── rag_assesment/             # RAGAS 质量评估
│   ├── classify_data/             # 分类器训练数据
│   └── models/                    # 本地模型文件
│       ├── bge-m3/                # BGE-M3 嵌入模型
│       ├── bge-reranker-large/    # BGE-Reranker 交叉编码器
│       └── bert-base-chinese/     # BERT 中文基础模型
│
├── scripts/                       # 运维脚本
│   └── seed_default_tenant.py     # 一键建表 + 写入默认租户
│
├── static/                        # Web 前端（HTML/CSS/JS）
├── new_main.py                    # 主调度器（IntegratedQASystem）
├── app.py                         # FastAPI 主入口（WebSocket + REST + 静态服务）
├── api.py                         # FastAPI SSE 流式接口
├── config.ini                     # 全局配置文件
├── pyproject.toml                 # 项目元数据与依赖（uv 管理）
└── logs/                          # 运行日志
```

## 环境要求

- **Python** ≥ 3.11, &lt; 3.13
- **uv**（Python 包管理器，推荐）
- **MySQL** 5.7+（建议 8.0）
- **Redis** 6.0+
- **Milvus** 2.4+（建议使用 Milvus Lite 或 Standalone）
- **GPU**（可选，加速 BERT 分类器、BGE 嵌入、CrossEncoder 推理）

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd integrated_qa_system

# 2. 安装依赖（uv 自动创建虚拟环境）
uv sync

# 3. 下载本地模型（放到 rag_qa/models/ 目录）
#   - BAAI/bge-m3              → rag_qa/models/bge-m3/
#   - BAAI/bge-reranker-large  → rag_qa/models/bge-reranker-large/
#   - google-bert/bert-base-chinese → rag_qa/models/bert-base-chinese/
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

[retrieval]
parent_chunk_size = 1200
child_chunk_size = 300
chunk_overlap = 50
retrieval_k = 5
candidate_m = 2

[app]
valid_sources = ["ai", "java", "test", "ops", "bigdata"]
customer_service_phone = 12345678

[auth]
jwt_secret_key = <your-random-64-char-hex-key>
access_token_expire_minutes = 30
refresh_token_expire_days = 7
bcrypt_cost_factor = 12

[tenant]
default_tenant_name = default

[logger]
log_file = logs/app.log
```

> **安全提示**：生产环境请通过环境变量注入敏感信息（API Key、密码），`config.py` 已支持 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 等环境变量覆盖。

## 数据库初始化

系统使用 **MySQL** 存储结构化数据（6 张表）和 **Milvus** 存储向量数据（1 个 Collection）。

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
    timestamp  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_id (session_id),
    INDEX idx_user_session (user_id, session_id),
    INDEX idx_tenant_session (tenant_id, session_id),
    INDEX idx_conv_tenant (tenant_id),
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
| `users` | 用户注册/登录 | SQLAlchemy ORM（自动建表） |
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

**执行构建**：

```bash
# 使用默认文档目录（rag_qa/data/）
uv run python rag_qa/rag_main.py --data-processing

# 指定自定义文档目录
uv run python rag_qa/rag_main.py --data-processing --data-dir /path/to/your/documents
```

`--data-dir` 默认为 `./data`（相对于项目根目录），即 `rag_qa/data/`。

此命令将：
- 加载各学科文档（MD / PDF / DOCX / PPTX / 图片）
- 通过 OCR 提取图片中的文字
- 执行父子块切分（父块 1200 字符，子块 300 字符）
- 子块写入 Milvus 向量库（BGE-M3 稠密 + 稀疏混合嵌入）

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
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

启动后访问 `http://localhost:8000` 使用 Web 聊天界面。

API 端点：

| 方法 | 路径 | 说明 | 认证 |
|------|------|------|------|
| GET | `/` | 聊天 Web 界面 | 无 |
| POST | `/api/auth/register` | 用户注册 | 无 |
| POST | `/api/auth/login` | 用户登录，返回 JWT | 无 |
| POST | `/api/auth/refresh` | 刷新 Access Token | Refresh Token |
| POST | `/api/auth/logout` | 登出（Token 加入黑名单） | 必须 |
| POST | `/api/create_session` | 创建会话 | 可选 |
| GET | `/api/sessions` | 获取用户会话列表 | 必须 |
| POST | `/api/query` | 非流式查询（BM25 快捷接口） | 可选 |
| WebSocket | `/api/stream` | 流式查询（支持 RAG 流式输出） | Token 参数 |
| GET | `/api/history/{session_id}` | 获取对话历史 | 必须 |
| DELETE | `/api/history/{session_id}` | 清除对话历史 | 必须 |
| GET | `/api/sources` | 获取支持的学科类别 | 无 |
| GET | `/health` | 健康检查 | 无 |

### 方式二：SSE 流式接口

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

POST `/query` 请求体：

```json
{
  "query": "用上下文管理器实现函数运行时间的计算？",
  "source_filter": "ai",
  "session_id": "a1b2c3d4-..."
}
```

### 方式三：命令行交互

```bash
uv run python new_main.py          # 集成问答（BM25 + RAG + 对话历史）
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

### 文本分割

- **ChineseRecursiveTextSplitter**：中文感知的递归分割器，使用 `。！？；，` 等中文标点作为分隔符
- **AliTextSplitter**：基于 ModelScope BERT 文档分割模型的语义分割

### 父子块策略

- **父块**（1200 字符）：保持文档段落完整性，作为 LLM 上下文
- **子块**（300 字符，50 重叠）：写入 Milvus 索引，提高检索精度
- 检索时命中的子块溯源到对应的父块，以完整父块作为 LLM 的输入上下文

### 混合检索与重排序

1. **BGE-M3 嵌入**：同时生成稠密向量（768 维）和稀疏向量（词权重）
2. **加权混合检索**：Milvus 中稠密权重 1.0，稀疏权重 0.7
3. **CrossEncoder 重排序**：BGE-Reranker-Large 对候选文档精排，取 Top-M 作为最终上下文

### 检索策略选择

LLM 根据查询特征自动从四种策略中选取：

| 策略 | 适用场景 |
|------|---------|
| 直接检索 | 查询明确、关键词清晰 |
| HyDE（假设问题） | 查询模糊，先生成假设答案再检索 |
| 子查询检索 | 复杂多问，拆分为子问题分别检索 |
| 回溯检索 | 专业术语问题，扩展别名后检索 |

### 对话历史管理

- 每个会话最多保留最近 **5 轮** 对话
- 历史存储于 MySQL `conversations` 表，通过 `session_id` + `user_id` + `tenant_id` 联合隔离
- 历史以 `[{question, answer}, ...]` 形式注入 RAG 提示词模板

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
- [LangChain](https://www.langchain.com/) — LLM 应用框架
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — OCR 引擎
