# ============================================================
# integrated_qa_system — 多阶段 Docker 构建（CPU 版本）
# 构建: docker build -t integrated-qa-system:latest .
# 运行: docker run -p 8000:8000 integrated-qa-system:latest
# ============================================================

# ---- Stage 1: 依赖安装 ----
FROM python:3.11-slim-bookworm AS builder

# 安装编译所需系统库（PaddlePaddle、OpenCV、PyTorch 的依赖）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make cmake \
    libffi-dev libssl-dev \
    libgl1-mesa-glx libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（比 pip 快 10-100x）
RUN pip install --no-cache-dir uv

WORKDIR /app

# 先复制依赖文件以利用 Docker 层缓存
COPY pyproject.toml uv.lock ./

# 安装项目依赖（--no-install-project 因为 pyproject.toml 无 [build-system]）
RUN uv sync --frozen --no-dev --no-install-project

# ---- Stage 2: 运行时 ----
FROM python:3.11-slim-bookworm AS runtime

# 安装运行时系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

WORKDIR /app

# 从 builder 阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制应用源码
COPY app.py main.py config.ini ./
COPY base/ base/
COPY db_models/ db_models/
COPY gateway/ gateway/
COPY mysql_qa/ mysql_qa/
COPY rag_qa/ rag_qa/
COPY repositories/ repositories/
COPY scripts/ scripts/
COPY static/ static/
COPY alembic/ alembic/

# 复制入口脚本
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# 将 venv 加入 PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# 创建运行时目录并设置权限
RUN mkdir -p /app/logs /app/checkpoints/embedding /app/rag_qa/models && \
    chown -R appuser:appuser /app

USER appuser

# 健康检查（复用应用自带的 /health 端点）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
