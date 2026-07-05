#!/bin/bash
# ============================================================
# integrated_qa_system — Docker 容器入口脚本
# ============================================================
set -e

echo "=== integrated_qa_system starting ==="

# 创建运行时目录
mkdir -p /app/logs /app/checkpoints/embedding /app/rag_qa/models

# 检查模型文件是否就绪（volume 挂载或预下载）
MODEL_MANIFEST="/app/rag_qa/models/.model_manifest.json"
if [ -f "$MODEL_MANIFEST" ]; then
    echo "Model manifest found, models are ready."
else
    echo "Model files not found in /app/rag_qa/models/."
    echo "The app will start but RAG features may be limited until models are downloaded."
    echo "To download models, mount them via volume or run scripts/model_download.py."
fi

echo "Starting uvicorn on 0.0.0.0:8000..."
exec uv run uvicorn app:app --host 0.0.0.0 --port 8000
