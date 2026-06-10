import os, sys
current_dir = os.path.abspath(__file__)
rag_qa_path = os.path.dirname(current_dir)
sys.path.insert(0, rag_qa_path)
from core.vector_store import VectorStore
# from core.rag_system import RAGSystem # 没有添加历史记录和流式输出时选择
from core.new_rag_system import RAGSystem # 添加历史记录和流式输出时选择
