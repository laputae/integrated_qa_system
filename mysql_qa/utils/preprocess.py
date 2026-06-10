# utils/preprocess.py
# 导入分词库
import jieba
# 导入日志
import os, sys
# 获取当前文件所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# print(f'current_dir--》{current_dir}')
module_dir = os.path.dirname(current_dir)
# print(f'module_dir--》{module_dir}')
project_root = os.path.dirname(module_dir)
sys.path.insert(0, project_root)
from base import logger

def preprocess_text(text):
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []

if __name__ == '__main__':
    print(preprocess_text(text="黑马程序员"))